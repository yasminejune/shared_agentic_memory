"""Amin et al. (2024) Algorithm 1 — private prediction over token logits.

Verbatim implementation of Algorithm 1 from
`Amin et al. 2024 <https://arxiv.org/abs/2407.12108>`_ with no
algorithmic deviation. For one batch of sensitive texts ``S`` and the
matching public prompt, :func:`generate` samples a synthetic token
sequence ``x`` under (epsilon, delta) differential privacy governed
by the Theorem 1 bound, returning the decoded string and a
:class:`PrivacyAccount` record.

Both the sensitive prompts and the public prompt are built from the
single :data:`agent_memories.agent.privacy.prompts.GENERIC_PROMPT`
template via :func:`agent_memories.agent.privacy.prompts.wrap`, then
tokenised once via :func:`token_generation.encode_chat` so each
prompt is wrapped in the Gemma 2 IT chat template (the model is
loaded as ``google/gemma-2-2b-it`` per WP2-plan §5; instruction-tuned
+ chat template is the project standard, see WP2-plan §5 and §11
deviation 8). The per-step loop appends the running token ids
``x_ids`` directly to each pre-tokenised prompt and feeds the
concatenation to :func:`token_generation.get_next_token_logits_from_ids`,
so the unchanging prompt prefix is never re-tokenised and the
running suffix is never decoded then re-encoded. Callers pass
pre-rendered memory-items blocks (one per batch member) and the
round-1 DP-released label string; the public prompt is the same
template with the items block replaced by the literal
``"(no examples)"`` (WP2-plan §3.4) and the same label in the same
position, which is public input to this round by post-processing and
incurs no additional privacy cost.

Steps per iteration (paper Algorithm 1 lines 9 to 22):

1. Build the logit batch ``Z = {logits(p . x) : p in S}`` and the
   public logits ``z_public = logits(p_public . x)``.
2. Estimate the L1 distance between the per-batch softmax average and
   the public softmax, add ``Laplace(2 * sigma)`` noise.
3. If the noisy distance meets the noisy threshold ``theta_hat`` the
   distributions are far enough apart that the public prompt is
   *unsafe*: sample a private token via the exponential mechanism
   (clip-recenter then ``softmax(z_bar / tau)``), spend one private
   token from the budget ``r``, refresh ``theta_hat``.
4. Otherwise the public prompt is safe: sample from
   ``softmax(z_public / tau_public)`` at no privacy cost.

The single change relative to the paper's free-form algorithm is that
this implementation handles a single batch (the project's standalone
WP2.3 demo only has one); the per-batch ``for`` loop on the paper's
line 6 is elided.
"""

from __future__ import annotations

import numpy as np
import torch

from . import token_generation as tg
from .privacy_accounting import PrivacyAccount, epsilon_from_rho, rho_for, solve_r
from .prompts import wrap


def clip_recenter(Z: torch.Tensor, c: float) -> torch.Tensor:
    """Amin et al. Equation 1.

    ``clip_c(z)_i = max(-c, z_i - max_j(z_j) + c)``. The shift by
    ``-max_j(z_j) + c`` puts the row maximum exactly at ``c``; the
    ``max(-c, .)`` then enforces the lower bound. The result lies in
    ``[-c, c]`` componentwise.

    ``Z`` has shape ``(batch, vocab)``; the row maximum is taken per
    prompt.
    """
    max_per_row = Z.max(dim=-1, keepdim=True).values
    shifted = Z - max_per_row + c
    return torch.clamp(shifted, min=-c)


def softmax_l1_distance(Z: torch.Tensor, z_public: torch.Tensor, s: int) -> float:
    """Amin et al. Equation 2.

    ``d(Z, z_public) = || (1/s) * sum_{z in Z} softmax(z)
                          - softmax(z_public) ||_1``

    The divisor is the *expected* batch size ``s`` and not the actual
    number of rows in ``Z`` (matches WP2-plan Section 3.1 and is what
    the privacy proof bounds).
    """
    p_batch = torch.softmax(Z, dim=-1)
    p_public = torch.softmax(z_public, dim=-1)
    p_avg = p_batch.sum(dim=0) / s
    return float(torch.norm(p_avg - p_public, p=1).item())


def sample_private(Z: torch.Tensor, c: float, tau: float, s: int) -> int:
    """Exponential-mechanism token sample from the clipped batch average.

    Returns one token id sampled from
    ``softmax((1/s) * sum clip_c(z) / tau)``. The divisor ``s`` is the
    expected batch size.
    """
    Z_clipped = clip_recenter(Z, c)
    z_bar = Z_clipped.sum(dim=0) / s
    probs = torch.softmax(z_bar / tau, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def sample_public(z_public: torch.Tensor, tau_public: float) -> int:
    """Standard softmax sample from the public-prompt logits."""
    probs = torch.softmax(z_public / tau_public, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _laplace(scale: float) -> float:
    return float(np.random.laplace(0.0, scale))


def generate(
    texts: list[str],
    *,
    label: str,
    s: int,
    c: float,
    tau: float,
    tau_public: float,
    sigma: float,
    theta: float,
    r: int | None = None,
    delta: float | None = None,
    target_epsilon: float | None = None,
    max_total_tokens: int = 256,
) -> tuple[str, PrivacyAccount]:
    """Run Amin et al. Algorithm 1 on one batch and return the synthetic
    string together with its privacy account.

    ``texts`` are pre-rendered memory-items blocks, one per batch
    member (each string drops into the ``{items}`` slot of
    :data:`~agent_memories.agent.privacy.prompts.GENERIC_PROMPT` per
    WP2-plan §3.3). ``label`` is the round-1 DP-released label
    string; it is public input to this round by post-processing and
    appears in the same position in both the batch prompts and the
    public prompt (``wrap(label=label)``), so the caller never
    composes prompt boilerplate themselves. Each wrapped prompt is
    tokenised once via :func:`tg.encode_chat` so the Gemma 2 IT chat
    template is applied; the per-step loop appends sampled token ids
    directly to each pre-tokenised prompt and feeds the concatenation
    to :func:`tg.get_next_token_logits_from_ids`, so the unchanging
    prompt prefix is never re-tokenised and the running suffix is
    never decoded then re-encoded.

    ``delta`` defaults to ``1 / s`` (the Amin Appendix C convention
    used by :func:`get_epsilon`); pass it explicitly to override.

    Exactly one of ``r`` and ``target_epsilon`` must be supplied: pass
    ``r`` to spend a fixed private-token budget and let the realised
    epsilon fall out of Theorem 1, or pass ``target_epsilon`` to have
    :func:`solve_r` pick the largest ``r`` whose realised epsilon fits
    inside the budget (capped at the ``solve_r`` default ``r_max=80``
    per WP2-plan §5). Passing both, or neither, raises ``ValueError``.
    """
    if delta is None:
        delta = 1.0 / s
    if (r is None) == (target_epsilon is None):
        raise ValueError(
            "Pass exactly one of `r` or `target_epsilon`; got "
            f"r={r!r}, target_epsilon={target_epsilon!r}."
        )
    if r is None:
        r = solve_r(target_epsilon, delta, s=s, c=c, tau=tau, sigma=sigma)

    stop = tg.stop_ids()  # eos plus <end_of_turn> for the IT regime
    x_ids: list[int] = []  # Token sequence using token ids
    t = 0  # number of private tokens used
    n_public = 0  # number of public tokens used
    theta_hat = theta + _laplace(sigma)  # noisy threshold

    prompts = [wrap(items=text, label=label) for text in texts]
    public_prompt = wrap(label=label)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)

    while t < r and len(x_ids) < max_total_tokens:
        Z = tg.get_next_token_logits_from_ids([ids + x_ids for ids in prompt_ids])
        z_public = tg.get_next_token_logits_from_ids([public_ids + x_ids])[0]

        d_hat = softmax_l1_distance(Z, z_public, s) + _laplace(2.0 * sigma)

        if d_hat >= theta_hat:
            tok = sample_private(Z, c, tau, s)
            t += 1
            theta_hat = theta + _laplace(sigma)
        else:
            tok = sample_public(z_public, tau_public)
            n_public += 1

        x_ids.append(tok)
        if tok in stop:
            break

    rho = rho_for(r, s, c, tau, sigma)
    account = PrivacyAccount(
        epsilon=epsilon_from_rho(rho, delta),
        delta=delta,
        rho=rho,
        r=r,
        s=s,
        c=c,
        tau=tau,
        sigma=sigma,
        private_tokens_used=t,
        public_tokens_used=n_public,
    )
    return tg.decode(x_ids), account
