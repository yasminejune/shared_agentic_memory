"""Amin et al. (2024) Algorithm 1 — private prediction over token logits.

Verbatim implementation of Algorithm 1 from
`Amin et al. 2024 <https://arxiv.org/abs/2407.12108>`_ with no
algorithmic deviation. For one batch of sensitive texts ``S`` and the
matching public prompt, :func:`generate` samples a synthetic token
sequence ``x`` under (epsilon, delta) differential privacy governed
by the Theorem 1 bound, returning the decoded string and a
:class:`PrivacyAccount` record.

Both the sensitive prompts and the public prompt are built from a
template in :mod:`agent_memories.agent.privacy.prompts` via a
wrapping function supplied by the caller (``wrap_fn``; defaults to
the round-2 :func:`prompts.wrap` so existing round-2 call sites
continue to work unchanged, but the round-1 caller in
``scripts/amin_et_al/WP2_8.py`` passes :func:`prompts.wrap_label`
instead). Any additional keyword arguments accepted by the chosen
``wrap_fn`` (``label=...`` for round 2, ``k=...`` for round 1) are
forwarded transparently via ``**wrap_kwargs``. Each wrapped prompt
is then tokenised once via :func:`token_generation.encode_chat` so
the Gemma 2 IT chat template is applied (the model is loaded as
``google/gemma-2-2b-it`` per WP2-plan §5; instruction-tuned + chat
template is the project standard, see WP2-plan §5 and §11
deviation 8). The ``s`` sensitive prompts and the single public
prompt are stacked into one ``s + 1`` batch and run through one
padded batched prefill via :func:`token_generation.prefill_padded`,
which both tokenises and forward-passes them together and returns a
:class:`token_generation.PrefillState` carrying the shared KV cache.
Each subsequent sampling step then issues a single ``(B, 1)``
continuation via :func:`token_generation.continue_batched` rather
than re-running the full prompt through every transformer layer: the
prompt prefix is neither re-tokenised nor re-attended-to. The public
prompt is built by calling the same ``wrap_fn`` with no ``items``
argument, so the items block defaults to the literal
``"(no examples)"`` (WP2-plan §3.4) and the rest of the template
aligns token-for-token with the private branch, ensuring only
content tokens consume privacy budget.

Steps per iteration (paper Algorithm 1 lines 9 to 22):

1. Read the logit batch ``Z = {logits(p . x) : p in S}`` and the
   public logits ``z_public = logits(p_public . x)`` off the running
   ``(s + 1, vocab)`` logits tensor (the first ``s`` rows are ``Z``,
   the last row is ``z_public``); both are kept in sync by feeding
   the same sampled ``x`` token to every row of the cached batch.
2. Estimate the L1 distance between the per-batch softmax average and
   the public softmax, add ``Laplace(2 * sigma)`` noise.
3. If the noisy distance meets the noisy threshold ``theta_hat`` the
   distributions are far enough apart that the public prompt is
   *unsafe*: sample a private token via the exponential mechanism
   (clip-recenter then ``softmax(z_bar / tau)``), spend one private
   token from the budget ``r``, refresh ``theta_hat``.
4. Otherwise the public prompt is safe: sample from
   ``softmax(z_public / tau_public)`` at no privacy cost.
5. Append the sampled token to every row of the batch and call
   :func:`token_generation.continue_batched` to extend the shared
   KV cache by one position; the next iteration's logits come back
   from that single ``(s + 1, 1)`` forward pass.

The single change relative to the paper's free-form algorithm is that
this implementation handles a single batch (the project's standalone
WP2.3 demo only has one); the per-batch ``for`` loop on the paper's
line 6 is elided.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    wrap_fn: Callable[..., str] = wrap,
    **wrap_kwargs: Any,
) -> tuple[str, PrivacyAccount]:
    """Run Amin et al. Algorithm 1 on one batch and return the synthetic
    string together with its privacy account.

    ``texts`` are pre-rendered memory-items blocks, one per batch
    member; each string drops into the ``{items}`` slot of whichever
    template ``wrap_fn`` formats. ``wrap_fn`` defaults to the round-2
    :func:`prompts.wrap` (WP2-plan §3.3 content-only template), so
    callers that previously passed ``label=...`` continue to work
    unchanged. The round-1 caller in ``scripts/amin_et_al/WP2_8.py``
    passes ``wrap_fn=prompts.wrap_label`` together with ``k=...``,
    and any other template-specific keyword arguments are forwarded
    through ``**wrap_kwargs``. The public prompt is built by calling
    ``wrap_fn(**wrap_kwargs)`` with no ``items`` argument so the
    items block defaults to the literal ``"(no examples)"``
    (WP2-plan §3.4), aligning format tokens between the public and
    private branches.

    Each wrapped prompt is tokenised once via :func:`tg.encode_chat`
    so the Gemma 2 IT chat template is applied; the ``s`` sensitive
    prompts and the public prompt are then stacked into one
    ``s + 1`` batch and forwarded together via
    :func:`tg.prefill_padded`, which returns the per-row next-token
    logits and a :class:`tg.PrefillState` carrying the shared KV cache.
    Each later sampling step issues a single ``(s + 1, 1)``
    continuation via :func:`tg.continue_batched`, so the prompt
    prefix is neither re-tokenised nor re-attended-to, and the
    sensitive and public branches advance in lockstep on the same
    sampled ``x_ids`` suffix.

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
        assert target_epsilon is not None  # XOR check above
        r = solve_r(target_epsilon, delta, s=s, c=c, tau=tau, sigma=sigma)

    stop = tg.stop_ids()  # eos plus <end_of_turn> for the IT regime
    x_ids: list[int] = []  # Token sequence using token ids
    t = 0  # number of private tokens used
    n_public = 0  # number of public tokens used
    theta_hat = theta + _laplace(sigma)  # noisy threshold

    prompts = [wrap_fn(items=text, **wrap_kwargs) for text in texts]
    public_prompt = wrap_fn(**wrap_kwargs)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)

    # Stack the s sensitive prompts and the single public prompt into one
    # batch of size s + 1; one padded batched prefill builds the shared
    # KV cache so every later token costs just one (B, 1) continuation
    # rather than s + 1 full-prompt forward passes.
    logits, state = tg.prefill_padded(prompt_ids + [public_ids])
    while t < r and len(x_ids) < max_total_tokens:
        Z = logits[: len(prompt_ids)]
        z_public = logits[len(prompt_ids)]

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

        # Same sampled token is appended to every row of the batch, so the
        # public branch tracks the private branch's running suffix verbatim
        # (matches the original [ids + x_ids] / [public_ids + x_ids] coupling).
        logits, state = tg.continue_batched(state, [tok] * (len(prompt_ids) + 1))

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
