"""Amin et al. (2024) Algorithm 1: private next-token sampling.

Clips and averages private logits, compares them to a public prompt via a
noisy L1 test, then samples either a private token (exponential mechanism)
or a free public token. Returns the decoded string and a PrivacyAccount.

Earlier private-prediction path; the deployed mechanism is InvisibleInk
in agent/invisible_ink. Templates and the Gemma wrapper are shared.
Averaging uses expected batch size s. The public prompt is the same
wrap_fn with "(no examples)".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ..lm import token_generation as tg
from ..lm.prompts import wrap
from .accounting import PrivacyAccount, epsilon_from_rho, rho_for, solve_r


def clip_recenter(Z: torch.Tensor, c: float) -> torch.Tensor:  # noqa: N803
    """Amin et al. Equation 1: ``clip_c(z)_i = max(-c, z_i - max_j(z_j) + c)``.

    Shifts the row max to ``c`` then clamps below at ``-c``, so each
    component lies in ``[-c, c]``. ``Z`` has shape ``(batch, vocab)``.
    """
    max_per_row = Z.max(dim=-1, keepdim=True).values
    shifted = Z - max_per_row + c
    return torch.clamp(shifted, min=-c)


def softmax_l1_distance(Z: torch.Tensor, z_public: torch.Tensor, s: int) -> float:  # noqa: N803
    """Amin et al. Equation 2: softmax L1 distance.

    ``d(Z, z_public) = || (1/s) * sum_{z in Z} softmax(z)
                          - softmax(z_public) ||_1``

    The divisor is the expected batch size ``s``, not the actual row
    count of ``Z``. That is the quantity the privacy proof bounds.
    """
    p_batch = torch.softmax(Z, dim=-1)
    p_public = torch.softmax(z_public, dim=-1)
    p_avg = p_batch.sum(dim=0) / s
    return float(torch.norm(p_avg - p_public, p=1).item())


def sample_private(Z: torch.Tensor, c: float, tau: float, s: int) -> int:  # noqa: N803
    """Exponential-mechanism sample from the clipped batch average.

    Draws one token from ``softmax((1/s) * sum clip_c(z) / tau)``.
    ``s`` is the expected batch size, not the row count of ``Z``.
    """
    Z_clipped = clip_recenter(Z, c)  # noqa: N806
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
    """Run Amin et al. Algorithm 1 on one batch.

    ``texts`` are pre-rendered items blocks; ``wrap_fn`` (default
    ``prompts.wrap``) fills the template. The public prompt is
    ``wrap_fn`` with no ``items``, so the slot is ``"(no examples)"``
    and only content tokens spend budget. Pass exactly one of ``r``
    (fixed private-token budget) or ``target_epsilon`` (largest ``r``
    whose Theorem 1 epsilon fits). ``delta`` defaults to ``1 / s``.
    Returns the decoded string and a PrivacyAccount.
    """
    if delta is None:
        delta = 1.0 / s
    if (r is None) == (target_epsilon is None):
        raise ValueError(
            "Pass exactly one of `r` or `target_epsilon`; got "
            f"r={r!r}, target_epsilon={target_epsilon!r}."
        )
    if r is None:
        assert target_epsilon is not None  # guaranteed by the XOR check
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

    # s private prompts plus one public prompt; later tokens reuse the KV cache.
    logits, state = tg.prefill_padded(prompt_ids + [public_ids])
    while t < r and len(x_ids) < max_total_tokens:
        Z = logits[: len(prompt_ids)]  # noqa: N806
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

        # Same token on every row so the public suffix tracks the private one.
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
