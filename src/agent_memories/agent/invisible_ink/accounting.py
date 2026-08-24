"""InvisibleInk wrappers around invink.utils.

B is used to mean both the batch size and the private-reference count in the InvisibleInk repo.
Thus we differentiate here:
* get_clip and get_epsilon take B+1 (inferences per token,
including the public prompt).
* compute_rho takes B, the private-reference count.
"""

from __future__ import annotations

from dataclasses import dataclass

from invink.utils import cdp_eps, cdp_rho, compute_rho, get_clip, get_epsilon


@dataclass(frozen=True)
class InvisibleInkAccount:
    """Result of one run of InvisibleInk Algorithm 1.

    Every generated token spends budget. t is the
    length of the generated text in tokens, used to calibrate c;
    tokens_used is the realised length (may be lower if a stop token arrived early).
    """

    epsilon: float
    delta: float
    rho: float
    t: int
    b: int
    c: float
    tau: float
    top_k: int
    tokens_used: int
    topk_plus_mean: float
    topk_plus_std: float
    expansion_set_count: int


def clip_for_budget(
    target_epsilon: float,
    delta: float,
    num_tokens: int,
    b: int,
    tau: float,
) -> float:
    """Calibrate clip norm C for a target (epsilon, delta)-DP budget (Theorem 2)."""
    return float(
        get_clip(
            epsilon=target_epsilon,
            num_toks=num_tokens,
            temp=tau,
            batch_size=b + 1,
            delta=delta,
        )
    )


def epsilon_for_tokens(
    num_tokens: int,
    c: float,
    b: int,
    tau: float,
    delta: float,
) -> float:
    """Realised (epsilon, delta)-DP epsilon after num_tokens private tokens."""
    if num_tokens <= 0:
        return 0.0
    return float(
        get_epsilon(
            num_toks=num_tokens,
            clip_norm=c,
            batch_size=b + 1,
            temp=tau,
            delta=delta,
        )
    )


def rho_for_tokens(num_tokens: int, c: float, b: int, tau: float) -> float:
    """zCDP cost rho_seq = T * (C / (B * tau))**2 / 2 (Theorem 2).

    Passes paper b directly: compute_rho treats batch_size as
    the private-reference count B.
    """
    if num_tokens <= 0:
        return 0.0
    return float(compute_rho(num_tokens, c, b, tau))


def epsilon_from_rho(rho: float, delta: float) -> float:
    """Tight zCDP to (epsilon, delta)-DP conversion (invink cdp_eps)."""
    if rho <= 0.0:
        return 0.0
    return float(cdp_eps(rho, delta))


def rho_from_epsilon(epsilon: float, delta: float) -> float:
    """Tight (epsilon, delta)-DP to zCDP conversion (invink cdp_rho)."""
    if epsilon <= 0.0:
        return 0.0
    return float(cdp_rho(epsilon, delta))
