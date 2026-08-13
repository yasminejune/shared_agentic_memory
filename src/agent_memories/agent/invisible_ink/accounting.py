"""Privacy accounting for InvisibleInk (Vinod et al., arXiv:2507.02974).

Thin typed adapters over ``invink.utils`` that fix the authors' inconsistent
``batch_size`` convention at a single boundary. Every public function in this
module takes the paper's ``B`` (number of private references) and translates
internally:

* :func:`invink.utils.get_clip` / :func:`invink.utils.get_epsilon` expect
  ``batch_size = B + 1`` (LLM inferences per generated token, including the
  public prompt) and subtract one before composing with Theorem 2.
* :func:`invink.utils.compute_rho` expects ``batch_size = B`` (private
  references only).

Passing paper ``B`` through both without this translation understates
``ρ_seq`` by ``((B+1)/B)²`` at the clip / epsilon call sites. The asymmetry
looks like a bug when reading the call sites; it is deliberate and matches
the ``invink`` docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

from invink.utils import cdp_eps, cdp_rho, compute_rho, get_clip, get_epsilon


@dataclass(frozen=True)
class InvisibleInkAccount:
    """Result of one run of InvisibleInk Algorithm 1.

    Pure-data record. Unlike Amin's :class:`PrivacyAccount` there is no SVT
    branch, so every generated token spends budget and there is no
    ``public_tokens_used`` / ``sigma`` / ``r`` field. ``t`` is the a-priori
    token budget used to calibrate ``c``; ``tokens_used`` is the realised
    length (may be lower if the model emitted a stop token early).
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
    """Calibrate clip norm ``C`` for a target ``(ε, δ)`` budget (Theorem 2).

    Translates paper ``b`` into invink's ``batch_size = b + 1`` convention.
    """
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
    """Realised ``(ε, δ)``-DP epsilon after ``num_tokens`` private tokens.

    Translates paper ``b`` into invink's ``batch_size = b + 1`` convention.
    """
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
    """zCDP cost ``ρ_seq = T · (C / (B τ))² / 2`` (Theorem 2).

    Passes paper ``b`` directly: :func:`invink.utils.compute_rho` already
    treats ``batch_size`` as the private-reference count ``B``.
    """
    if num_tokens <= 0:
        return 0.0
    return float(compute_rho(num_tokens, c, b, tau))


def epsilon_from_rho(rho: float, delta: float) -> float:
    """Tight zCDP → ``(ε, δ)``-DP conversion (invink ``cdp_eps``)."""
    if rho <= 0.0:
        return 0.0
    return float(cdp_eps(rho, delta))


def rho_from_epsilon(epsilon: float, delta: float) -> float:
    """Tight ``(ε, δ)``-DP → zCDP conversion (invink ``cdp_rho``)."""
    if epsilon <= 0.0:
        return 0.0
    return float(cdp_rho(epsilon, delta))
