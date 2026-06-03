"""Amin et al. (2024) private prediction mechanism (WP2.3).

Verbatim implementation of Algorithm 1 from
`Amin et al. 2024 <https://arxiv.org/abs/2407.12108>`_, the
zCDP-based privacy accountant that derives the per-batch ``(epsilon,
delta)`` from the hyperparameters, and the Gemma 2 2B IT wrapper
that exposes per-position token logits.
"""

from .privacy_accounting import (
    DeltaCheck,
    PrivacyAccount,
    check_delta,
    delta_from_rho_epsilon,
    epsilon_from_rho,
    rho_for,
    solve_r,
)
from .privatisation import generate

__all__ = [
    "DeltaCheck",
    "PrivacyAccount",
    "check_delta",
    "delta_from_rho_epsilon",
    "epsilon_from_rho",
    "generate",
    "rho_for",
    "solve_r",
]
