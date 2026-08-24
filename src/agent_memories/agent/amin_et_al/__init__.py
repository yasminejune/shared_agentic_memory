"""Amin et al. (2024) Algorithm 1 sampler and its zCDP accountant.

DP over a batch of s private prompts, with a
check that falls back to a public prompt when the private and public next
token distributions agree. Shares the Gemma wrapper and prompt templates
with agent/lm/.
"""

from .accounting import (
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
