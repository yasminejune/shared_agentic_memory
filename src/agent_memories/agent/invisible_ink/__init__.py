"""InvisibleInk (Vinod et al., arXiv:2507.02974) token-level DP generation.

The thesis deploys this mechanism for Step 1 labels and Step 3
shared-memory content. Shares Gemma plumbing and prompt templates with
agent/privacy/. Guarantee is example-level (one MemoryEntry), not
user-level.
"""

from __future__ import annotations

from .accounting import (
    InvisibleInkAccount,
    clip_for_budget,
    epsilon_for_tokens,
    epsilon_from_rho,
    rho_for_tokens,
    rho_from_epsilon,
)
from .generation import generate, generate_microbatched, generate_with_oom_fallback

__all__ = [
    "InvisibleInkAccount",
    "clip_for_budget",
    "epsilon_for_tokens",
    "epsilon_from_rho",
    "generate",
    "generate_microbatched",
    "generate_with_oom_fallback",
    "rho_for_tokens",
    "rho_from_epsilon",
]
