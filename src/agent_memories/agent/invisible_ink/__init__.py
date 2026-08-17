"""InvisibleInk (Vinod et al., arXiv:2507.02974) private text generation (WP2).

Wraps the authors' ``invink`` privacy primitives (DClip, Top-k+, zCDP
accounting) in a generation loop that reuses this project's Gemma 2 IT
token plumbing and prompt templates. Sibling of
:mod:`agent_memories.agent.privacy` (Amin et al. 2024); the two packages
share :mod:`~agent_memories.agent.privacy.token_generation` and
:mod:`~agent_memories.agent.privacy.prompts` but keep their mechanisms
and accountants separate.
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
