"""Backward-compatible re-exports from the Playwright agent backend."""

from agent_memories.agent.playwright.nodes import (
    DEFAULT_STUCK_THRESHOLD,
    MEMORY_INJECTION_INSTRUCTION,
    OBSERVATION_CHAR_BUDGET,
    THINK_SYSTEM_PROMPT,
    NodeFn,
    make_act,
    make_observe,
    make_think,
    make_think_scripted,
)

__all__ = [
    "DEFAULT_STUCK_THRESHOLD",
    "MEMORY_INJECTION_INSTRUCTION",
    "OBSERVATION_CHAR_BUDGET",
    "THINK_SYSTEM_PROMPT",
    "NodeFn",
    "make_act",
    "make_observe",
    "make_think",
    "make_think_scripted",
]
