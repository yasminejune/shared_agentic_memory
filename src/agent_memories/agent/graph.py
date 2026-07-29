"""Backward-compatible re-exports from the Playwright agent backend."""

from agent_memories.agent.playwright.graph import DEFAULT_MAX_STEPS, build_graph

__all__ = ["DEFAULT_MAX_STEPS", "build_graph"]
