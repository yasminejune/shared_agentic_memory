"""Re-export the Playwright backend so older imports still work."""

from agent_memories.agent.playwright.graph import DEFAULT_MAX_STEPS, build_graph

__all__ = ["DEFAULT_MAX_STEPS", "build_graph"]
