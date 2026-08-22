"""Re-export the Playwright backend so older imports still work."""

from agent_memories.agent.playwright.observation import format_aria_snapshot

__all__ = ["format_aria_snapshot"]
