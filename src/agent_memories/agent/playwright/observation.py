"""ARIA YAML tree for the Observe node.

Playwright's aria_snapshot(mode="ai") returns YAML with [ref=eN]
markers. Refs are ephemeral: they are invalid after the next action.
"""

from __future__ import annotations

from playwright.sync_api import Page


def format_aria_snapshot(page: Page) -> str:
    """Return the page's ARIA snapshot as YAML with [ref=eN] markers.

    Falsy Playwright returns become an empty string.
    """
    return page.aria_snapshot(mode="ai") or ""
