"""Action dispatch for the Act node.

Actions are plain dictionaries with a ``type`` discriminator and the
fields each type needs. Keeping the schema as data (not classes) means
the same action dictionaries can later be produced by an LLM and
validated with a JSON schema without any structural change here.
"""

from __future__ import annotations

from typing import Any

from playwright.sync_api import Page


class UnknownActionError(ValueError):
    """Raised when an action's ``type`` is not in the supported set."""


def dispatch(page: Page, action: dict[str, Any]) -> None:
    """Execute ``action`` against ``page``."""
    kind = action.get("type")
    if kind == "goto":
        page.goto(action["url"])
    elif kind == "click":
        page.click(action["selector"])
    elif kind == "fill":
        page.fill(action["selector"], action["value"])
    else:
        raise UnknownActionError(f"Unknown action type: {kind!r}")
