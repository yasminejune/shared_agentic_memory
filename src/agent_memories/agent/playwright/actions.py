"""Action grammar, parser and dispatch for the Act node.

Two routes coexist on purpose:

* Selector-based actions (``{"type": "click", "selector": ...}`` etc.)
  remain available so the WP1.1 and WP1.2 scripted-Think entry points
  keep working unchanged.
* ARIA-ref actions (``{"type": "click", "ref": "eN"}``) are what the
  WP1.3 Mistral-driven Think emits, after running its single-line
  response through ``parse_action``. The dispatcher resolves refs via
  ``page.locator("aria-ref=eN")``, matching the snapshot contract in
  ``observation.py``.

The grammar the LLM is constrained to is intentionally small:

    click [eN]
    type [eN] "text"
    enter
    scroll up
    scroll down
    goto "url"
    stop

``enter`` presses the Enter key globally; in practice this submits the
input the agent just filled, because ``type`` leaves that element
focused. ``stop`` is consumed by Think (it flips ``done``) and never
reaches the dispatcher.
"""

from __future__ import annotations

import re
from typing import Any

from playwright.sync_api import Page


class UnknownActionError(ValueError):
    """Raised when an action's ``type`` is not in the supported set."""


class ActionParseError(ValueError):
    """Raised when an LLM response does not match the action grammar."""


_CLICK_RE = re.compile(r"^click\s+\[(e\d+)\]$")
# type [eN] "..." with support for escaped quotes inside the value.
_TYPE_RE = re.compile(r'^type\s+\[(e\d+)\]\s+"((?:[^"\\]|\\.)*)"$')
_GOTO_RE = re.compile(r'^goto\s+"((?:[^"\\]|\\.)*)"$')


def parse_action(line: str) -> dict[str, Any]:
    """Parse a single grammar line into an action dictionary.

    Accepted forms (with ``N`` a positive integer):

    * ``click [eN]``        -> ``{"type": "click", "ref": "eN"}``
    * ``type [eN] "text"``  -> ``{"type": "fill",  "ref": "eN", "value": "text"}``
    * ``enter``             -> ``{"type": "press", "key": "Enter"}``
    * ``scroll up``         -> ``{"type": "scroll", "direction": "up"}``
    * ``scroll down``       -> ``{"type": "scroll", "direction": "down"}``
    * ``goto "url"``        -> ``{"type": "goto", "url": "url"}``
    * ``stop``              -> ``{"type": "stop"}``

    Anything else raises :class:`ActionParseError`. The whole input
    must be exactly one line; surrounding whitespace is tolerated but
    embedded newlines are not.
    """
    if line is None:
        raise ActionParseError("Action line was None")
    stripped = line.strip()
    if not stripped:
        raise ActionParseError("Action line was empty")
    if "\n" in stripped:
        raise ActionParseError(f"Action must be a single line, got: {line!r}")

    if stripped == "stop":
        return {"type": "stop"}
    if stripped == "enter":
        return {"type": "press", "key": "Enter"}
    if stripped == "scroll up":
        return {"type": "scroll", "direction": "up"}
    if stripped == "scroll down":
        return {"type": "scroll", "direction": "down"}

    match = _CLICK_RE.match(stripped)
    if match:
        return {"type": "click", "ref": match.group(1)}

    match = _TYPE_RE.match(stripped)
    if match:
        ref, raw_value = match.group(1), match.group(2)
        return {"type": "fill", "ref": ref, "value": _unescape(raw_value)}

    match = _GOTO_RE.match(stripped)
    if match:
        return {"type": "goto", "url": _unescape(match.group(1))}

    raise ActionParseError(f"Could not parse action: {stripped!r}")


def _unescape(raw: str) -> str:
    # Mirror the small set of escapes the regex permits: \" and \\.
    return raw.replace('\\"', '"').replace("\\\\", "\\")


def dispatch(page: Page, action: dict[str, Any]) -> None:
    """Execute ``action`` against ``page``.

    Actions may target an element by selector (legacy scripted-Think
    path) or by ARIA ref (WP1.3 LLM path). ``stop`` is handled by the
    Think node and must never reach this function.
    """
    kind = action.get("type")
    if kind == "goto":
        page.goto(action["url"])
    elif kind == "click":
        _target(page, action).click()
    elif kind == "fill":
        _target(page, action).fill(action["value"])
    elif kind == "scroll":
        key = "PageDown" if action.get("direction") == "down" else "PageUp"
        page.keyboard.press(key)
    elif kind == "press":
        page.keyboard.press(action["key"])
    else:
        raise UnknownActionError(f"Unknown action type: {kind!r}")


def _target(page: Page, action: dict[str, Any]) -> Any:
    if "ref" in action:
        return page.locator(f"aria-ref={action['ref']}")
    if "selector" in action:
        return page.locator(action["selector"])
    raise UnknownActionError(f"Action {action.get('type')!r} needs a 'ref' or 'selector' field")
