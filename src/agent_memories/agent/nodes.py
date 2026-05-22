"""Factory functions that produce LangGraph nodes bound to a Playwright page.

LangGraph nodes have signature ``(state) -> state``, so we cannot pass
the browser handle as an argument. Instead, each factory closes over
the ``Page`` (or, for Think, the stub action) and returns the node
function. This keeps the browser out of the graph's state and out of
module-level globals.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

from .actions import dispatch
from .state import AgentState

NodeFn = Callable[[AgentState], AgentState]


def make_observe(page: Page) -> NodeFn:
    """Read the current page into ``state['observation']``.

    The observation is deliberately minimal for WP1.1: URL, title and
    body text. Richer observations (accessibility tree, screenshot) are
    a WP1.2+ concern.
    """

    def observe(state: AgentState) -> AgentState:
        observation: dict[str, Any] = {
            "url": page.url,
            "title": page.title(),
            "body_text": page.locator("body").inner_text(),
        }
        return {**state, "observation": observation, "url": page.url}

    return observe


def make_think_stub(action: dict[str, Any]) -> NodeFn:
    """Deterministic Think for the scaffold.

    Returns ``action`` verbatim regardless of state. The real Mistral
    call replaces this in a later WP1 ticket; the node signature is
    identical so the graph does not change.
    """

    def think(state: AgentState) -> AgentState:
        thought = f"Stubbed think: emit action of type {action.get('type')!r}"
        return {**state, "thought": thought, "action": dict(action)}

    return think


def make_act(page: Page) -> NodeFn:
    """Execute ``state['action']`` on ``page`` and mark the cycle done.

    WP1.1 is a single-step task, so Act always sets ``done=True``. A
    multi-step controller is the next ticket.
    """

    def act(state: AgentState) -> AgentState:
        dispatch(page, state["action"])
        return {**state, "step": state["step"] + 1, "done": True}

    return act
