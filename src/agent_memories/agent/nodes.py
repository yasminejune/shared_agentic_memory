"""Factory functions that produce LangGraph nodes bound to a Playwright page.

LangGraph nodes have signature ``(state) -> state``, so we cannot pass
the browser handle as an argument. Instead, each factory closes over
the ``Page`` (or, for Think, the scripted action list) and returns the
node function. This keeps the browser out of the graph's state and out
of module-level globals.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

from .actions import dispatch
from .observation import format_aria_snapshot
from .state import AgentState

NodeFn = Callable[[AgentState], AgentState]


def make_observe(page: Page) -> NodeFn:
    """Read the current page's ARIA snapshot into ``state['observation']``.

    The observation carries page metadata (``url``, ``title``) and the
    YAML accessibility tree (``tree_yaml``) produced by
    ``aria_snapshot(mode="ai")``. The YAML embeds ``[ref=eN]`` markers
    that WP1.3 will resolve via ``page.locator("aria-ref=eN")``.
    """

    def observe(state: AgentState) -> AgentState:
        observation: dict[str, Any] = {
            "url": page.url,
            "title": page.title(),
            "tree_yaml": format_aria_snapshot(page),
        }
        return {**state, "observation": observation, "url": page.url}

    return observe


def make_think_scripted(actions: list[dict[str, Any]]) -> NodeFn:
    """Walk a fixed list of actions; flip ``done`` when exhausted.

    Uses ``state['step']`` as the index into ``actions``. Once the
    index is past the end of the list, the Think node signals
    termination by setting ``done=True``. The Mistral-driven Think
    replaces this in WP1.3 with the same ``(state) -> state`` signature
    so the graph topology stays unchanged.
    """

    def think(state: AgentState) -> AgentState:
        idx = state["step"]
        if idx >= len(actions):
            return {
                **state,
                "thought": "Scripted actions exhausted; stopping.",
                "done": True,
            }
        action = actions[idx]
        thought = f"Scripted action {idx}: type={action.get('type')!r}"
        return {**state, "thought": thought, "action": dict(action)}

    return think


def make_act(page: Page) -> NodeFn:
    """Execute ``state['action']`` on ``page`` and increment the step counter.

    Act no longer decides termination; that responsibility belongs to
    Think. The graph loops Observe -> Think -> Act -> Observe until
    Think sets ``done=True``.
    """

    def act(state: AgentState) -> AgentState:
        dispatch(page, state["action"])
        return {**state, "step": state["step"] + 1}

    return act
