"""Compile the Observe -> Think -> Act loop.

The graph cycles until Think sets ``state['done']`` to ``True``. In
WP1.2 the Think node is a scripted stub that exhausts a fixed list of
actions; in WP1.3 it is replaced by a Mistral-driven node with the
same signature, so the graph topology does not change.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Page

from .nodes import make_act, make_observe, make_think_scripted
from .state import AgentState


def build_graph(page: Page, stub_actions: list[dict[str, Any]]) -> Any:
    """Return a compiled LangGraph app bound to ``page``.

    ``stub_actions`` is the list of actions the scripted Think will
    emit, one per cycle. The graph terminates when Think runs out of
    actions and sets ``done=True``.
    """
    workflow: StateGraph = StateGraph(AgentState)
    workflow.add_node("observe", make_observe(page))  # type: ignore[call-overload]
    workflow.add_node("think", make_think_scripted(stub_actions))  # type: ignore[call-overload]
    workflow.add_node("act", make_act(page))  # type: ignore[call-overload]
    workflow.add_edge(START, "observe")
    workflow.add_edge("observe", "think")
    workflow.add_conditional_edges(
        "think",
        lambda state: END if state["done"] else "act",
        {END: END, "act": "act"},
    )
    workflow.add_edge("act", "observe")
    return workflow.compile()
