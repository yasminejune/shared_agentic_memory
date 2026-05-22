"""Compile the Observe -> Think -> Act graph for a single-step task."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Page

from .nodes import make_act, make_observe, make_think_stub
from .state import AgentState


def build_graph(page: Page, stub_action: dict[str, Any]) -> Any:
    """Return a compiled LangGraph app bound to ``page``.

    The Think node is the deterministic stub; ``stub_action`` is the
    action it will emit on every invocation.
    """
    workflow: StateGraph = StateGraph(AgentState)
    workflow.add_node("observe", make_observe(page))  # type: ignore[call-overload]
    workflow.add_node("think", make_think_stub(stub_action))  # type: ignore[call-overload]
    workflow.add_node("act", make_act(page))  # type: ignore[call-overload]
    workflow.add_edge(START, "observe")
    workflow.add_edge("observe", "think")
    workflow.add_edge("think", "act")
    workflow.add_edge("act", END)
    return workflow.compile()
