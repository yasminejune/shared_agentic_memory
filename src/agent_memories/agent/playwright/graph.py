"""Compile the Observe-Think-Act loop.

The caller supplies Think (scripted or LLM). After Think, an empty
action returns to Observe; done or step >= max_steps ends the run.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Page

from agent_memories.agent.state import AgentState

from .nodes import NodeFn, make_act, make_observe

DEFAULT_MAX_STEPS = 10


def build_graph(page: Page, think_fn: NodeFn, *, max_steps: int = DEFAULT_MAX_STEPS) -> Any:
    """Return a compiled LangGraph app bound to page and think_fn.

    After Think: done or step >= max_steps ends; empty action returns
    to Observe; otherwise Act then Observe.
    """

    def route_from_think(state: AgentState) -> str:
        if state["done"]:
            return END
        if state["step"] >= max_steps:
            return END
        if not state["action"]:
            return "observe"
        return "act"

    workflow: StateGraph = StateGraph(AgentState)
    workflow.add_node("observe", make_observe(page))  # type: ignore[call-overload]
    workflow.add_node("think", think_fn)  # type: ignore[call-overload]
    workflow.add_node("act", make_act(page))  # type: ignore[call-overload]
    workflow.add_edge(START, "observe")
    workflow.add_edge("observe", "think")
    workflow.add_conditional_edges(
        "think",
        route_from_think,
        {END: END, "observe": "observe", "act": "act"},
    )
    workflow.add_edge("act", "observe")
    return workflow.compile()
