"""Compile the Observe -> Think -> Act loop.

The Think factory is the caller's job: WP1.1 and WP1.2 pass
``make_think_scripted(stub_actions)``, WP1.3 onwards pass
``make_think(client)`` (the LLM-driven Think node). The graph keeps the
same topology and adds two safety routes after Think:

* parse failure (Think returned an empty action) -> back to Observe
  with the failure logged in ``state['history']``; the next prompt
  will see it. No special error field is involved.
* ``state['step'] >= max_steps`` -> END, regardless of ``done``. This
  is a hard cap so unproductive loops (repeated parse failures, the
  LLM never emitting ``stop``) cannot run forever. The thesis proposal
  flags 30 as a conservative default.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Page

from .nodes import NodeFn, make_act, make_observe
from .state import AgentState

DEFAULT_MAX_STEPS = 30


def build_graph(page: Page, think_fn: NodeFn, *, max_steps: int = DEFAULT_MAX_STEPS) -> Any:
    """Return a compiled LangGraph app bound to ``page`` and ``think_fn``.

    Routing after Think has four outcomes:

    * ``state['done']`` is True            -> END
    * ``state['step'] >= max_steps``       -> END (hard cap)
    * ``state['action']`` is empty         -> Observe (parse failure
                                              recovery via history)
    * otherwise                            -> Act -> Observe
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
