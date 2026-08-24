"""The graph for the Observe - Think - Act loop on a BrowserGym WebArena env."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent_memories.agent.state import AgentState

from .env import WebArenaEnvWrapper
from .nodes import NodeFn, make_act, make_observe

DEFAULT_MAX_STEPS = 30


def build_graph(
    wrapper: WebArenaEnvWrapper,
    think_fn: NodeFn,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Any:
    """Compile Observe -> Think -> Act. Think routes to act, observe, or END."""

    def route_from_think(state: AgentState) -> str:
        # After think, the agent should either terminate if done, end if max steps has been reached,
        # make an observation if the observation didn't go through, or act.
        if state["done"]:
            return END
        if state["step"] >= max_steps:
            return END
        if not state["action"]:
            return "observe"
        return "act"

    workflow: StateGraph = StateGraph(AgentState)
    workflow.add_node("observe", make_observe(wrapper))  # type: ignore[call-overload]
    workflow.add_node("think", think_fn)  # type: ignore[call-overload]
    workflow.add_node("act", make_act(wrapper))  # type: ignore[call-overload]
    workflow.add_edge(START, "observe")
    # After observe, the agent always thinks. No function necessary.
    workflow.add_edge("observe", "think")
    # After think, the agent has multiple options.
    workflow.add_conditional_edges(
        "think",
        route_from_think,
        {END: END, "observe": "observe", "act": "act"},
    )
    # After act, always observe. No function necessary.
    workflow.add_edge("act", "observe")
    return workflow.compile()
