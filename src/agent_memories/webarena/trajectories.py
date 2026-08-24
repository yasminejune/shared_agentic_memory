"""Rebuild agent state from a recorded trajectory row."""

from __future__ import annotations

import json

from agent_memories.agent.state import AgentState, new_state


def state_from_trajectory_row(row: dict[str, str]) -> AgentState:
    """Rebuild AgentState from a trajectories CSV row."""
    intent = row.get("intent", "")
    state = new_state(aim=intent)
    state["memories"] = []
    raw = row.get("raw_trajectory", "").strip()
    if raw:
        state["history"] = json.loads(raw)
    else:
        state["history"] = []
    return state


def truncate_observation(tree_yaml: str, budget: int) -> str:
    if budget <= 0 or len(tree_yaml) <= budget:
        return tree_yaml
    marker = "\n# ... <observation truncated to fit context window> ..."
    head = tree_yaml[: max(0, budget - len(marker))]
    return head + marker
