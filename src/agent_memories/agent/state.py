"""State for one Observe-Think-Act run.

is not dependent on the backend i.e. any Playwright or BrowserGym.
history is the current trajectory. memories is the retrieved title/content
items, empty when none were fetched.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Fields passed between Observe, Think and Act."""

    aim: str
    url: str
    observation: dict[str, Any]
    thought: str
    action: dict[str, Any]
    step: int
    done: bool
    memories: list[dict[str, str]]
    history: list[dict[str, Any]]


def new_state(aim: str) -> AgentState:
    """Return an AgentState with empty observation and history."""
    return AgentState(
        aim=aim,
        url="",
        observation={},
        thought="",
        action={},
        step=0,
        done=False,
        memories=[],
        history=[],
    )
