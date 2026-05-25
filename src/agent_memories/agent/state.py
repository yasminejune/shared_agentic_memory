"""State schema for the Think-Act-Observe loop.

A plain ``TypedDict`` rather than ``MessagesState``: this agent is not a
chat. The fields are the four things any web agent needs across one
cycle (what we want, what we see, what we think, what we do) plus a
small amount of bookkeeping for termination.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """The state passed between Observe, Think and Act."""

    aim: str
    url: str
    observation: dict[str, Any]
    thought: str
    action: dict[str, Any]
    step: int
    done: bool
    memories: list[str]


def new_state(aim: str) -> AgentState:
    """Return a freshly initialised state with an empty observation."""
    return AgentState(
        aim=aim,
        url="",
        observation={},
        thought="",
        action={},
        step=0,
        done=False,
        memories=[],
    )
