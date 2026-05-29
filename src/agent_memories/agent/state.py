"""State schema for the Think-Act-Observe loop.

A plain ``TypedDict`` rather than ``MessagesState``: this agent is not
a chat. The fields are the four things any web agent needs across one
cycle (what we want, what we see, what we think, what we do) plus a
small amount of bookkeeping for termination, plus two memory channels.

The two memory channels are intentionally distinct:

* ``history`` is the *in-trajectory working memory*: an ordered list of
  ``(step, thought, action, outcome)`` records appended on every Think
  and Act call. It is what the LLM-driven Think feeds back into its
  next prompt so the agent can reason about its own past steps, and
  it is the substrate the WP1.6 memory pipeline reads from.
* ``memories`` is reserved for the *persistent, retrieved memories*
  WP1.5 introduces (per-user private store) and WP2 extends (shared
  cross-user store under differential privacy). It is empty in WP1.3.
  From WP1.6 onwards each retrieved memory is a ``{"title", "content"}``
  dict (the ReasoningBank "title + content" view per Appendix A.2);
  the runner flattens :class:`MemoryEntry.items` into this shape
  before invoking the graph.

Keeping these two channels separate avoids overloading the word
"memory" -- only ``memories`` is governed by the thesis' privacy
guarantees.
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
    memories: list[dict[str, str]]
    history: list[dict[str, Any]]


def new_state(aim: str) -> AgentState:
    """Return a freshly initialised state with empty observation and history."""
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
