"""Token budget, fenced-reply recovery and stuck detection on the BrowserGym Think node."""

from __future__ import annotations

import pytest

from agent_memories.agent.browsergym import nodes as browsergym_nodes
from agent_memories.agent.browsergym.nodes import make_think
from agent_memories.agent.state import AgentState, new_state
from tests.conftest import FakeChatClient

pytestmark = pytest.mark.unit


def _seed_state(*, aim: str = "Find the top product") -> AgentState:
    state = new_state(aim=aim)
    state["observation"] = {
        "url": "http://localhost:7780/",
        "title": "Dashboard",
        "tree_yaml": "[53] button 'Search'",
    }
    return state


def test_think_budget_covers_reasoning_and_action() -> None:
    client = FakeChatClient(["click('53')"])
    make_think(client)(_seed_state())

    assert client.calls[0]["max_tokens"] == browsergym_nodes.THINK_MAX_TOKENS
    assert browsergym_nodes.THINK_MAX_TOKENS == 1024


def test_fenced_reply_is_recovered_as_action() -> None:
    client = FakeChatClient(["```python\nclick('53')\n```"])
    result = make_think(client)(_seed_state())

    assert result["action"] == {"type": "browsergym", "string": "click('53')"}
    assert result["thought"] == "click('53')"
    assert result["history"] == []


def test_empty_repeated_thoughts_trip_stuck() -> None:
    client = FakeChatClient([])
    think = make_think(client, stuck_threshold=5)
    state = _seed_state()
    state["history"] = [
        {"step": i, "thought": "", "action": "", "outcome": "parse_failure: empty"}
        for i in range(5)
    ]

    result = think(state)

    assert result["done"] is True
    assert client.calls == []
    assert result["history"][-1]["outcome"].startswith("stuck:")
