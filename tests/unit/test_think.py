"""Prompt assembly, action parsing and the stuck detector in the Think node."""

from __future__ import annotations

import pytest

from agent_memories.agent.playwright.nodes import make_think
from agent_memories.agent.state import AgentState, new_state
from tests.conftest import FakeChatClient

pytestmark = pytest.mark.unit


def _seed_state(*, aim: str = "Click Go", tree: str = "- button [ref=e1]") -> AgentState:
    state = new_state(aim=aim)
    state["observation"] = {"url": "about:blank", "title": "", "tree_yaml": tree}
    return state


def test_success_leaves_history_for_act() -> None:
    client = FakeChatClient(["click [e1]"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {"type": "click", "ref": "e1"}
    assert result["done"] is False
    assert result["thought"] == "click [e1]"
    assert result["history"] == [], "Think should not log success; Act does that"


def test_type_action_round_trip() -> None:
    client = FakeChatClient(['type [e7] "London"'])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {"type": "fill", "ref": "e7", "value": "London"}


def test_stop_sets_done_and_logs_history() -> None:
    client = FakeChatClient(["stop"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["done"] is True
    assert result["action"] == {}
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert record["outcome"] == "stop"
    assert record["thought"] == "stop"
    assert record["step"] == 0


def test_parse_failure_advances_step() -> None:
    client = FakeChatClient(["I'll click the button now"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {}
    assert result["done"] is False
    assert result["step"] == 1, "parse failure must advance step against max_steps"
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert record["outcome"].startswith("parse_failure:")
    assert "I'll click the button now" in record["thought"]


def test_history_is_threaded_into_next_prompt() -> None:
    client = FakeChatClient(["click [e1]"])
    think = make_think(client)

    state = _seed_state()
    state["history"] = [
        {"step": 0, "thought": "oops", "action": {}, "outcome": "parse_failure: bad"},
        {
            "step": 1,
            "thought": "click [e9]",
            "action": {"type": "click", "ref": "e9"},
            "outcome": "ok",
        },
    ]

    think(state)

    assert len(client.calls) == 1
    user_prompt = str(client.calls[0]["user"])
    assert "Aim: Click Go" in user_prompt
    assert "History (most recent last):" in user_prompt
    assert "parse_failure: bad" in user_prompt
    assert "click [e9]" in user_prompt
    assert "- button [ref=e1]" in user_prompt


def test_response_is_stripped_before_parsing() -> None:
    client = FakeChatClient(["  click [e2]\n"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {"type": "click", "ref": "e2"}


def test_scroll_and_goto_round_trip() -> None:
    client = FakeChatClient(["scroll down"])
    result = make_think(client)(_seed_state())
    assert result["action"] == {"type": "scroll", "direction": "down"}

    client = FakeChatClient(['goto "https://example.com"'])
    result = make_think(client)(_seed_state())
    assert result["action"] == {"type": "goto", "url": "https://example.com"}


def test_observation_yaml_is_truncated() -> None:
    from agent_memories.agent.constants import OBSERVATION_CHAR_BUDGET

    client = FakeChatClient(["click [e1]"])
    huge_tree = "- node\n" * 5000  # well over the 12k character budget
    state = _seed_state(tree=huge_tree)

    make_think(client)(state)

    user_prompt = str(client.calls[0]["user"])
    accessibility_block = user_prompt.split("Accessibility tree:\n", 1)[1]
    accessibility_block = accessibility_block.split("\n\nReply", 1)[0]
    assert len(accessibility_block) <= OBSERVATION_CHAR_BUDGET
    assert "<observation truncated" in accessibility_block


def test_system_prompt_explains_refs() -> None:
    from agent_memories.agent.playwright.nodes import THINK_SYSTEM_PROMPT

    assert "EXACTLY one line" in THINK_SYSTEM_PROMPT
    assert "no markdown" in THINK_SYSTEM_PROMPT
    assert "[ref=e29]" in THINK_SYSTEM_PROMPT
    assert "[e29]" in THINK_SYSTEM_PROMPT
    assert "MANDATORY" in THINK_SYSTEM_PROMPT


def test_terminal_trace_is_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = FakeChatClient(["click [e1]"])
    think = make_think(client)

    think(_seed_state())

    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert out_lines == ["[Think]: click [e1]"]


def test_parse_failure_outcome_has_no_hint() -> None:
    client = FakeChatClient(['type 7 "paper"'])
    think = make_think(client)

    result = think(_seed_state())

    assert len(result["history"]) == 1
    outcome = result["history"][0]["outcome"]
    assert outcome.startswith("parse_failure:")
    assert "reminder:" not in outcome
    assert "hint" not in outcome.lower()


def test_retrieved_memories_appear_in_prompt() -> None:
    from agent_memories.agent.constants import MEMORY_INJECTION_INSTRUCTION

    client = FakeChatClient(["click [e1]"])
    state = _seed_state()
    state["memories"] = [
        {
            "title": "Dismiss cookie banners early",
            "content": "Dismiss the cookie banner before searching.",
        },
        {
            "title": "Use Enter to submit",
            "content": "Use Enter to submit the search box.",
        },
    ]

    make_think(client)(state)

    user_prompt = str(client.calls[0]["user"])
    assert MEMORY_INJECTION_INSTRUCTION in user_prompt
    assert "Title: Dismiss cookie banners early" in user_prompt
    assert "Content: Dismiss the cookie banner before searching." in user_prompt
    assert "Title: Use Enter to submit" in user_prompt
    assert "Content: Use Enter to submit the search box." in user_prompt
    aim_index = user_prompt.index("Aim: Click Go")
    memories_index = user_prompt.index(MEMORY_INJECTION_INSTRUCTION)
    history_index = user_prompt.index("History (most recent last):")
    assert aim_index < memories_index < history_index


def test_empty_memories_list_omits_memory_block() -> None:
    from agent_memories.agent.constants import MEMORY_INJECTION_INSTRUCTION

    client = FakeChatClient(["click [e1]"])
    state = _seed_state()
    state["memories"] = []

    make_think(client)(state)

    user_prompt = str(client.calls[0]["user"])
    assert MEMORY_INJECTION_INSTRUCTION not in user_prompt


def test_memory_injection_instruction_is_paper_verbatim() -> None:
    from agent_memories.agent.constants import MEMORY_INJECTION_INSTRUCTION

    expected = (
        "Below are some memory items that I accumulated from past interaction "
        "from the environment that may be helpful to solve the task. You can "
        "use it when you feel it's relevant. In each step, please first "
        "explicitly discuss if you want to use each memory item or not, and "
        "then take action."
    )
    assert MEMORY_INJECTION_INSTRUCTION == expected


def test_memory_description_is_not_shown() -> None:
    client = FakeChatClient(["click [e1]"])
    state = _seed_state()
    state["memories"] = [
        {
            "title": "Use the labelled search input",
            "content": "Find the main search box by its accessible label.",
        },
    ]

    make_think(client)(state)

    user_prompt = str(client.calls[0]["user"])
    assert "Description:" not in user_prompt


def test_stuck_on_repeated_parse_failures() -> None:
    client = FakeChatClient([])  # unused: stuck path does not call chat
    think = make_think(client, stuck_threshold=5)

    state = _seed_state()
    state["history"] = [
        {
            "step": i,
            "thought": 'type [e1] "x"\nenter',
            "action": {},
            "outcome": "parse_failure: Action must be a single line, got: ...",
        }
        for i in range(5)
    ]

    result = think(state)

    assert result["done"] is True
    assert client.calls == [], "Think must not consult the LLM once stuck"
    assert result["history"][-1]["outcome"].startswith("stuck:")
    assert result["action"] == {}


def test_stuck_on_repeated_successful_acts() -> None:
    client = FakeChatClient([])
    think = make_think(client, stuck_threshold=5)

    state = _seed_state()
    state["history"] = [
        {
            "step": i,
            "thought": "click [e118]",
            "action": {"type": "click", "ref": "e118"},
            "outcome": "ok",
        }
        for i in range(5)
    ]

    result = think(state)

    assert result["done"] is True
    assert client.calls == []
    assert "stuck" in result["history"][-1]["outcome"]


def test_stuck_threshold_is_configurable() -> None:
    client = FakeChatClient(["click [e1]"])
    think = make_think(client, stuck_threshold=3)

    state_below = _seed_state()
    state_below["history"] = [
        {"step": i, "thought": "click [e118]", "action": {}, "outcome": "ok"} for i in range(2)
    ]
    think(state_below)
    assert len(client.calls) == 1, "two identical records is below threshold=3"

    state_at = _seed_state()
    state_at["history"] = [
        {"step": i, "thought": "click [e118]", "action": {}, "outcome": "ok"} for i in range(3)
    ]
    result = think(state_at)
    assert result["done"] is True
    assert len(client.calls) == 1, "stuck must not consult the LLM again"


def test_stuck_ignores_non_consecutive_repeats() -> None:
    client = FakeChatClient(["click [e1]"])
    think = make_think(client, stuck_threshold=5)

    state = _seed_state()
    state["history"] = [
        {"step": 0, "thought": "click [e1]", "action": {}, "outcome": "ok"},
        {"step": 1, "thought": "click [e1]", "action": {}, "outcome": "ok"},
        {"step": 2, "thought": "click [e2]", "action": {}, "outcome": "ok"},
        {"step": 3, "thought": "click [e1]", "action": {}, "outcome": "ok"},
        {"step": 4, "thought": "click [e1]", "action": {}, "outcome": "ok"},
        {"step": 5, "thought": "click [e1]", "action": {}, "outcome": "ok"},
    ]

    result = think(state)

    assert result["done"] is False
    assert len(client.calls) == 1, "non-consecutive repeats must not trip the detector"


def test_chat_exception_propagates() -> None:
    class _BoomClient:
        def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
            raise TimeoutError("read timed out")

    think = make_think(_BoomClient())

    with pytest.raises(TimeoutError):
        think(_seed_state())
