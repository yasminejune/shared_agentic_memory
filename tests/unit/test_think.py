"""Unit tests for the LLM-driven Think node.

A fake chat client is injected so no network call is made. The tests
cover the three Think outcomes (success / stop / parse failure) and
confirm that:

* successful parses set ``action`` and leave ``history`` untouched
  (Act is the one that logs successful dispatches);
* ``stop`` flips ``done`` and appends a ``"stop"`` record;
* parse failures append a ``"parse_failure: ..."`` record, clear
  ``action``, and increment ``step`` so the loop cannot spin forever;
* the running ``history`` is serialised into the next user prompt so
  the model can reason over its own past steps.
"""

from __future__ import annotations

import pytest

from agent_memories.agent.nodes import make_think
from agent_memories.agent.state import AgentState, new_state


class _FakeClient:
    """Chat client double that returns a queued reply per call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, str]] = []

    def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user})
        return self._replies.pop(0)


def _seed_state(*, aim: str = "Click Go", tree: str = "- button [ref=e1]") -> AgentState:
    state = new_state(aim=aim)
    state["observation"] = {"url": "about:blank", "title": "", "tree_yaml": tree}
    return state


@pytest.mark.unit
def test_success_populates_action_and_leaves_history_for_act() -> None:
    client = _FakeClient(["click [e1]"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {"type": "click", "ref": "e1"}
    assert result["done"] is False
    assert result["thought"] == "click [e1]"
    assert result["history"] == [], "Think should not log success; Act does that"


@pytest.mark.unit
def test_type_action_round_trip() -> None:
    client = _FakeClient(['type [e7] "London"'])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {"type": "fill", "ref": "e7", "value": "London"}


@pytest.mark.unit
def test_stop_sets_done_and_appends_history_record() -> None:
    client = _FakeClient(["stop"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["done"] is True
    assert result["action"] == {}
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert record["outcome"] == "stop"
    assert record["thought"] == "stop"
    assert record["step"] == 0


@pytest.mark.unit
def test_parse_failure_logs_history_and_increments_step() -> None:
    client = _FakeClient(["I'll click the button now"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {}
    assert result["done"] is False
    assert result["step"] == 1, "parse failure must advance step against max_steps"
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert record["outcome"].startswith("parse_failure:")
    assert "I'll click the button now" in record["thought"]


@pytest.mark.unit
def test_history_is_threaded_into_next_prompt() -> None:
    client = _FakeClient(["click [e1]"])
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
    user_prompt = client.calls[0]["user"]
    assert "Aim: Click Go" in user_prompt
    assert "History (most recent last):" in user_prompt
    assert "parse_failure: bad" in user_prompt
    assert "click [e9]" in user_prompt
    assert "- button [ref=e1]" in user_prompt


@pytest.mark.unit
def test_response_is_stripped_before_parsing() -> None:
    client = _FakeClient(["  click [e2]\n"])
    think = make_think(client)

    result = think(_seed_state())

    assert result["action"] == {"type": "click", "ref": "e2"}


@pytest.mark.unit
def test_scroll_and_goto_round_trip() -> None:
    client = _FakeClient(["scroll down"])
    result = make_think(client)(_seed_state())
    assert result["action"] == {"type": "scroll", "direction": "down"}

    client = _FakeClient(['goto "https://example.com"'])
    result = make_think(client)(_seed_state())
    assert result["action"] == {"type": "goto", "url": "https://example.com"}


@pytest.mark.unit
def test_observation_yaml_is_truncated_in_user_prompt() -> None:
    """An over-budget ARIA snapshot must be capped before reaching the LLM.

    Without this cap a site like amazon.co.uk produces a 60k+ token
    snapshot that overflows an 8B local model's context window, the
    server silently throws away the system prompt and aim, and the
    loop wedges.
    """
    from agent_memories.agent.nodes import OBSERVATION_CHAR_BUDGET

    client = _FakeClient(["click [e1]"])
    huge_tree = "- node\n" * 5000  # well over the 12k character budget
    state = _seed_state(tree=huge_tree)

    make_think(client)(state)

    user_prompt = client.calls[0]["user"]
    accessibility_block = user_prompt.split("Accessibility tree:\n", 1)[1]
    accessibility_block = accessibility_block.split("\n\nReply", 1)[0]
    assert len(accessibility_block) <= OBSERVATION_CHAR_BUDGET
    assert "<observation truncated" in accessibility_block


@pytest.mark.unit
def test_system_prompt_explains_ref_to_action_mapping() -> None:
    """The grammar prompt must tell the model how [ref=eN] becomes [eN].

    This is the minimum the system prompt must carry: an explicit
    description of the snapshot-to-action ref translation, plus the
    one-line / no-markdown reply requirement. Anything richer (worked
    examples, anti-examples, type-then-enter rules) is opt-in prompt
    engineering on top.
    """
    from agent_memories.agent.nodes import THINK_SYSTEM_PROMPT

    assert "EXACTLY one line" in THINK_SYSTEM_PROMPT
    assert "no markdown" in THINK_SYSTEM_PROMPT
    assert "[ref=e29]" in THINK_SYSTEM_PROMPT
    assert "[e29]" in THINK_SYSTEM_PROMPT
    assert "MANDATORY" in THINK_SYSTEM_PROMPT


@pytest.mark.unit
def test_terminal_trace_is_only_observe_think_act(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Terminal output from a Think turn must be a single ``[Think]:`` line.

    The runner deliberately does not dump the LLM prompt, a thinking
    heartbeat, or any other chrome -- the three trace prefixes
    (``[Observe]:``, ``[Think]:``, ``[Act]:``) are the entire user-
    visible channel.
    """
    client = _FakeClient(["click [e1]"])
    think = make_think(client)

    think(_seed_state())

    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert out_lines == ["[Think]: click [e1]"]


@pytest.mark.unit
def test_parse_failure_outcome_has_no_hint_appended() -> None:
    """Parse failures must not pre-encode corrective hints in their outcome.

    Knowledge about which grammar mistakes are common belongs in the
    later memory layer, not baked into the loop. The Think failure
    record carries only the raw parser exception so the LLM sees what
    went wrong without the loop pre-deciding how to fix it.
    """
    client = _FakeClient(['type 7 "paper"'])
    think = make_think(client)

    result = think(_seed_state())

    assert len(result["history"]) == 1
    outcome = result["history"][0]["outcome"]
    assert outcome.startswith("parse_failure:")
    assert "reminder:" not in outcome
    assert "hint" not in outcome.lower()


@pytest.mark.unit
def test_chat_exception_propagates() -> None:
    """Any ``client.chat`` exception must reach the runner, not the history.

    The previous implementation converted transient failures (timeout,
    rate limit) into parse-failure-shaped history records and let the
    loop keep going. That hid real problems (e.g. an unpulled Ollama
    model returning 404 forever) behind ``max_steps`` worth of identical
    "errors". The simplified contract is: anything the chat client
    raises propagates, the runner's ``finally`` block closes the
    browser, and the user sees a real error.
    """

    class _BoomClient:
        def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
            raise TimeoutError("read timed out")

    think = make_think(_BoomClient())

    with pytest.raises(TimeoutError):
        think(_seed_state())
