"""Tests for BrowserGym WebArena action parsing."""

from __future__ import annotations

import pytest

from agent_memories.agent.browsergym.actions import (
    ActionParseError,
    extract_action_line,
    parse_action,
)
from agent_memories.agent.browsergym.env import THINK_SYSTEM_PROMPT


def test_parse_click() -> None:
    assert parse_action("click('a12')") == "click('a12')"


def test_parse_scroll() -> None:
    assert parse_action("scroll(0, 200)") == "scroll(0, 200)"


def test_parse_noop() -> None:
    assert parse_action("noop()") == "noop()"


def test_parse_send_msg_to_user() -> None:
    assert parse_action("send_msg_to_user('Quest Band')") == "send_msg_to_user('Quest Band')"


def test_parse_rejects_stop() -> None:
    with pytest.raises(ActionParseError):
        parse_action("stop")


def test_parse_rejects_multiline() -> None:
    with pytest.raises(ActionParseError):
        parse_action("click('a')\nfill('b', 'x')")


def test_think_system_prompt_from_describe() -> None:
    assert "scroll(0, 200)" in THINK_SYSTEM_PROMPT
    assert "Only a single action can be provided at once" in THINK_SYSTEM_PROMPT


def test_extract_fenced_python_block() -> None:
    reply = "```python\nclick('53')\n```"
    assert extract_action_line(reply) == "click('53')"


def test_extract_preamble_then_action() -> None:
    reply = "I will use the search box next.\nclick('53')"
    assert extract_action_line(reply) == "click('53')"


def test_extract_last_of_several_candidate_lines() -> None:
    reply = "click('10')\nfill('20', 'q')\nsend_msg_to_user('done')"
    assert extract_action_line(reply) == "send_msg_to_user('done')"


def test_extract_no_candidate_returns_stripped_text() -> None:
    reply = "I am not sure what to click."
    assert extract_action_line(reply) == "I am not sure what to click."
    with pytest.raises(ActionParseError):
        parse_action(extract_action_line(reply))


def test_extract_drops_leaked_think_block() -> None:
    reply = "<think>Consider the search button.</think>\nclick('53')"
    assert extract_action_line(reply) == "click('53')"


def test_extract_unterminated_think_with_no_action() -> None:
    reply = "<think>Still reasoning about the page and never finishes"
    assert extract_action_line(reply) == ""
    with pytest.raises(ActionParseError):
        parse_action(extract_action_line(reply))
