"""Unit tests for BrowserGym WebArena action parsing."""

from __future__ import annotations

import pytest

from agent_memories.agent.browsergym.actions import ActionParseError, parse_action
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
