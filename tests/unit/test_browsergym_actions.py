"""Unit tests for BrowserGym WebArena action parsing."""

from __future__ import annotations

import pytest

from agent_memories.agent.browsergym.actions import ActionParseError, parse_action
from agent_memories.agent.browsergym.env import (
    THINK_SYSTEM_PROMPT,
    WEBARENA_ACTION_SET,
    webarena_action_to_python,
)


def test_parse_click() -> None:
    assert parse_action("click('a12')") == "click('a12')"


def test_parse_scroll() -> None:
    assert parse_action("scroll(0, 200)") == "scroll(0, 200)"


def test_parse_noop() -> None:
    assert parse_action("noop()") == "noop()"


def test_parse_send_msg_to_user() -> None:
    assert parse_action("send_msg_to_user('Quest Band')") == "send_msg_to_user('Quest Band')"


def test_parse_goto() -> None:
    assert parse_action("goto('http://localhost:9999/f/books')") == (
        "goto('http://localhost:9999/f/books')"
    )


def test_parse_go_home() -> None:
    assert parse_action("go_home()") == "go_home()"


def test_parse_rejects_go_forward() -> None:
    with pytest.raises(ActionParseError, match="Disallowed action function"):
        parse_action("go_forward()")


def test_parse_rejects_stop() -> None:
    with pytest.raises(ActionParseError):
        parse_action("stop")


def test_parse_rejects_multiline() -> None:
    with pytest.raises(ActionParseError):
        parse_action("click('a')\nfill('b', 'x')")


def test_think_system_prompt_from_describe() -> None:
    assert "scroll(0, 200)" in THINK_SYSTEM_PROMPT
    assert "Only a single action can be provided at once" in THINK_SYSTEM_PROMPT
    assert "go_home()" in THINK_SYSTEM_PROMPT
    assert "goto('http://localhost:9999/f/books')" in THINK_SYSTEM_PROMPT
    assert "go_forward" not in THINK_SYSTEM_PROMPT


def test_action_set_describe_parses() -> None:
    grammar = WEBARENA_ACTION_SET.describe(with_long_description=False, with_examples=True)
    assert "go_home()" in grammar
    assert "goto(url: str)" in grammar
    assert "go_forward" not in grammar
    assert "go_forward" not in WEBARENA_ACTION_SET.action_set
    assert "goto" in WEBARENA_ACTION_SET.action_set
    assert "go_home" in WEBARENA_ACTION_SET.action_set


def test_webarena_action_to_python_compiles() -> None:
    for action in (
        "click('a51')",
        "go_home()",
        "goto('https://www.reddit.com/r/books')",
        "noop()",
    ):
        code = webarena_action_to_python(action)
        compile(code, "<string>", "exec")
        assert code.rstrip().endswith(action) or code.rstrip().endswith(action + ")")
    code = webarena_action_to_python("goto('https://www.reddit.com/')")
    assert "def goto" in code
    assert "resolve_local_url" in code
