"""The one-line action grammar the Playwright Think node emits."""

from __future__ import annotations

import pytest

from agent_memories.agent.playwright.actions import ActionParseError, parse_action

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("click [e1]", {"type": "click", "ref": "e1"}),
        ("click [e12]", {"type": "click", "ref": "e12"}),
        ('type [e34] "London"', {"type": "fill", "ref": "e34", "value": "London"}),
        ('type [e1] ""', {"type": "fill", "ref": "e1", "value": ""}),
        ("enter", {"type": "press", "key": "Enter"}),
        ("scroll up", {"type": "scroll", "direction": "up"}),
        ("scroll down", {"type": "scroll", "direction": "down"}),
        (
            'goto "https://example.com/path"',
            {"type": "goto", "url": "https://example.com/path"},
        ),
        ("stop", {"type": "stop"}),
    ],
)
def test_parses_canonical_grammar(line: str, expected: dict[str, object]) -> None:
    assert parse_action(line) == expected


def test_strips_surrounding_whitespace() -> None:
    assert parse_action("  click [e7]  ") == {"type": "click", "ref": "e7"}


def test_type_action_supports_escaped_quotes() -> None:
    parsed = parse_action(r'type [e2] "she said \"hi\""')
    assert parsed == {"type": "fill", "ref": "e2", "value": 'she said "hi"'}


def test_type_action_supports_escaped_backslash() -> None:
    parsed = parse_action(r'type [e2] "a\\b"')
    assert parsed == {"type": "fill", "ref": "e2", "value": "a\\b"}


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "click",
        "click 12",
        "click [12]",  # bare integer form is rejected: ARIA-ref only
        "click [e]",
        "click  e1",
        "type [e1]",
        "type [e1] London",  # missing quotes
        'type [e1] "missing close',
        "press [e1] Enter",  # unsupported verb
        "click [e1]; click [e2]",
        "click [e1]\nclick [e2]",
        "scroll",
        "scroll left",
        "scroll up 500",  # pixel amounts not part of the grammar
        "goto example.com",  # missing quotes
        'goto "missing close',
        "press Enter",  # grammar uses bare 'enter', not 'press X'
        'press "Enter"',
        "enter [e1]",  # 'enter' takes no arguments
    ],
)
def test_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(line)


def test_none_raises_parse_error() -> None:
    with pytest.raises(ActionParseError):
        parse_action(None)  # type: ignore[arg-type]
