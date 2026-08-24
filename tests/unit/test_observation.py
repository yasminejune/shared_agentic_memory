"""The ARIA-snapshot wrapper feeding the Playwright Observe node."""

from __future__ import annotations

import pytest

from agent_memories.agent.playwright.observation import format_aria_snapshot

pytestmark = pytest.mark.unit


class _FakePage:
    """Page double exposing only aria_snapshot."""

    def __init__(self, snapshot: str | None) -> None:
        self._snapshot = snapshot
        self.calls: list[dict[str, object]] = []

    def aria_snapshot(self, *, mode: str = "default") -> str | None:
        self.calls.append({"mode": mode})
        return self._snapshot


def test_returns_snapshot_yaml_unchanged() -> None:
    yaml = '- generic [ref=e1]:\n  - button "Go" [ref=e2]\n'
    page = _FakePage(yaml)

    assert format_aria_snapshot(page) == yaml  # type: ignore[arg-type]


def test_requests_ai_mode() -> None:
    page = _FakePage("- generic [ref=e1]\n")
    format_aria_snapshot(page)  # type: ignore[arg-type]

    assert page.calls == [{"mode": "ai"}]


@pytest.mark.parametrize("snapshot", [None, ""])
def test_absent_snapshot_becomes_empty_string(snapshot: str | None) -> None:
    assert format_aria_snapshot(_FakePage(snapshot)) == ""  # type: ignore[arg-type]
