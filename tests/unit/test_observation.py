"""Tests for the ARIA-snapshot wrapper."""

from __future__ import annotations

import pytest

from agent_memories.agent.observation import format_aria_snapshot


class _FakePage:
    """Page double exposing only aria_snapshot."""

    def __init__(self, snapshot: str | None) -> None:
        self._snapshot = snapshot
        self.calls: list[dict[str, object]] = []

    def aria_snapshot(self, *, mode: str = "default") -> str | None:
        self.calls.append({"mode": mode})
        return self._snapshot


@pytest.mark.unit
def test_returns_snapshot_yaml_unchanged() -> None:
    yaml = '- generic [ref=e1]:\n  - button "Go" [ref=e2]\n'
    page = _FakePage(yaml)

    assert format_aria_snapshot(page) == yaml  # type: ignore[arg-type]


@pytest.mark.unit
def test_requests_ai_mode() -> None:
    page = _FakePage("- generic [ref=e1]\n")
    format_aria_snapshot(page)  # type: ignore[arg-type]

    assert page.calls == [{"mode": "ai"}]


@pytest.mark.unit
def test_none_snapshot_becomes_empty_string() -> None:
    page = _FakePage(None)
    assert format_aria_snapshot(page) == ""  # type: ignore[arg-type]


@pytest.mark.unit
def test_empty_snapshot_becomes_empty_string() -> None:
    page = _FakePage("")
    assert format_aria_snapshot(page) == ""  # type: ignore[arg-type]
