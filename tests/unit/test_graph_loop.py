"""Unit tests for the graph-level safety properties added in WP1.3.

These tests run the compiled LangGraph but use a fake Page so they
stay browser-free. They verify:

* ``make_act`` captures dispatch exceptions into ``history`` instead of
  crashing the graph;
* the ``max_steps`` cap terminates a loop that would otherwise run
  indefinitely (parse failures forever);
* a parse failure routes back to Observe and the next Think turn sees
  the failure in ``history``.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.nodes import make_act, make_think


class _FakePage:
    """Tiny Page double that exposes only what Observe and Act need."""

    def __init__(self, *, snapshot: str = "- button [ref=e1]") -> None:
        self.url = "about:blank"
        self._snapshot = snapshot
        self.calls: list[dict[str, Any]] = []
        self._raise_on_next = False

    def title(self) -> str:
        return "fake"

    def aria_snapshot(self, *, mode: str = "default") -> str:
        return self._snapshot

    def locator(self, target: str) -> _FakeLocator:
        return _FakeLocator(self, target)

    def goto(self, url: str) -> None:
        self.calls.append({"goto": url})

    @property
    def keyboard(self) -> _FakeKeyboard:
        return _FakeKeyboard(self)


class _FakeKeyboard:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def press(self, key: str) -> None:
        self._page.calls.append({"press": key})


class _FakeLocator:
    def __init__(self, page: _FakePage, target: str) -> None:
        self._page = page
        self._target = target

    def click(self) -> None:
        if self._page._raise_on_next:
            self._page._raise_on_next = False
            raise RuntimeError("locator detached")
        self._page.calls.append({"click": self._target})

    def fill(self, value: str) -> None:
        self._page.calls.append({"fill": self._target, "value": value})


class _ScriptedClient:
    """Chat client double that returns a queued reply per call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        if not self._replies:
            return "stop"
        return self._replies.pop(0)


@pytest.mark.unit
def test_act_captures_dispatch_exceptions_into_history() -> None:
    page = _FakePage()
    page._raise_on_next = True
    act = make_act(page)  # type: ignore[arg-type]

    result = act(
        {
            **new_state(aim="x"),
            "thought": "click [e1]",
            "action": {"type": "click", "ref": "e1"},
        }
    )

    assert result["step"] == 1
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert record["action"] == {"type": "click", "ref": "e1"}
    assert record["outcome"].startswith("error: RuntimeError: locator detached")


@pytest.mark.unit
def test_max_steps_caps_a_parse_failure_loop() -> None:
    """An LLM that never produces a parseable action must still terminate."""
    page = _FakePage()
    # The client always returns nonsense, so every Think is a parse failure.
    client = _ScriptedClient(["nonsense"] * 100)
    graph = build_graph(page, make_think(client), max_steps=3)  # type: ignore[arg-type]

    result = graph.invoke(new_state(aim="x"))

    assert result["step"] == 3
    assert result["done"] is False, "termination is via the cap, not a stop signal"
    assert len(result["history"]) == 3
    assert all(r["outcome"].startswith("parse_failure:") for r in result["history"])


@pytest.mark.unit
def test_parse_failure_then_recovery_threads_history() -> None:
    page = _FakePage()
    client = _ScriptedClient(["I'll click", "click [e1]", "stop"])
    graph = build_graph(page, make_think(client), max_steps=10)  # type: ignore[arg-type]

    result = graph.invoke(new_state(aim="x"))

    assert result["done"] is True
    # parse failure -> click -> stop, so history has 3 entries.
    assert len(result["history"]) == 3
    outcomes = [r["outcome"] for r in result["history"]]
    assert outcomes[0].startswith("parse_failure:")
    assert outcomes[1] == "ok"
    assert outcomes[2] == "stop"
    assert {"click": "aria-ref=e1"} in page.calls


class _AlwaysTimingOutClient:
    """Chat client double that always raises a timeout-style exception."""

    def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        raise TimeoutError("read timed out after 60s")


@pytest.mark.unit
def test_loop_propagates_chat_exceptions() -> None:
    """A chat-client exception must escape the graph, not be swallowed.

    The simplified contract is: any ``client.chat`` failure (timeout,
    HTTP 404 for an unpulled Ollama model, auth error) propagates up
    through the graph. The runner's ``finally`` block then closes the
    browser and the user gets one real error instead of ``max_steps``
    iterations of identical failures.
    """
    page = _FakePage()
    graph = build_graph(page, make_think(_AlwaysTimingOutClient()), max_steps=4)  # type: ignore[arg-type]

    with pytest.raises(TimeoutError):
        graph.invoke(new_state(aim="x"))
