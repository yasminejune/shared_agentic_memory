"""Unit tests for :class:`MemoryPipeline`.

A ``FakeClient`` returns a queued reply per call so no network is hit;
a ``FakeEmbedder`` keeps similarity behaviour deterministic. The tests
cover:

* a ``"None"``-shaped reply (in any case, optionally quoted) produces
  no new memory and writes nothing to disk;
* a non-``"None"`` reply is appended to the store verbatim;
* the user prompt the curator LLM sees carries the run summary and a
  numbered list of the top-k existing memories.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memories.agent.state import new_state
from agent_memories.memory.pipeline import MemoryPipeline
from agent_memories.memory.store import MemoryStore


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = dict(mapping or {})
        self._dim = 3

    def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0] * self._dim))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _FakeClient:
    """Chat double that returns a queued reply per call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, str]] = []

    def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user})
        return self._replies.pop(0)


def _make_store(tmp_path: Path, mapping: dict[str, list[float]] | None = None) -> MemoryStore:
    return MemoryStore(
        path=tmp_path / "user_a.jsonl",
        user_id="user_a",
        embedder=_FakeEmbedder(mapping),
    )


@pytest.mark.unit
def test_none_reply_does_not_add_memory(tmp_path: Path) -> None:
    """The curator's ``"None"`` reply must result in no new memory."""
    store = _make_store(tmp_path)
    client = _FakeClient(["None"])
    pipeline = MemoryPipeline(client=client, store=store)

    state = new_state(aim="aim")
    state["done"] = True

    result = pipeline.create_from_run(state)

    assert result is None
    assert len(store) == 0
    assert not store.path.exists()


@pytest.mark.unit
def test_none_reply_is_case_and_quote_insensitive(tmp_path: Path) -> None:
    """Variations like ``"none"``, ``'None'`` must also be treated as no-op."""
    for variant in ("none", "  NONE\n", '"None"', "'none'"):
        store = _make_store(tmp_path / variant.strip(" '\"\n").lower())
        pipeline = MemoryPipeline(
            client=_FakeClient([variant]),
            store=store,
        )
        assert pipeline.create_from_run(new_state(aim="x")) is None
        assert len(store) == 0


@pytest.mark.unit
def test_non_none_reply_is_stored_verbatim(tmp_path: Path) -> None:
    """Any other reply text is appended to the store as a new memory."""
    store = _make_store(tmp_path)
    client = _FakeClient(["Always dismiss the cookie banner before searching."])
    pipeline = MemoryPipeline(client=client, store=store)

    result = pipeline.create_from_run(new_state(aim="aim"))

    assert result is not None
    assert result.text == "Always dismiss the cookie banner before searching."
    assert len(store) == 1
    stored = store.all()[0]
    assert stored.text == result.text


@pytest.mark.unit
def test_reply_whitespace_is_stripped_before_storing(tmp_path: Path) -> None:
    """Curator output often has trailing newlines; the store must not see them."""
    store = _make_store(tmp_path)
    client = _FakeClient(["  Use Enter to submit the search.  \n"])
    pipeline = MemoryPipeline(client=client, store=store)

    result = pipeline.create_from_run(new_state(aim="aim"))

    assert result is not None
    assert result.text == "Use Enter to submit the search."


@pytest.mark.unit
def test_prompt_contains_run_summary_and_top_k_block(tmp_path: Path) -> None:
    """The user prompt must include both the run summary and existing memories."""
    mapping = {
        "Aim: shop": [1.0, 0.0, 0.0],
        "cookie banner": [1.0, 0.0, 0.0],
        "unrelated train": [0.0, 1.0, 0.0],
    }
    store = _make_store(tmp_path, mapping)
    store.add_many(["cookie banner", "unrelated train"])

    client = _FakeClient(["None"])
    pipeline = MemoryPipeline(client=client, store=store, k_for_dedup=5)

    state = new_state(aim="shop")
    state["url"] = "https://example.com"
    state["step"] = 2
    state["done"] = True
    state["history"] = [
        {"step": 0, "thought": "click [e1]", "action": {"type": "click"}, "outcome": "ok"},
        {"step": 1, "thought": "stop", "action": {"type": "stop"}, "outcome": "stop"},
    ]

    pipeline.create_from_run(state)

    assert len(client.calls) == 1
    user_prompt = client.calls[0]["user"]
    assert "Run:" in user_prompt
    assert "Aim: shop" in user_prompt
    assert "Final URL: https://example.com" in user_prompt
    assert "Steps taken: 2" in user_prompt
    assert "Trajectory (most recent last):" in user_prompt
    assert "'click [e1]' -> ok" in user_prompt
    assert "Existing memories (top-5 most similar" in user_prompt
    assert "1. cookie banner" in user_prompt
    assert "2. unrelated train" in user_prompt


@pytest.mark.unit
def test_prompt_handles_empty_memory_bank(tmp_path: Path) -> None:
    """When the store is empty the prompt must say so rather than crash."""
    store = _make_store(tmp_path)
    client = _FakeClient(["None"])
    pipeline = MemoryPipeline(client=client, store=store)

    pipeline.create_from_run(new_state(aim="aim"))

    user_prompt = client.calls[0]["user"]
    assert "(the memory bank is empty)" in user_prompt


@pytest.mark.unit
def test_k_for_dedup_caps_existing_memories_in_prompt(tmp_path: Path) -> None:
    """Only the configured number of nearest existing memories should be shown."""
    mapping = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.9, 0.1, 0.0],
        "gamma": [0.8, 0.2, 0.0],
        "delta": [0.7, 0.3, 0.0],
    }
    store = _make_store(tmp_path, mapping)
    store.add_many(["alpha", "beta", "gamma", "delta"])

    client = _FakeClient(["None"])
    pipeline = MemoryPipeline(client=client, store=store, k_for_dedup=2)

    pipeline.create_from_run(new_state(aim="alpha"))

    user_prompt = client.calls[0]["user"]
    assert "1. " in user_prompt and "2. " in user_prompt
    assert "3. " not in user_prompt
    assert "top-2 most similar" in user_prompt
