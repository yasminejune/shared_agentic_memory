"""Unit tests for :class:`MemoryStore`.

A deterministic in-memory ``FakeEmbedder`` is injected so the tests are
fast and never load the real sentence-transformers model. The fake
maps each known string to a hand-picked unit vector so cosine
similarities are predictable: identical texts hit similarity ``1.0``,
orthogonal vectors hit ``0``, and the top-k ranking can be reasoned
about by reading the test alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memories.memory.store import Memory, MemoryStore


class _FakeEmbedder:
    """Deterministic embedder backed by an explicit text -> vector mapping.

    Texts not present in the mapping fall back to a zero vector of the
    same dimensionality as the first registered vector. Callers should
    register every test string up front to keep the cosine results
    unambiguous.
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = {k: list(v) for k, v in mapping.items()}
        first = next(iter(self._mapping.values()), [0.0])
        self._dim = len(first)

    def embed(self, text: str) -> list[float]:
        if text not in self._mapping:
            return [0.0] * self._dim
        return list(self._mapping[text])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make_store(tmp_path: Path, mapping: dict[str, list[float]]) -> MemoryStore:
    path = tmp_path / "user_a.jsonl"
    return MemoryStore(path=path, user_id="user_a", embedder=_FakeEmbedder(mapping))


@pytest.mark.unit
def test_add_persists_to_disk_and_returns_memory(tmp_path: Path) -> None:
    """``add`` must write a JSONL line and return a populated :class:`Memory`."""
    store = _make_store(tmp_path, {"first memory": [1.0, 0.0, 0.0]})

    memory = store.add("first memory")

    assert memory.text == "first memory"
    assert memory.user_id == "user_a"
    assert memory.embedding == [1.0, 0.0, 0.0]
    assert memory.created_at != ""

    raw_line = store.path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw_line)
    assert parsed["text"] == "first memory"
    assert parsed["user_id"] == "user_a"
    assert parsed["embedding"] == [1.0, 0.0, 0.0]


@pytest.mark.unit
def test_search_returns_top_k_by_cosine(tmp_path: Path) -> None:
    """``search`` must return memories sorted by descending cosine similarity."""
    mapping = {
        "apples": [1.0, 0.0, 0.0],
        "pears": [0.9, 0.1, 0.0],
        "trains": [0.0, 1.0, 0.0],
        "query about fruit": [1.0, 0.0, 0.0],
    }
    store = _make_store(tmp_path, mapping)
    store.add_many(["apples", "pears", "trains"])

    top2 = store.search("query about fruit", k=2)

    assert [m.text for m in top2] == ["apples", "pears"]


@pytest.mark.unit
def test_search_returns_empty_on_empty_store(tmp_path: Path) -> None:
    """A query against an empty store must produce an empty result list."""
    store = _make_store(tmp_path, {"anything": [1.0]})

    assert store.search("anything", k=3) == []


@pytest.mark.unit
def test_search_caps_at_store_size(tmp_path: Path) -> None:
    """Requesting more neighbours than exist returns all of them, ranked."""
    mapping = {
        "one": [1.0, 0.0],
        "two": [0.0, 1.0],
        "query": [1.0, 0.0],
    }
    store = _make_store(tmp_path, mapping)
    store.add_many(["one", "two"])

    results = store.search("query", k=10)

    assert len(results) == 2
    assert results[0].text == "one"


@pytest.mark.unit
def test_load_round_trip_skips_re_embedding(tmp_path: Path) -> None:
    """A second ``MemoryStore.load`` must read the JSONL without re-embedding.

    We swap in a *different* embedder (mapping every string to the wrong
    vector) and check that ``search`` still works against the cached
    embeddings persisted on disk. If the store ever re-embeds on
    reload, the second store's search would rank by the new -- wrong --
    embeddings and fail this test.
    """
    seed_mapping = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "query": [1.0, 0.0, 0.0],
    }
    store = _make_store(tmp_path, seed_mapping)
    store.add_many(["alpha", "beta"])

    poison_mapping = {
        "alpha": [0.0, 1.0, 0.0],
        "beta": [1.0, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0],
    }
    reloaded = MemoryStore.load(
        store.path,
        user_id="user_a",
        embedder=_FakeEmbedder(poison_mapping),
    )

    assert len(reloaded) == 2
    top1 = reloaded.search("query", k=1)
    assert top1[0].text == "alpha"


@pytest.mark.unit
def test_load_missing_file_yields_empty_store(tmp_path: Path) -> None:
    """A non-existent path must produce an empty store, not an error."""
    embedder = _FakeEmbedder({"x": [1.0]})
    store = MemoryStore.load(
        tmp_path / "does_not_exist.jsonl",
        user_id="user_a",
        embedder=embedder,
    )

    assert len(store) == 0
    assert store.search("x", k=1) == []


@pytest.mark.unit
def test_add_creates_parent_directory(tmp_path: Path) -> None:
    """The store must create missing parent directories on first ``add``."""
    nested = tmp_path / "deep" / "nested" / "user_a.jsonl"
    store = MemoryStore(
        path=nested,
        user_id="user_a",
        embedder=_FakeEmbedder({"hello": [1.0]}),
    )

    store.add("hello")

    assert nested.exists()


@pytest.mark.unit
def test_memory_from_jsonl_dict_handles_missing_optional_fields() -> None:
    """Older or hand-written JSONL records may omit metadata/created_at."""
    record = {"text": "minimal", "user_id": "user_a"}

    memory = Memory.from_jsonl_dict(record)

    assert memory.text == "minimal"
    assert memory.metadata == {}
    assert memory.embedding is None
    assert memory.created_at == ""
