"""Persistence and cosine retrieval in MemoryStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memories.memory.store import MemoryEntry, MemoryItem, MemoryStore
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.unit


def _make_store(tmp_path: Path, mapping: dict[str, list[float]]) -> MemoryStore:
    path = tmp_path / "user_a.jsonl"
    return MemoryStore(path=path, user_id="user_a", embedder=FakeEmbedder(mapping))


def _item(title: str = "t", description: str = "d", content: str = "c") -> MemoryItem:
    return MemoryItem(title=title, description=description, content=content)


def test_add_entry_persists_and_returns_entry(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"first query": [1.0, 0.0, 0.0]})

    entry = store.add_entry(
        query="first query",
        outcome="successful",
        items=[_item(title="t1", description="d1", content="c1")],
    )

    assert entry.query == "first query"
    assert entry.user_id == "user_a"
    assert entry.outcome == "successful"
    assert entry.embedding == [1.0, 0.0, 0.0]
    assert entry.created_at != ""
    assert [it.title for it in entry.items] == ["t1"]

    raw_line = store.path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw_line)
    assert parsed["query"] == "first query"
    assert parsed["user_id"] == "user_a"
    assert parsed["outcome"] == "successful"
    assert parsed["embedding"] == [1.0, 0.0, 0.0]
    assert parsed["items"] == [{"title": "t1", "description": "d1", "content": "c1"}]


def test_search_ranks_by_query_cosine(tmp_path: Path) -> None:
    mapping = {
        "apples": [1.0, 0.0, 0.0],
        "pears": [0.9, 0.1, 0.0],
        "trains": [0.0, 1.0, 0.0],
        "query about fruit": [1.0, 0.0, 0.0],
    }
    store = _make_store(tmp_path, mapping)
    for q in ("apples", "pears", "trains"):
        store.add_entry(query=q, outcome="successful", items=[_item(content=q)])

    top2 = store.search("query about fruit", k=2)

    assert [e.query for e in top2] == ["apples", "pears"]


def test_search_on_empty_store(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"anything": [1.0]})

    assert store.search("anything", k=3) == []


def test_search_caps_k_at_store_size(tmp_path: Path) -> None:
    mapping = {
        "one": [1.0, 0.0],
        "two": [0.0, 1.0],
        "query": [1.0, 0.0],
    }
    store = _make_store(tmp_path, mapping)
    for q in ("one", "two"):
        store.add_entry(query=q, outcome="successful", items=[_item(content=q)])

    results = store.search("query", k=10)

    assert len(results) == 2
    assert results[0].query == "one"


def test_load_reuses_stored_embeddings(tmp_path: Path) -> None:
    seed_mapping = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "query": [1.0, 0.0, 0.0],
    }
    store = _make_store(tmp_path, seed_mapping)
    for q in ("alpha", "beta"):
        store.add_entry(query=q, outcome="successful", items=[_item(content=q)])

    poison_mapping = {
        "alpha": [0.0, 1.0, 0.0],
        "beta": [1.0, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0],
    }
    reloaded = MemoryStore.load(
        store.path,
        user_id="user_a",
        embedder=FakeEmbedder(poison_mapping),
    )

    assert len(reloaded) == 2
    top1 = reloaded.search("query", k=1)
    assert top1[0].query == "alpha"


def test_load_missing_file(tmp_path: Path) -> None:
    embedder = FakeEmbedder({"x": [1.0]})
    store = MemoryStore.load(
        tmp_path / "does_not_exist.jsonl",
        user_id="user_a",
        embedder=embedder,
    )

    assert len(store) == 0
    assert store.search("x", k=1) == []


def test_add_entry_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "user_a.jsonl"
    store = MemoryStore(
        path=nested,
        user_id="user_a",
        embedder=FakeEmbedder({"hello": [1.0]}),
    )

    store.add_entry(query="hello", outcome="successful", items=[_item(content="hi")])

    assert nested.exists()


def test_from_jsonl_dict_defaults_optional_fields() -> None:
    record = {
        "user_id": "user_a",
        "query": "minimal",
        "outcome": "failed",
        "items": [{"title": "t", "description": "d", "content": "c"}],
    }

    entry = MemoryEntry.from_jsonl_dict(record)

    assert entry.query == "minimal"
    assert entry.outcome == "failed"
    assert entry.embedding is None
    assert entry.created_at == ""
    assert entry.items[0].title == "t"


def test_unknown_outcome_defaults_to_failed() -> None:
    record = {
        "user_id": "user_a",
        "query": "q",
        "outcome": "yes-please",
        "items": [],
    }

    entry = MemoryEntry.from_jsonl_dict(record)

    assert entry.outcome == "failed"
