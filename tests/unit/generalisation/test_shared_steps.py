"""Tests for the InvisibleInk shared-memory step functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agent_memories.agent.invisible_ink.accounting import InvisibleInkAccount
from agent_memories.generalisation.buffer import entry_id
from agent_memories.generalisation.io import (
    load_memory_entries_from_csv,
    render_item_block,
)
from agent_memories.generalisation.step1_labels import run_label_generation
from agent_memories.generalisation.step2_assignment import run_assignment
from agent_memories.generalisation.step3_content import run_content_generation
from agent_memories.generalisation.step4_store import run_store_write
from agent_memories.memory import MemoryEntry, MemoryItem, MemoryStore

pytestmark = pytest.mark.unit


def _account() -> InvisibleInkAccount:
    return InvisibleInkAccount(
        epsilon=1.0,
        delta=1e-5,
        rho=0.1,
        t=8,
        b=2,
        c=1.0,
        tau=1.0,
        top_k=5,
        tokens_used=2,
        topk_plus_mean=4.0,
        topk_plus_std=0.0,
        expansion_set_count=0,
    )


def _entry(
    user_id: str,
    query: str,
    *,
    embedding: list[float],
    content: str = "lesson content",
) -> MemoryEntry:
    return MemoryEntry(
        user_id=user_id,
        query=query,
        outcome="successful",
        items=[
            MemoryItem(
                title="A title",
                description="A description.",
                content=content,
            )
        ],
        embedding=embedding,
        created_at="2026-08-17T00:00:00+00:00",
    )


class _FakeEmbedder:
    """Maps known label strings to axis-aligned vectors."""

    def embed(self, text: str) -> list[float]:
        if text == "label-a":
            return [1.0, 0.0]
        if text == "label-b":
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((system, user))
        return self.reply


def test_step1_prompts_use_query_not_items() -> None:
    captured: dict[str, object] = {}

    def fake_generate(texts: list[str], **kwargs: object) -> tuple[str, InvisibleInkAccount, str]:
        captured["texts"] = list(texts)
        captured["b"] = kwargs["b"]
        captured["k"] = kwargs["k"]
        captured["wrap_fn"] = kwargs["wrap_fn"]
        return '["pricing risk", "supply delay"]', _account(), "production"

    entries = [
        _entry("u0", "Find the cheapest hairbrush", embedding=[1.0, 0.0], content="SECRET ITEM"),
        _entry("u1", "Post a gitlab issue", embedding=[0.0, 1.0], content="SECRET ITEM"),
    ]
    artefact = run_label_generation(entries, k=2, generate_fn=fake_generate)

    assert captured["texts"] == ["Find the cheapest hairbrush", "Post a gitlab issue"]
    assert captured["b"] == 2
    assert captured["k"] == 2
    assert "SECRET ITEM" not in captured["texts"]
    assert artefact["labels"] == ["pricing risk", "supply delay"]
    assert artefact["queries"] == captured["texts"]
    wrap_fn = captured["wrap_fn"]
    assert wrap_fn is not None
    rendered = wrap_fn(items=entries[0].query, k=2)
    assert "Find the cheapest hairbrush" in rendered
    assert "SECRET ITEM" not in rendered


def test_step2_gates_at_seven(tmp_path: Path) -> None:
    labels = ["label-a", "label-b"]
    a_entries = [_entry(f"ua{i}", f"task a {i}", embedding=[1.0, 0.0]) for i in range(7)]
    b_entries = [_entry(f"ub{i}", f"task b {i}", embedding=[0.0, 1.0]) for i in range(2)]
    artefact = run_assignment(
        a_entries + b_entries,
        labels,
        embedder=_FakeEmbedder(),
        x_per_label=7,
    )
    assert artefact["triggered_labels"] == [0]
    assert artefact["bucket_sizes"] == [7, 2]
    assert len(artefact["qualifying"]) == 1
    assert artefact["qualifying"][0]["label"] == "label-a"
    assert len(artefact["qualifying"][0]["entries"]) == 7
    carry = artefact["_carry_over_entries"]
    assert len(carry) == 2
    assert {entry_id(entry) for entry in carry} == {entry_id(entry) for entry in b_entries}


def test_step3_accounts_at_b_seven_with_larger_bucket() -> None:
    captured: dict[str, object] = {}

    def fake_generate(texts: list[str], **kwargs: object) -> tuple[str, InvisibleInkAccount, str]:
        captured["n_texts"] = len(texts)
        captured["b"] = kwargs["b"]
        captured["label"] = kwargs["label"]
        return "A generalisable lesson.", _account(), "production"

    entries = [_entry(f"u{i}", f"task {i}", embedding=[1.0, 0.0]) for i in range(9)]
    qualifying = [
        {
            "label_index": 0,
            "label": "label-a",
            "entries": [entry.to_jsonl_dict() for entry in entries],
        }
    ]
    records = run_content_generation(qualifying, accounting_b=7, generate_fn=fake_generate)
    assert captured["n_texts"] == 9
    assert captured["b"] == 7
    assert captured["label"] == "label-a"
    assert records[0]["content"] == "A generalisable lesson."
    assert records[0]["actual_batch_size"] == 9
    assert records[0]["accounting_b"] == 7
    assert records[0]["chunk_index"] == 0
    assert len(records) == 1


def test_step3_renders_memory_items_not_queries() -> None:
    seen: list[str] = []

    def fake_generate(texts: list[str], **kwargs: object) -> tuple[str, InvisibleInkAccount, str]:
        seen.extend(texts)
        return "lesson", _account(), "production"

    entries = [
        _entry(f"u{i}", f"QUERY {i} MUST NOT APPEAR", embedding=[1.0, 0.0], content=f"CONTENT {i}")
        for i in range(7)
    ]
    qualifying = [
        {
            "label_index": 0,
            "label": "label-a",
            "entries": [entry.to_jsonl_dict() for entry in entries],
        }
    ]
    run_content_generation(qualifying, accounting_b=7, generate_fn=fake_generate)
    assert len(seen) == 7
    for i, block in enumerate(seen):
        assert f"CONTENT {i}" in block
        assert f"QUERY {i}" not in block


@pytest.mark.parametrize(
    "n, expected_sizes",
    [
        (7, [7]),
        (10, [10]),
        (15, [7, 8]),
    ],
)
def test_step3_splits_large_buckets(n: int, expected_sizes: list[int]) -> None:
    calls: list[int] = []

    def fake_generate(texts: list[str], **kwargs: object) -> tuple[str, InvisibleInkAccount, str]:
        calls.append(len(texts))
        return f"lesson-{len(calls)}", _account(), "production"

    entries = [_entry(f"u{i}", f"task {i}", embedding=[1.0, 0.0]) for i in range(n)]
    qualifying = [
        {
            "label_index": 0,
            "label": "label-a",
            "entries": [entry.to_jsonl_dict() for entry in entries],
        }
    ]
    records = run_content_generation(qualifying, accounting_b=7, generate_fn=fake_generate)
    assert calls == expected_sizes
    assert [record["actual_batch_size"] for record in records] == expected_sizes
    assert [record["chunk_index"] for record in records] == list(range(len(expected_sizes)))
    assert all(record["label"] == "label-a" for record in records)
    assert all(record["accounting_b"] == 7 for record in records)
    written_ids = [eid for record in records for eid in record["entry_ids"]]
    assert written_ids == [entry_id(entry) for entry in entries]


def test_step4_writes_shared_store(tmp_path: Path) -> None:
    client = _FakeClient(
        "Title: Confirm Venue Before Paying\n" "Description: Check the issuer page before paying.\n"
    )
    records = [
        {
            "label": "label-a",
            "content": "When booking tickets, confirm the venue on the issuer page.",
        }
    ]
    store_path = tmp_path / "shared.jsonl"
    written = run_store_write(
        records,
        shared_path=store_path,
        client=client,
        embedder=_FakeEmbedder(),
    )
    assert written[0]["title"] == "Confirm Venue Before Paying"
    store = MemoryStore.load(store_path, user_id="shared", embedder=_FakeEmbedder())
    assert len(store) == 1
    entry = store.all()[0]
    assert entry.query == "label-a"
    assert entry.outcome == "shared"
    assert entry.items[0].content == records[0]["content"]


def test_render_item_block_includes_all_fields() -> None:
    entry = _entry("u0", "a query", embedding=[1.0, 0.0], content="the lesson")
    block = render_item_block(entry)
    assert block.startswith("Memory 1:")
    assert "A title" in block
    assert "A description." in block
    assert "the lesson" in block
    assert "a query" not in block


def test_csv_loader_skips_unextracted(tmp_path: Path) -> None:
    path = tmp_path / "memories.csv"
    kept = _entry("u0", "kept query", embedding=[0.1, 0.2])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task_id",
                "memory",
                "embedding",
                "memory_extracted",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "12",
                "memory": json.dumps(kept.to_jsonl_dict()),
                "embedding": json.dumps(kept.embedding),
                "memory_extracted": "True",
            }
        )
        writer.writerow(
            {
                "task_id": "13",
                "memory": "{}",
                "embedding": "[]",
                "memory_extracted": "False",
            }
        )
    pairs = load_memory_entries_from_csv(path)
    assert len(pairs) == 1
    assert pairs[0][0] == 12
    assert pairs[0][1].query == "kept query"
