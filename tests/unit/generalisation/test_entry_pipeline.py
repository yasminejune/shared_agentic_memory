"""Unit tests for entry-level WP2 batching and privacy accounting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from scripts.memories import WP2_pipeline

from agent_memories.generalisation import (
    assign_memories_to_labels,
    entry_id,
    read_buffer,
    write_buffer,
)
from agent_memories.memory import MemoryEntry, MemoryItem


class _UnusedEmbedder:
    """Embedder double for records that already contain embeddings."""

    def embed(self, text: str) -> list[float]:
        raise AssertionError(f"Unexpected embedding request for {text!r}")


def _entry(
    user_id: str,
    query: str,
    created_at: str,
    *,
    n_items: int = 1,
    embedding: list[float] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        user_id=user_id,
        query=query,
        outcome="successful",
        items=[
            MemoryItem(
                title=f"Title {i}",
                description=f"Description {i}.",
                content=f"Content {i}.",
            )
            for i in range(n_items)
        ],
        embedding=embedding if embedding is not None else [1.0, 0.0],
        created_at=created_at,
    )


def test_entry_id_omits_item_index() -> None:
    entry = _entry(
        "user_00",
        "Find a product",
        "2026-08-10T10:00:00+00:00",
        n_items=3,
    )

    assert entry_id(entry) == "user_00::Find a product::2026-08-10T10:00:00+00:00"


def test_cycle_assembly_takes_whole_entries_not_sibling_items(tmp_path) -> None:
    first = _entry(
        "user_00",
        "First task",
        "2026-08-10T10:00:00+00:00",
        n_items=3,
    )
    second = _entry(
        "user_01",
        "Second task",
        "2026-08-10T11:00:00+00:00",
    )
    third = _entry(
        "user_02",
        "Third task",
        "2026-08-10T12:00:00+00:00",
    )
    for entry in (first, second, third):
        path = tmp_path / f"{entry.user_id}.jsonl"
        path.write_text(json.dumps(entry.to_jsonl_dict()) + "\n", encoding="utf-8")

    new_entries, assignment_batch, processed_ids = WP2_pipeline._assemble_cycle_entries(
        tmp_path,
        set(),
        [],
        _UnusedEmbedder(),
        entries_per_cycle=2,
    )

    assert [entry.query for entry in new_entries] == ["First task", "Second task"]
    assert assignment_batch == new_entries
    assert processed_ids == set()
    assert all(len(entry.items) == expected for entry, expected in zip(new_entries, (3, 1)))

    processed_ids.update(entry_id(entry) for entry in new_entries)
    next_entries, _, _ = WP2_pipeline._assemble_cycle_entries(
        tmp_path,
        processed_ids,
        [],
        _UnusedEmbedder(),
        entries_per_cycle=2,
    )

    assert next_entries == []


def test_cycle_waits_when_fewer_than_b_new_entries_are_available(tmp_path) -> None:
    entry = _entry(
        "user_00",
        "Only task",
        "2026-08-10T10:00:00+00:00",
        n_items=2,
    )
    (tmp_path / "user_00.jsonl").write_text(
        json.dumps(entry.to_jsonl_dict()) + "\n",
        encoding="utf-8",
    )

    new_entries, assignment_batch, processed_ids = WP2_pipeline._assemble_cycle_entries(
        tmp_path,
        set(),
        [],
        _UnusedEmbedder(),
        entries_per_cycle=3,
    )

    assert new_entries == []
    assert assignment_batch == []
    assert processed_ids == set()


def test_missing_embedding_raises_when_loading_entries(tmp_path) -> None:
    entry = _entry(
        "user_00",
        "Task",
        "2026-08-10T10:00:00+00:00",
        embedding=None,
    )
    payload = entry.to_jsonl_dict()
    payload["embedding"] = None
    (tmp_path / "user_00.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no query embedding"):
        WP2_pipeline._load_per_user_entries(tmp_path, set(), _UnusedEmbedder())


def test_assignment_raises_on_missing_embedding() -> None:
    entry = _entry(
        "user_00",
        "Task",
        "2026-08-10T10:00:00+00:00",
    )
    entry.embedding = None

    class _Embedder:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    with pytest.raises(ValueError, match="has no query embedding"):
        assign_memories_to_labels([entry], ["Label"], embedder=_Embedder())


def test_entry_buffer_round_trips_and_rejects_item_level_records(tmp_path) -> None:
    entry = _entry("user_00", "Task", "2026-08-10T10:00:00+00:00", n_items=2)
    path = tmp_path / ".shared_buffer.jsonl"

    write_buffer([entry], path)

    assert read_buffer(path) == [entry]

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "user_id": "user_00",
                "query": "Legacy",
                "outcome": "successful",
                "item": {"title": "T", "description": "D", "content": "C"},
                "item_index": 0,
                "embedding": [1.0, 0.0],
                "created_at": "2026-08-09T10:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy item-level"):
        read_buffer(path)


def test_checkpoint_rejects_legacy_item_level_state(tmp_path) -> None:
    path = tmp_path / ".shared_state.json"
    path.write_text(
        json.dumps({"schema_version": 2, "step1_processed_item_ids": ["legacy-item"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="legacy item-level"):
        WP2_pipeline._load_checkpoint(path)


def test_round1_uses_one_prompt_row_per_entry(monkeypatch) -> None:
    entries = [
        _entry(
            "user_00",
            "Repeated parent query",
            "2026-08-10T10:00:00+00:00",
            n_items=3,
        )
    ]
    observed: dict[str, object] = {}

    monkeypatch.setattr(WP2_pipeline, "solve_r", lambda *args, **kwargs: 1)
    monkeypatch.setattr(WP2_pipeline, "rho_for", lambda *args: 0.1)
    monkeypatch.setattr(WP2_pipeline, "epsilon_from_rho", lambda *args: 1.0)
    monkeypatch.setattr(
        WP2_pipeline,
        "generate",
        lambda texts, **kwargs: (
            observed.update(texts=texts, s=kwargs["s"], delta=kwargs["delta"]) or '["search"]',
            object(),
        ),
    )

    WP2_pipeline._run_round1(
        entries,
        expected_batch_size=1,
        k_labels=1,
        target_epsilon=5.0,
        delta=1e-5,
    )

    assert observed == {
        "texts": ["Repeated parent query"],
        "s": 1,
        "delta": 1e-5,
    }


def test_round2_uses_one_numbered_block_per_entry(monkeypatch) -> None:
    entries = [
        _entry(
            "user_00",
            "Task",
            "2026-08-10T10:00:00+00:00",
            n_items=3,
        )
    ]
    observed: dict[str, object] = {}

    monkeypatch.setattr(WP2_pipeline, "solve_r", lambda *args, **kwargs: 1)
    monkeypatch.setattr(WP2_pipeline, "rho_for", lambda *args: 0.1)
    monkeypatch.setattr(WP2_pipeline, "epsilon_from_rho", lambda *args: 1.0)
    monkeypatch.setattr(
        WP2_pipeline,
        "generate",
        lambda texts, **kwargs: (
            observed.update(texts=texts, s=kwargs["s"], delta=kwargs["delta"]) or "Shared lesson.",
            object(),
        ),
    )

    WP2_pipeline._run_round2_for_label(
        label="search",
        entries=entries,
        expected_batch_size=2,
        target_epsilon=5.0,
        delta=1e-5,
    )

    assert observed == {
        "texts": [
            "Memory 1: Title 0 | Description 0. | Content 0.\n"
            "Memory 2: Title 1 | Description 1. | Content 1.\n"
            "Memory 3: Title 2 | Description 2. | Content 2."
        ],
        "s": 2,
        "delta": 1e-5,
    }


def test_pipeline_audit_counts_entries_not_sibling_items(
    tmp_path,
    monkeypatch,
) -> None:
    entry = _entry(
        "user_00",
        "Task",
        "2026-08-10T10:00:00+00:00",
        n_items=3,
    )
    (tmp_path / "user_00.jsonl").write_text(
        json.dumps(entry.to_jsonl_dict()) + "\n",
        encoding="utf-8",
    )
    generate_calls: list[dict[str, object]] = []

    class _Embedder:
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    def fake_generate(texts: list[str], **kwargs):
        generate_calls.append(
            {
                "texts": texts,
                "s": kwargs["s"],
                "delta": kwargs["delta"],
            }
        )
        account = SimpleNamespace(
            private_tokens_used=1,
            public_tokens_used=0,
            r=1,
        )
        if kwargs["wrap_fn"] is WP2_pipeline.wrap_label:
            return '["search"]', account
        return "Shared lesson.", account

    monkeypatch.setattr(WP2_pipeline, "Embedder", _Embedder)
    monkeypatch.setattr(WP2_pipeline, "OllamaClient", lambda **kwargs: object())
    monkeypatch.setattr(WP2_pipeline, "solve_r", lambda *args, **kwargs: 1)
    monkeypatch.setattr(WP2_pipeline, "rho_for", lambda *args: 0.1)
    monkeypatch.setattr(WP2_pipeline, "epsilon_from_rho", lambda *args: 1.0)
    monkeypatch.setattr(WP2_pipeline, "generate", fake_generate)
    monkeypatch.setattr(
        WP2_pipeline,
        "title_and_description",
        lambda **kwargs: ("Shared title", "Shared description."),
    )

    WP2_pipeline.main(
        [
            "--skip-trajectories",
            "--entries-per-cycle",
            "1",
            "--k-labels",
            "1",
            "--x-per-label",
            "1",
            "--delta",
            "1e-5",
            "--memory-dir",
            str(tmp_path),
        ]
    )

    audit = json.loads((tmp_path / ".shared_audit.jsonl").read_text(encoding="utf-8"))
    checkpoint = json.loads((tmp_path / ".shared_state.json").read_text(encoding="utf-8"))

    assert generate_calls == [
        {"texts": ["Task"], "s": 1, "delta": 1e-5},
        {
            "texts": [
                "Memory 1: Title 0 | Description 0. | Content 0.\n"
                "Memory 2: Title 1 | Description 1. | Content 1.\n"
                "Memory 3: Title 2 | Description 2. | Content 2."
            ],
            "s": 1,
            "delta": 1e-5,
        },
    ]
    assert audit["s1"] == 1
    assert audit["s3"] == 1
    assert audit["round1"]["actual_batch_size"] == 1
    assert audit["round2_per_label"][0]["actual_batch_size"] == 1
    assert audit["round2_per_label"][0]["entry_ids"] == [entry_id(entry)]
    assert audit["trigger_delta"] == pytest.approx(2e-5)
    assert checkpoint["cumulative_delta"] == pytest.approx(2e-5)
    assert checkpoint["schema_version"] == 3
