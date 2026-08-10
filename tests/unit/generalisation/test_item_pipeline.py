"""Unit tests for item-level WP2 batching and privacy accounting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from scripts.memories import WP2_pipeline

from agent_memories.generalisation import (
    assign_memories_to_labels,
    flatten_entry,
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
        embedding=embedding or [1.0, 0.0],
        created_at=created_at,
    )


def test_flatten_entry_makes_each_memory_item_a_separate_example() -> None:
    entry = _entry(
        "user_00",
        "Find a product",
        "2026-08-10T10:00:00+00:00",
        n_items=3,
    )

    items = flatten_entry(entry)

    assert len(items) == 3
    assert [item.item_index for item in items] == [0, 1, 2]
    assert len({item.item_id for item in items}) == 3
    assert [item.item.title for item in items] == ["Title 0", "Title 1", "Title 2"]
    assert all(item.query == entry.query for item in items)


def test_cycle_assembly_takes_exactly_b_items_and_splits_parent_entries(tmp_path) -> None:
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
    for entry in (first, second):
        path = tmp_path / f"{entry.user_id}.jsonl"
        path.write_text(json.dumps(entry.to_jsonl_dict()) + "\n", encoding="utf-8")

    new_items, assignment_batch, processed_ids = WP2_pipeline._assemble_cycle_items(
        tmp_path,
        set(),
        [],
        _UnusedEmbedder(),
        items_per_cycle=2,
    )

    assert [item.item_index for item in new_items] == [0, 1]
    assert assignment_batch == new_items
    assert processed_ids == set()

    processed_ids.update(item.item_id for item in new_items)
    next_items, _, _ = WP2_pipeline._assemble_cycle_items(
        tmp_path,
        processed_ids,
        [],
        _UnusedEmbedder(),
        items_per_cycle=2,
    )

    assert [(item.user_id, item.item_index) for item in next_items] == [
        ("user_00", 2),
        ("user_01", 0),
    ]


def test_cycle_waits_when_fewer_than_b_new_items_are_available(tmp_path) -> None:
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

    new_items, assignment_batch, processed_ids = WP2_pipeline._assemble_cycle_items(
        tmp_path,
        set(),
        [],
        _UnusedEmbedder(),
        items_per_cycle=3,
    )

    assert new_items == []
    assert assignment_batch == []
    assert processed_ids == set()


def test_item_buffer_round_trips_and_rejects_legacy_entry_records(tmp_path) -> None:
    item = flatten_entry(_entry("user_00", "Task", "2026-08-10T10:00:00+00:00"))[0]
    path = tmp_path / ".shared_buffer.jsonl"

    write_buffer([item], path)

    assert read_buffer(path) == [item]

    path.write_text(
        json.dumps(_entry("user_00", "Legacy", "2026-08-09T10:00:00+00:00").to_jsonl_dict()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy entry-level"):
        read_buffer(path)


def test_checkpoint_rejects_legacy_entry_level_state(tmp_path) -> None:
    path = tmp_path / ".shared_state.json"
    path.write_text(
        json.dumps({"step1_processed_entry_ids": ["legacy-entry"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="legacy entry-level"):
        WP2_pipeline._load_checkpoint(path)


def test_assignment_treats_sibling_items_as_separate_rows() -> None:
    items = flatten_entry(
        _entry(
            "user_00",
            "Task",
            "2026-08-10T10:00:00+00:00",
            n_items=3,
        )
    )

    class _Embedder:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["Label"]
            return [[1.0, 0.0]]

    assignments = assign_memories_to_labels(items, ["Label"], embedder=_Embedder())

    assert len(assignments) == 3
    assert [assignment.item_index for assignment in assignments] == [0, 1, 2]


def test_round1_uses_b_and_one_prompt_row_per_item(monkeypatch) -> None:
    items = flatten_entry(
        _entry(
            "user_00",
            "Repeated parent query",
            "2026-08-10T10:00:00+00:00",
            n_items=3,
        )
    )
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
        items,
        expected_batch_size=3,
        k_labels=1,
        target_epsilon=5.0,
        delta=1e-5,
    )

    assert observed == {
        "texts": ["Repeated parent query"] * 3,
        "s": 3,
        "delta": 1e-5,
    }


def test_round2_uses_x_and_one_prompt_row_per_item(monkeypatch) -> None:
    items = flatten_entry(
        _entry(
            "user_00",
            "Task",
            "2026-08-10T10:00:00+00:00",
            n_items=3,
        )
    )
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
        items=items,
        expected_batch_size=2,
        target_epsilon=5.0,
        delta=1e-5,
    )

    assert observed == {
        "texts": [
            "Title 0 | Description 0. | Content 0.",
            "Title 1 | Description 1. | Content 1.",
            "Title 2 | Description 2. | Content 2.",
        ],
        "s": 2,
        "delta": 1e-5,
    }


def test_pipeline_audit_keeps_expected_and_actual_item_counts_separate(
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
            "--items-per-cycle",
            "3",
            "--k-labels",
            "1",
            "--x-per-label",
            "2",
            "--delta",
            "1e-5",
            "--memory-dir",
            str(tmp_path),
        ]
    )

    audit = json.loads((tmp_path / ".shared_audit.jsonl").read_text(encoding="utf-8"))
    checkpoint = json.loads((tmp_path / ".shared_state.json").read_text(encoding="utf-8"))

    assert generate_calls == [
        {"texts": ["Task", "Task", "Task"], "s": 3, "delta": 1e-5},
        {
            "texts": [
                "Title 0 | Description 0. | Content 0.",
                "Title 1 | Description 1. | Content 1.",
                "Title 2 | Description 2. | Content 2.",
            ],
            "s": 2,
            "delta": 1e-5,
        },
    ]
    assert audit["s1"] == 3
    assert audit["s3"] == 2
    assert audit["round1"]["actual_batch_size"] == 3
    assert audit["round2_per_label"][0]["actual_batch_size"] == 3
    assert len(audit["round2_per_label"][0]["item_ids"]) == 3
    assert audit["trigger_delta"] == pytest.approx(2e-5)
    assert checkpoint["cumulative_delta"] == pytest.approx(2e-5)
