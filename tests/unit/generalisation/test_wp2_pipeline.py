"""Unit tests for WP2 cycle assembly and checkpoint migration."""

from __future__ import annotations

import json

from scripts.memories import WP2_pipeline
from scripts.memories.WP2_pipeline import (
    _assemble_cycle_entries,
    _entry_id,
    _load_checkpoint,
)

from agent_memories.generalisation import select_round2_inputs
from agent_memories.memory import MemoryEntry, MemoryItem


class _UnusedEmbedder:
    """Embedder double for records that already contain embeddings."""

    def embed(self, text: str) -> list[float]:
        raise AssertionError(f"Unexpected embedding request for {text!r}")


def _entry(user_id: str, query: str, created_at: str) -> MemoryEntry:
    return MemoryEntry(
        user_id=user_id,
        query=query,
        outcome="successful",
        items=[
            MemoryItem(
                title="Test title",
                description="Test description.",
                content="Test content.",
            )
        ],
        embedding=[1.0, 0.0],
        created_at=created_at,
    )


def test_cycle_assembly_excludes_carry_over_from_round1_without_dropping_it(tmp_path) -> None:
    held = _entry("user_00", "Held task", "2026-08-09T10:00:00+00:00")
    new = _entry("user_00", "New task", "2026-08-10T10:00:00+00:00")
    user_store = tmp_path / "user_00.jsonl"
    user_store.write_text(
        "\n".join(json.dumps(entry.to_jsonl_dict()) for entry in (held, new)) + "\n",
        encoding="utf-8",
    )

    new_entries, assignment_batch, processed_ids = _assemble_cycle_entries(
        tmp_path,
        set(),
        [held],
        _UnusedEmbedder(),
    )

    assert new_entries == [new]
    assert assignment_batch == [new, held]
    assert assignment_batch.count(held) == 1
    assert _entry_id(held) in processed_ids
    assert _entry_id(new) not in processed_ids


def test_checkpoint_migration_treats_consumed_entries_as_step1_processed(tmp_path) -> None:
    checkpoint_path = tmp_path / ".shared_state.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "last_run_at": "2026-08-09T10:00:00+00:00",
                "consumed_entry_ids": ["entry-1"],
                "trigger_count": 1,
                "cumulative_epsilon": 2.0,
            }
        ),
        encoding="utf-8",
    )

    checkpoint = _load_checkpoint(checkpoint_path)

    assert checkpoint["step1_processed_entry_ids"] == ["entry-1"]


def test_gating_passes_every_entry_from_a_qualifying_label() -> None:
    result = select_round2_inputs(
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
        x_per_label=5,
    )

    assert result.label_inputs == [[0, 1, 2, 3, 4, 5, 6], None]
    assert result.carry_over == [7, 8, 9, 10]


def test_round2_uses_threshold_as_expected_batch_size(monkeypatch) -> None:
    entries = [
        _entry(f"user_{i:02d}", f"Task {i}", f"2026-08-10T10:{i:02d}:00+00:00")
        for i in range(7)
    ]
    observed: dict[str, int] = {}

    def fake_solve_r(
        target_epsilon: float,
        delta: float,
        *,
        s: int,
        c: float,
        tau: float,
        sigma: float,
        r_max: int,
    ) -> int:
        observed["accounting_s"] = s
        return 1

    def fake_generate(texts: list[str], *, s: int, **kwargs):
        observed["actual_batch_size"] = len(texts)
        observed["generation_s"] = s
        return "Shared lesson.", object()

    monkeypatch.setattr(WP2_pipeline, "solve_r", fake_solve_r)
    monkeypatch.setattr(WP2_pipeline, "rho_for", lambda *args: 0.1)
    monkeypatch.setattr(WP2_pipeline, "epsilon_from_rho", lambda *args: 1.0)
    monkeypatch.setattr(WP2_pipeline, "generate", fake_generate)

    result = WP2_pipeline._run_round2_for_label(
        label="Product search",
        entries=entries,
        expected_batch_size=5,
        target_epsilon=5.0,
    )

    assert result is not None
    assert observed == {
        "accounting_s": 5,
        "actual_batch_size": 7,
        "generation_s": 5,
    }
