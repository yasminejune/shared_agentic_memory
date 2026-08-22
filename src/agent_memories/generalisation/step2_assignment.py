"""Step 2: assign each private entry to one DP label by cosine similarity.

Assignment is cosine of embeddings (label vs query). Per Amin Assumption 1
it depends only on the entry itself. A label proceeds to Step 3 once
enough entries sit in it. Writes assignments.json and the carry-over
buffer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_memories.generalisation.assignment import (
    assign_memories_to_labels,
    group_by_label,
)
from agent_memories.generalisation.buffer import (
    entry_id,
    read_buffer,
    select_round2_inputs,
    write_buffer,
)
from agent_memories.generalisation.io import (
    BUCKET_SIZE,
    MEMORIES_CSV,
    WORK_DIR,
    assignments_path,
    buffer_path,
    labels_path,
    load_memory_entries_from_csv,
    read_json,
    write_json,
)
from agent_memories.memory import Embedder, MemoryEntry


def _dedupe_entries(*groups: Sequence[MemoryEntry]) -> list[MemoryEntry]:
    seen: set[str] = set()
    out: list[MemoryEntry] = []
    for group in groups:
        for entry in group:
            key = entry_id(entry)
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
    return out


def run_assignment(
    entries: Sequence[MemoryEntry],
    labels: Sequence[str],
    *,
    embedder: Embedder,
    x_per_label: int = BUCKET_SIZE,
    carry_over: Sequence[MemoryEntry] = (),
) -> dict[str, Any]:
    """Assign tasks to labels and gate at x_per_label.

    Returns the assignments artefact. Qualifying buckets carry full entry
    records for Step 3. Carry-over entries are returned for the caller
    to persist.
    """
    if not labels:
        raise ValueError("Step 2 requires at least one label.")
    batch = _dedupe_entries(entries, carry_over)
    if not batch:
        raise ValueError("Step 2 requires at least one task to assign.")

    assignments = assign_memories_to_labels(list(batch), list(labels), embedder=embedder)
    buckets = group_by_label(assignments, n_labels=len(labels))
    gating = select_round2_inputs(buckets, x_per_label=x_per_label)

    qualifying: list[dict[str, Any]] = []
    for label_idx in gating.triggered_labels:
        indices = gating.label_inputs[label_idx]
        assert indices is not None
        qualifying.append(
            {
                "label_index": label_idx,
                "label": labels[label_idx],
                "entries": [batch[i].to_jsonl_dict() for i in indices],
            }
        )

    return {
        "labels": list(labels),
        "x_per_label": x_per_label,
        "triggered_labels": gating.triggered_labels,
        "bucket_sizes": [len(bucket) for bucket in buckets],
        "qualifying": qualifying,
        "assignments": [
            {
                "item_index": row.item_index,
                "label_index": row.label_index,
                "similarity": row.similarity,
                "entry_id": entry_id(batch[row.item_index]),
            }
            for row in assignments
        ],
        "carry_over_entry_ids": [entry_id(batch[i]) for i in gating.carry_over],
        "_carry_over_entries": [batch[i] for i in gating.carry_over],
    }


def main() -> None:
    csv_path, work_dir = MEMORIES_CSV, WORK_DIR
    labels_file = labels_path(work_dir)
    labels_artefact = read_json(labels_file)
    labels = list(labels_artefact["labels"])

    pairs = load_memory_entries_from_csv(csv_path)
    csv_entries = [entry for _, entry in pairs]
    carry = read_buffer(buffer_path(work_dir))
    print(
        f"[Step 2] {len(csv_entries)} CSV tasks, {len(carry)} carry-over, "
        f"{len(labels)} labels from {labels_file}"
    )

    embedder = Embedder()
    artefact = run_assignment(
        csv_entries,
        labels,
        embedder=embedder,
        x_per_label=BUCKET_SIZE,
        carry_over=carry,
    )
    carry_entries: list[MemoryEntry] = artefact.pop("_carry_over_entries")
    write_buffer(carry_entries, buffer_path(work_dir))
    write_json(assignments_path(work_dir), artefact)

    print(f"[Step 2] bucket sizes: {artefact['bucket_sizes']}")
    print(
        f"[Step 2] triggered labels (bucket >= {BUCKET_SIZE}): " f"{artefact['triggered_labels']}"
    )
    print(f"[Step 2] carry-over: {len(carry_entries)} entries -> {buffer_path(work_dir)}")
    print(f"[Step 2] Wrote {assignments_path(work_dir)}")


if __name__ == "__main__":
    main()
