"""Step 2: cosine assignment of tasks to DP labels, gated at B=7.

Writes ``assignments.json`` (qualifying buckets) and the carry-over
buffer (labels below the gate). Independently runnable against
``labels.json`` and the memories CSV.
"""

from __future__ import annotations

import argparse
import sys
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
    add_io_arguments,
    assignments_path,
    buffer_path,
    labels_path,
    load_memory_entries_from_csv,
    read_json,
    resolve_io_paths,
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
    """Assign tasks to labels and gate at ``x_per_label``.

    Returns the assignments artefact (qualifying buckets carry full
    entry records for Step 3). Carry-over entry indices are in
    ``carry_over_indices``; the caller persists those entries.
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_io_arguments(parser)
    parser.add_argument("--x-per-label", type=int, default=BUCKET_SIZE)
    args = parser.parse_args(argv)

    csv_path, work_dir = resolve_io_paths(args)
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
        x_per_label=args.x_per_label,
        carry_over=carry,
    )
    carry_entries: list[MemoryEntry] = artefact.pop("_carry_over_entries")
    write_buffer(carry_entries, buffer_path(work_dir))
    write_json(assignments_path(work_dir), artefact)

    print(f"[Step 2] bucket sizes: {artefact['bucket_sizes']}")
    print(
        f"[Step 2] triggered labels (bucket >= {args.x_per_label}): "
        f"{artefact['triggered_labels']}"
    )
    print(f"[Step 2] carry-over: {len(carry_entries)} entries -> {buffer_path(work_dir)}")
    print(f"[Step 2] Wrote {assignments_path(work_dir)}")


if __name__ == "__main__":
    main(sys.argv[1:])
