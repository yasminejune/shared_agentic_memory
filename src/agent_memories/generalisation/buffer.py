"""Carry-over buffer for entries whose labels have not yet reached the gate.

After Step 2 assigns entries to labels, a label proceeds to Step 3
once enough entries sit in it. Smaller buckets are written here and
rejoin assignment on the next run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent_memories.memory import MemoryEntry

BUFFER_SCHEMA_VERSION = 3


def entry_id(entry: MemoryEntry) -> str:
    """Return the stable identifier used by checkpoints and audit records."""
    return f"{entry.user_id}::{entry.query}::{entry.created_at}"


@dataclass(frozen=True)
class GatingResult:
    """Output of select_round2_inputs for one trigger.

    label_inputs[k] is the list of batch indices assigned to qualifying
    label k, or None when the label did not hit the threshold. carry_over
    holds every entry from non-qualifying labels, in label-then-position
    order.
    """

    label_inputs: list[list[int] | None]
    carry_over: list[int]

    @property
    def triggered_labels(self) -> list[int]:
        """Indices of labels that hit the threshold this trigger."""
        return [k for k, inputs in enumerate(self.label_inputs) if inputs is not None]


def select_round2_inputs(
    buckets: list[list[int]],
    x_per_label: int,
) -> GatingResult:
    """Apply the size gate to every per-label bucket.

    buckets[k] is the list of batch indices assigned to label k.
    """
    if x_per_label < 1:  # x_per_label has to be positive
        raise ValueError(f"x_per_label must be >= 1; got {x_per_label}.")

    label_inputs: list[list[int] | None] = []
    carry_over: list[int] = []
    for bucket in buckets:
        if len(bucket) >= x_per_label:
            label_inputs.append(list(bucket))
        else:
            label_inputs.append(None)
            carry_over.extend(bucket)
    return GatingResult(label_inputs=label_inputs, carry_over=carry_over)


def write_buffer(entries: list[MemoryEntry], path: Path) -> None:
    """Persist the carry-over buffer as a JSONL.

    Each line is a versioned MemoryEntry record. Overwriting is
    intentional since the buffer is what remains for the next run, so that
    run should not see stale carry-over from an earlier invocation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            payload = {"schema_version": BUFFER_SCHEMA_VERSION, **entry.to_jsonl_dict()}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_buffer(path: Path) -> list[MemoryEntry]:
    """Load the carry-over buffer as entry-level protected examples.

    Returns an empty list when the file does not exist so the caller
    can always concatenate carry-over with the new-memory list.
    """
    if not path.exists():
        return []
    entries: list[MemoryEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("schema_version") != BUFFER_SCHEMA_VERSION:
                raise ValueError(
                    "Carry-over buffer uses the legacy item-level schema. "
                    "Reset the WP2 shared state before running entry-level privacy."
                )
            entries.append(MemoryEntry.from_jsonl_dict(data))
    return entries
