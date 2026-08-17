"""WP2 X-gating + carry-over ("set aside") buffer for round-2 inputs.

After :func:`agent_memories.generalisation.assign_memories_to_labels`
partitions every round-1 input into one of ``K`` per-label buckets,
this module decides which buckets are full enough to spend round-2
privacy budget on and what to do with the leftover trajectory entries.
The caller's policy (from the user's design notes) is:

* For each label, if the bucket size is at least ``X_PER_LABEL``,
  pass the whole bucket to round-2 generation.
* For each label whose bucket holds fewer than ``X`` entries, push
  the whole bucket to the carry-over buffer ("any memory that was
  added to a label with insufficient memories will be set aside").

The carry-over set is persisted as JSONL alongside the per-user
stores. In the next trigger, these entries skip round 1 because
they have already paid that privacy cost. They rejoin at round-2
batch assignment and are assigned to whichever newly generated
label has the highest cosine similarity.

The X-gating itself is content-dependent ("did this label hit ``X``
entries this trigger?") and so leaks information about the per-label
distribution. WP2-plan §10 open question 2 names this. The user
opted into ``X > 1`` knowingly; documenting the leak in the
methodology chapter is the agreed mitigation.
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
    """Output of :func:`select_round2_inputs` for one trigger.

    ``label_inputs[k]`` is either the complete list of batch indices
    assigned to qualifying label ``k`` or ``None`` when the label did
    not hit the ``X`` threshold this trigger. ``carry_over`` contains
    every entry from non-qualifying labels, in label-then-position
    order so repeated invocations on the same input give a
    deterministic persistence file.
    """

    label_inputs: list[list[int] | None]
    carry_over: list[int]

    @property
    def triggered_labels(self) -> list[int]:
        """Indices of labels that hit the ``X`` threshold this trigger."""
        return [k for k, inputs in enumerate(self.label_inputs) if inputs is not None]


def select_round2_inputs(
    buckets: list[list[int]],
    x_per_label: int,
) -> GatingResult:
    """Apply the X-gating rule to every per-label bucket.

    ``buckets[k]`` is the list of batch indices assigned to label
    ``k`` by :func:`group_by_label`. ``x_per_label`` must be a
    positive integer; passing ``0`` would make every label trigger
    on every empty bucket and is rejected explicitly to surface the
    config mistake rather than silently generate noise from empty
    batches.
    """
    if x_per_label < 1:
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
    """Persist the carry-over buffer as JSONL at ``path`` (overwrite).

    Each line is a versioned :class:`MemoryEntry` record. Schema v3 is
    deliberately incompatible with the v2 item-level buffer because
    the protected example is now the whole trajectory entry.

    Overwriting (rather than appending) is the right semantic here:
    the buffer always reflects "what is left to carry into the next
    trigger after the orchestrator finished this trigger", so the
    next trigger should never see stale carry-over from a previous
    pipeline invocation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            payload = {"schema_version": BUFFER_SCHEMA_VERSION, **entry.to_jsonl_dict()}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_buffer(path: Path) -> list[MemoryEntry]:
    """Load the carry-over buffer as entry-level protected examples.

    Returns an empty list when the file does not exist (the cold-start
    case before the first WP2 trigger has ever written one) so the
    orchestrator can call this unconditionally and ``+`` the result
    onto the new-memory list without a presence check.
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
