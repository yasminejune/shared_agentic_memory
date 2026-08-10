"""WP2 X-gating + carry-over ("set aside") buffer for round-2 inputs.

After :func:`agent_memories.generalisation.assign_memories_to_labels`
partitions every round-1 input into one of ``K`` per-label buckets,
this module decides which buckets are full enough to spend round-2
privacy budget on and what to do with the leftover memory items. The
caller's policy (from the user's design notes) is:

* For each label, if the bucket size is at least ``X_PER_LABEL``,
  pass the whole bucket to round-2 generation.
* For each label whose bucket holds fewer than ``X`` items, push
  the whole bucket to the carry-over buffer ("any memory that was
  added to a label with insufficient memories will be set aside").

The carry-over set is persisted as JSONL alongside the per-user
stores. In the next trigger, these items skip round 1 because
they have already paid that privacy cost. They rejoin at round-2
batch assignment and are assigned to whichever newly generated
label has the highest cosine similarity.

The X-gating itself is content-dependent ("did this label hit ``X``
items this trigger?") and so leaks information about the per-label
distribution. WP2-plan §10 open question 2 names this. The user
opted into ``X > 1`` knowingly; documenting the leak in the
methodology chapter is the agreed mitigation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .cycle import CycleItem


@dataclass(frozen=True)
class GatingResult:
    """Output of :func:`select_round2_inputs` for one trigger.

    ``label_inputs[k]`` is either the complete list of item-batch indices
    values assigned to qualifying label ``k`` or ``None`` when the
    label did not hit the ``X`` threshold this trigger.
    ``carry_over`` contains every item from non-qualifying labels, in
    label-then-position order so repeated invocations on the same
    input give a deterministic persistence file.
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

    ``buckets[k]`` is the list of item-batch indices assigned to
    label ``k`` by :func:`group_by_label`. ``x_per_label`` must be a
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


def write_buffer(items: list[CycleItem], path: Path) -> None:
    """Persist the carry-over buffer as JSONL at ``path`` (overwrite).

    Each line is a versioned :class:`CycleItem` record. The item-level
    schema is deliberately incompatible with the old entry-level
    buffer because one trajectory may be split across cycles.

    Overwriting (rather than appending) is the right semantic here:
    the buffer always reflects "what is left to carry into the next
    trigger after the orchestrator finished this trigger", so the
    next trigger should never see stale carry-over from a previous
    pipeline invocation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_jsonl_dict(), ensure_ascii=False) + "\n")


def read_buffer(path: Path) -> list[CycleItem]:
    """Load the carry-over buffer as item-level protected examples.

    Returns an empty list when the file does not exist (the cold-start
    case before the first WP2 trigger has ever written one) so the
    orchestrator can call this unconditionally and ``+`` the result
    onto the new-memory list without a presence check.
    """
    if not path.exists():
        return []
    items: list[CycleItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(CycleItem.from_jsonl_dict(json.loads(line)))
    return items
