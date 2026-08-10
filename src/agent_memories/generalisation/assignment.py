"""WP2 round-2 batch assignment: per-item nearest-label by cosine.

After WP2-plan §3.2 round 1 emits a list of DP-released label strings,
the orchestrator embeds each label and assigns every round-2-candidate
memory item to its nearest label. The DP guarantee for round-2 batch
assignment rests on Assumption 1 — the assignment must depend only on
the prompt itself plus public information, never on other memories in
the batch (WP2-plan §4.3 / WP2.5 resolution). Cosine similarity
between a memory's own ``query`` embedding and the public label
embeddings satisfies that constraint by construction: the labels are
post-processed DP artifacts (public) and each memory's embedding is a
property of that memory alone.

The :class:`~agent_memories.memory.MemoryStore` already embeds every
parent entry's ``query`` field via the same
:class:`~agent_memories.memory.Embedder` (WP1.6 schema, see
``MemoryStore.add_entry``), so this module reuses the stored
``entry.embedding`` directly and only spends model time on the ``K``
label embeddings, not the ``N`` item embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_memories.memory import Embedder

from .cycle import CycleItem


@dataclass(frozen=True)
class LabelAssignment:
    """Result of :func:`assign_memories_to_labels` for one memory item.

    ``item_index`` is the position of the item in the input list
    (so the caller can rebuild a parallel structure without relying
    on object identity); ``label_index`` is the chosen ``argmax``
    over the ``K`` labels; ``similarity`` is the matching cosine
    similarity in ``[-1, 1]`` for downstream diagnostics or
    threshold-based filtering. The label string itself is not
    embedded in this record because the orchestrator already holds
    the ``labels`` list and can look it up by index.
    """

    item_index: int
    label_index: int
    similarity: float


def assign_memories_to_labels(
    memories: list[CycleItem],
    labels: list[str],
    *,
    embedder: Embedder,
) -> list[LabelAssignment]:
    """Assign each item-level protected example to its nearest label.

    The label strings are embedded with ``embedder.embed_batch`` (one
    forward pass over all ``K`` labels). Each memory's stored
    ``item.embedding`` is used as-is — it was computed at write time
    over the parent trajectory's ``query`` field via the same embedder model
    (``all-MiniLM-L6-v2`` by default), so the cosine similarities are
    comparable without re-embedding. Memories whose ``embedding`` is
    ``None`` raise ``ValueError`` because the WP1.6 store only writes
    Returns one :class:`LabelAssignment` per input memory in input
    order. The caller can group by ``label_index`` to build per-label
    buckets for the X-gating step.
    """
    if not labels:
        raise ValueError("`labels` must contain at least one label.")
    if not memories:
        return []

    label_matrix = np.asarray(embedder.embed_batch(labels), dtype=np.float32)
    label_norms = np.linalg.norm(label_matrix, axis=1)
    label_norms = np.where(label_norms == 0, 1e-12, label_norms)
    label_unit = label_matrix / label_norms[:, None]

    assignments: list[LabelAssignment] = []
    for idx, item in enumerate(memories):
        memory_vec = np.asarray(item.embedding, dtype=np.float32)
        memory_norm = float(np.linalg.norm(memory_vec)) or 1e-12
        sims = (label_unit @ memory_vec) / memory_norm
        best = int(np.argmax(sims))
        assignments.append(
            LabelAssignment(
                item_index=idx,
                label_index=best,
                similarity=float(sims[best]),
            )
        )
    return assignments


def group_by_label(
    assignments: list[LabelAssignment],
    n_labels: int,
) -> list[list[int]]:
    """Bucket ``LabelAssignment`` indices by their ``label_index``.

    Returns ``n_labels`` lists, each holding the ``item_index`` of
    every item assigned to that label, in the order the items
    appeared in the original input. Empty buckets are returned for
    labels that received no memories so the caller can iterate over
    a fixed ``range(n_labels)`` rather than handling missing keys.
    The fixed ``range(n_labels)`` shape is also what the X-gating
    step expects ("for each label, is the bucket size ≥ X?").
    """
    buckets: list[list[int]] = [[] for _ in range(n_labels)]
    for assignment in assignments:
        if 0 <= assignment.label_index < n_labels:
            buckets[assignment.label_index].append(assignment.item_index)
    return buckets
