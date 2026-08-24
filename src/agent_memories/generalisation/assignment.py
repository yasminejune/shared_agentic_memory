"""Nearest label by cosine of label vs query embeddings.

Each private entry is assigned to one label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_memories.memory import Embedder, MemoryEntry


@dataclass(frozen=True)
class LabelAssignment:
    """Nearest-label result for one memory entry.

    item_index is the position in the input list; label_index is the
    argmax over the labels; similarity is the matching cosine in [-1, 1].
    """

    item_index: int
    label_index: int
    similarity: float


def assign_memories_to_labels(
    memories: list[MemoryEntry],
    labels: list[str],
    *,
    embedder: Embedder,
) -> list[LabelAssignment]:
    """Assign each entry to its nearest label by cosine similarity.

    Labels are embedded in one batch. Each memory uses its stored query
    embedding. Entries with no embedding raise a ValueError.
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
    for idx, entry in enumerate(memories):
        if entry.embedding is None:
            raise ValueError(
                f"Memory entry for user_id={entry.user_id!r}, query={entry.query!r} "
                "has no query embedding."
            )
        memory_vec = np.asarray(entry.embedding, dtype=np.float32)
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
    """Bucket assignment indices by label_index.

    Returns n_labels lists of item_index values, including empty buckets
    so the caller can ask whether each label has enough entries for Step 3.
    """
    buckets: list[list[int]] = [[] for _ in range(n_labels)]
    for assignment in assignments:
        if 0 <= assignment.label_index < n_labels:
            buckets[assignment.label_index].append(assignment.item_index)
    return buckets
