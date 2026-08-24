"""Private and shared memory records, and cosine retrieval over them.

A record carries its provenance alongside the entry: the audit key is a
source task id for private memories and the DP label for shared ones,
and it is what the run CSV records as retrieved_task_ids.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent_memories.memory import Embedder, MemoryEntry

# source ("private" or "shared"), audit key (task id or DP label), entry
MemoryRecord = tuple[str, "int | str", "MemoryEntry"]


def load_memory_entries_from_csv(csv_path: Path) -> list[tuple[int, MemoryEntry]]:
    """Load (task_id, MemoryEntry) pairs from the memories CSV.

    Skips rows with no extracted memory or no stored embedding.
    """
    from agent_memories.memory import MemoryEntry

    if not csv_path.exists():
        print(f"Memories file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[int, MemoryEntry]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task_id_raw = row.get("task_id", "").strip()
            if not task_id_raw.isdigit():
                continue
            if row.get("memory_extracted", "").strip() != "True":
                continue
            memory_data = json.loads(row.get("memory", "{}") or "{}")
            embedding = json.loads(row.get("embedding", "[]") or "[]")
            if not memory_data or not embedding:
                continue
            entry = MemoryEntry.from_jsonl_dict(memory_data)
            entry.embedding = [float(x) for x in embedding]
            pairs.append((int(task_id_raw), entry))
    return pairs


def load_private_records(csv_path: Path) -> list[MemoryRecord]:
    """Private memories from the ReasoningBank CSV, keyed by source task_id."""
    return [
        ("private", task_id, entry) for task_id, entry in load_memory_entries_from_csv(csv_path)
    ]


def load_shared_records(store_path: Path, embedder: Embedder) -> list[MemoryRecord]:
    """Shared memories from the JSONL store, keyed by DP label."""
    from agent_memories.memory import MemoryStore

    store = MemoryStore.load(store_path, user_id="shared", embedder=embedder)
    return [("shared", entry.query, entry) for entry in store.all() if entry.embedding is not None]


class MemoryIndex:
    """Cosine top-k over private and/or shared memory records.

    Every condition retrieves the same way; only the record list differs.
    MemoryStore.search runs the same _cosine_top_k call, so shared entries
    are indexed here rather than searched through the store.
    """

    def __init__(self, records: list[MemoryRecord], embedder: Embedder) -> None:
        self._records = records
        self._embedder = embedder
        self._matrix = np.asarray(
            [entry.embedding for _, _, entry in records],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self._records)

    def search(self, intent: str, *, k: int) -> list[MemoryRecord]:
        """Return the top-k records by cosine similarity to intent."""
        if not self._records or k <= 0:
            return []
        from agent_memories.memory.store import _cosine_top_k

        query_vec = np.asarray(self._embedder.embed(intent), dtype=np.float32)
        indices = _cosine_top_k(query_vec, self._matrix, k)
        return [self._records[i] for i in indices]


def audit_ids(records: list[MemoryRecord]) -> list[int | str]:
    """Audit keys in rank order: private task ids, shared DP labels."""
    return [key for _, key, _ in records]


def flatten_records_for_think(records: list[MemoryRecord]) -> list[dict[str, str]]:
    """Flatten records to the {title, content} list Think expects."""
    flat: list[dict[str, str]] = []
    for _, _, entry in records:
        for item in entry.items:
            flat.append({"title": item.title, "content": item.content})
    return flat
