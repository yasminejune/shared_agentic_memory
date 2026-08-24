"""Per-user JSONL memory store.

One line per trajectory. The embedding is the task query. Search is
brute-force cosine over those vectors.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .embedder import Embedder

Outcome = Literal["successful", "failed", "shared"]


@dataclass
class MemoryItem:
    """One ReasoningBank item: title, description, and content.

    The extractor fills all three. An item missing any field is dropped
    before it reaches the store.
    """

    title: str
    description: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        return cls(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            content=str(data.get("content", "")),
        )


@dataclass
class MemoryEntry:
    """One trajectory's memories: task, outcome, and up to three items.

    The embedding is over tasks, aka queries. All items share one outcome from the
    judge.
    """

    user_id: str
    query: str
    outcome: Outcome
    items: list[MemoryItem] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: str = ""

    def to_jsonl_dict(self) -> dict[str, Any]:
        """JSON dict for one JSONL line."""
        return {
            "user_id": self.user_id,
            "query": self.query,
            "outcome": self.outcome,
            "items": [item.to_dict() for item in self.items],
            "embedding": self.embedding,
            "created_at": self.created_at,
        }

    def to_dict_without_embedding(self) -> dict[str, Any]:
        """JSON dict without the embedding vector."""
        data = self.to_jsonl_dict()
        del data["embedding"]
        return data

    @classmethod
    def from_jsonl_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Rebuild an entry from one parsed JSONL line."""
        return cls(
            user_id=data["user_id"],
            query=str(data.get("query", "")),
            outcome=_coerce_outcome(data.get("outcome", "failed")),
            items=[MemoryItem.from_dict(it) for it in data.get("items", [])],
            embedding=data.get("embedding"),
            created_at=str(data.get("created_at", "")),
        )


def _coerce_outcome(raw: Any) -> Outcome:
    """Map raw to successful, failed, or shared.

    Unrecognised values become failed. shared is kept so the
    cross-user store round-trips through load.
    """
    if raw == "successful":  # I.e. successful private memories
        return "successful"
    if raw == "shared":  # I.e. shared memories (non-private)
        return "shared"
    return "failed"  # I.e. failed private memories


def _utcnow_iso() -> str:
    """UTC now as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Indices of the k rows in matrix closest to query by cosine.

    Zero-norm rows are clamped so the division does not produce NaNs.
    """
    if matrix.shape[0] == 0 or k <= 0:
        return []
    query_norm = float(np.linalg.norm(query)) or 1e-12
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0, 1e-12, row_norms)
    sims = (matrix @ query) / (row_norms * query_norm)
    take = min(k, sims.shape[0])
    # argpartition gives the unsorted top-take; sort that small slice by score.
    top_unsorted = np.argpartition(-sims, take - 1)[:take]
    top_sorted = top_unsorted[np.argsort(-sims[top_unsorted])]
    return [int(i) for i in top_sorted]


class MemoryStore:
    """JSONL store with in-memory cosine search.

    One JSON object per line. The same class on a different path is the
    shared store; those embeddings are over the label rather than the task.
    """

    def __init__(self, path: Path, user_id: str, embedder: Embedder) -> None:
        self.path = Path(path)
        self.user_id = user_id
        self.embedder = embedder
        self._entries: list[MemoryEntry] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def load(cls, path: Path, user_id: str, embedder: Embedder) -> MemoryStore:
        """Build a store and load path if the file exists.

        A missing file is fine: the user has no memories yet, and the
        first add_entry creates it.
        """
        store = cls(path=path, user_id=user_id, embedder=embedder)
        if store.path.exists():
            store._load_from_disk()
        return store

    def _load_from_disk(self) -> None:
        entries: list[MemoryEntry] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entries.append(MemoryEntry.from_jsonl_dict(json.loads(line)))
        self._entries = entries
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        embeddings = [e.embedding for e in self._entries if e.embedding is not None]
        if not embeddings:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        self._matrix = np.asarray(embeddings, dtype=np.float32)

    def add_entry(
        self,
        *,
        query: str,
        outcome: Outcome,
        items: Iterable[MemoryItem],
    ) -> MemoryEntry:
        """Embed query, then append the entry on disk and in memory.

        For private memories query is the task; for shared memories it
        is the label. Reload reads the stored vector and does not re-embed.
        """
        item_list = list(items)
        embedding = self.embedder.embed(query)
        entry = MemoryEntry(
            user_id=self.user_id,
            query=query,
            outcome=outcome,
            items=item_list,
            embedding=embedding,
            created_at=_utcnow_iso(),
        )
        self._append_to_disk(entry)
        self._entries.append(entry)
        self._rebuild_matrix()
        return entry

    def _append_to_disk(self, entry: MemoryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_jsonl_dict(), ensure_ascii=False) + "\n")

    def search(self, query: str, *, k: int) -> list[MemoryEntry]:
        """Return the k stored entries closest to query by cosine.

        Empty store or k <= 0 returns an empty list. Results are
        sorted by descending similarity.
        """
        if not self._entries or k <= 0:
            return []
        query_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        indices = _cosine_top_k(query_vec, self._matrix, k)
        return [self._entries[i] for i in indices]

    def all(self) -> list[MemoryEntry]:
        """Every stored entry in insertion order (a copy)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
