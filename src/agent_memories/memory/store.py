"""Per-user :class:`MemoryStore` and the records it holds.

Storage is intentionally minimal: one JSONL file per user, one record
per line, embedding inlined alongside the structured items. Reload is
therefore O(file size) and never needs to call the embedder again,
which keeps the controlled-task smoke runs deterministic across
machines.

Cosine search runs against an in-memory ``numpy`` matrix that the store
rebuilds whenever an entry is added. WP1.6 operates at a scale of at
most a handful of memories per user; brute-force cosine is far simpler
than introducing FAISS or a vector DB, and the proposal explicitly
punts retrieval optimisation (§3.2.2).

WP1.6 schema (ReasoningBank, Ouyang et al. 2025, Appendix A.2):

* Each disk record is one *trajectory* keyed by its task query.
* The embedding is computed over the **query string**, not over the
  memory items themselves, so retrieval is query-to-query similarity.
* Each record holds 1-3 :class:`MemoryItem`s with the paper's
  ``{title, description, content}`` schema and an ``outcome`` of either
  ``"successful"`` or ``"failed"`` (one outcome per trajectory, shared
  by all items extracted from it).

WP2.4's shared cross-user store is just a second :class:`MemoryStore`
pointed at a different path (e.g. ``data/memories/shared.jsonl``).
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
    """One distilled ReasoningBank memory item.

    Fields match the paper's schema (Appendix A.1): ``title`` is the
    short identifier, ``description`` is a one-sentence summary, and
    ``content`` carries the distilled reasoning steps. All three are
    populated by the LLM extractor; an item missing any of them is
    treated as malformed and dropped before reaching the store.
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
    """One trajectory's worth of ReasoningBank memory.

    ``query`` is the task aim; it is the *only* field used for
    retrieval embedding (Appendix A.2 "ReasoningBank Storage"). All
    items in the entry share one ``outcome`` because the
    LLM-as-Judge produces one signal per trajectory.
    """

    user_id: str
    query: str
    outcome: Outcome
    items: list[MemoryItem] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: str = ""

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for the on-disk JSONL line."""
        return {
            "user_id": self.user_id,
            "query": self.query,
            "outcome": self.outcome,
            "items": [item.to_dict() for item in self.items],
            "embedding": self.embedding,
            "created_at": self.created_at,
        }

    def to_dict_without_embedding(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict omitting the embedding vector."""
        data = self.to_jsonl_dict()
        del data["embedding"]
        return data

    @classmethod
    def from_jsonl_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Hydrate an entry from a parsed JSONL line in the WP1.6 schema."""
        return cls(
            user_id=data["user_id"],
            query=str(data.get("query", "")),
            outcome=_coerce_outcome(data.get("outcome", "failed")),
            items=[MemoryItem.from_dict(it) for it in data.get("items", [])],
            embedding=data.get("embedding"),
            created_at=str(data.get("created_at", "")),
        )


def _coerce_outcome(raw: Any) -> Outcome:
    """Constrain ``raw`` to one of the three literal outcomes.

    Defaults to ``"failed"`` on any unrecognised value so a hand-edited
    JSONL file with a typo does not silently produce a memory tagged
    as a validated strategy. ``"shared"`` is recognised so the WP2
    cross-user store (WP2-plan §7.1 / §8.1) round-trips cleanly when
    loaded back via :meth:`MemoryStore.load`; pre-WP2 per-user JSONLs
    that only ever wrote ``"successful"`` / ``"failed"`` continue to
    load identically.
    """
    if raw == "successful":
        return "successful"
    if raw == "shared":
        return "shared"
    return "failed"


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Return indices of the ``k`` rows in ``matrix`` most similar to ``query``.

    Uses cosine similarity. Zero-norm rows would produce NaNs, so the
    denominator is clamped to a small positive number; in practice
    sentence-transformers outputs are never zero-length.
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
    """Per-user JSONL-backed memory store with in-memory cosine search.

    The on-disk format is one JSON object per line (no trailing comma,
    no array wrapper) so a partial write of a single ``add_entry`` cannot
    corrupt earlier records, and the file remains easy to read by hand.
    """

    def __init__(self, path: Path, user_id: str, embedder: Embedder) -> None:
        self.path = Path(path)
        self.user_id = user_id
        self.embedder = embedder
        self._entries: list[MemoryEntry] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def load(cls, path: Path, user_id: str, embedder: Embedder) -> MemoryStore:
        """Construct a store and populate it from ``path`` if the file exists.

        A missing file is not an error: it just means this user has no
        memories yet, and the first :meth:`add_entry` call will create
        the file (along with any missing parent directories).
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
        """Embed ``query`` and append the entry on disk and in memory.

        The embedding is computed over ``query`` (not over the items)
        and stored inline on the JSONL line so reload never has to
        call the model again. Empty ``items`` are allowed but
        discouraged; callers should drop a 0-item extraction at the
        pipeline layer rather than write a useless entry.
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
        """Return the ``k`` stored entries whose query is most similar to ``query``.

        Returns an empty list when the store is empty or ``k <= 0``.
        Results are sorted by descending cosine similarity over the
        stored query embeddings. The caller is responsible for
        flattening ``entry.items`` if it wants a flat list of items.
        """
        if not self._entries or k <= 0:
            return []
        query_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        indices = _cosine_top_k(query_vec, self._matrix, k)
        return [self._entries[i] for i in indices]

    def all(self) -> list[MemoryEntry]:
        """Return every stored entry in insertion order (defensive copy)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
