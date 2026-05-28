"""Per-user :class:`MemoryStore` and the :class:`Memory` record it holds.

Storage is intentionally minimal: one JSONL file per user, one record
per line, embedding inlined alongside the text. Reload is therefore
O(file size) and never needs to call the embedder again, which keeps
the controlled-task smoke runs deterministic across machines.

Cosine search runs against an in-memory ``numpy`` matrix that the store
rebuilds whenever a memory is added. WP1.5 is operating at the
seed-handful scale; brute-force cosine is far simpler than introducing
FAISS or a vector DB, and the proposal explicitly punts retrieval
optimisation (§3.2.2).

Forward compatibility:

* :class:`Memory` carries a free-form ``metadata`` dict that WP1.6
  ReasoningBank can populate with ``{"title", "description", "outcome"}``
  without changing the on-disk format.
* WP2.4's shared cross-user store is just a second :class:`MemoryStore`
  pointed at a different path (e.g. ``data/memories/shared.jsonl``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import Embedder


@dataclass
class Memory:
    """One stored memory.

    ``text`` is the only field used for embedding and retrieval.
    ``metadata`` is reserved for downstream work packages (WP1.6
    ReasoningBank fields, WP2 shared-store flags) and is empty in WP1.5.
    ``embedding`` is stored inline so reload does not re-embed; tests
    that construct fixtures by hand can also leave it ``None`` and call
    :meth:`MemoryStore.add` instead of building the record directly.
    """

    text: str
    user_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = ""

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for the on-disk JSONL line."""
        return asdict(self)

    @classmethod
    def from_jsonl_dict(cls, data: dict[str, Any]) -> Memory:
        """Hydrate a memory from a parsed JSONL line."""
        return cls(
            text=data["text"],
            user_id=data["user_id"],
            metadata=data.get("metadata") or {},
            embedding=data.get("embedding"),
            created_at=data.get("created_at", ""),
        )


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
    no array wrapper) so a partial write of a single ``add`` cannot
    corrupt earlier records, and seed fixtures are easy to read by hand.
    """

    def __init__(self, path: Path, user_id: str, embedder: Embedder) -> None:
        self.path = Path(path)  # where the memories are stored
        self.user_id = user_id
        self.embedder = embedder
        self._memories: list[Memory] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def load(cls, path: Path, user_id: str, embedder: Embedder) -> MemoryStore:
        """Construct a store and populate it from ``path`` if the file exists.

        A missing file is not an error: it just means this user has no
        memories yet, and the first :meth:`add` call will create the
        file (along with any missing parent directories).
        """
        store = cls(path=path, user_id=user_id, embedder=embedder)
        if store.path.exists():
            store._load_from_disk()
        return store

    def _load_from_disk(self) -> None:
        memories: list[Memory] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                memories.append(Memory.from_jsonl_dict(json.loads(line)))
        self._memories = memories
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        embeddings = [m.embedding for m in self._memories if m.embedding is not None]
        if not embeddings:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        self._matrix = np.asarray(embeddings, dtype=np.float32)

    def add(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Embed ``text`` and append it to the store on disk and in memory.

        The embedding is computed via :attr:`embedder` and stored inline
        on the JSONL line so reload never has to call the model again.
        """
        embedding = self.embedder.embed(text)
        memory = Memory(
            text=text,
            user_id=self.user_id,
            metadata=dict(metadata or {}),
            embedding=embedding,
            created_at=_utcnow_iso(),
        )
        self._append_to_disk(memory)
        self._memories.append(memory)
        self._rebuild_matrix()
        return memory

    def _append_to_disk(self, memory: Memory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(memory.to_jsonl_dict(), ensure_ascii=False) + "\n")

    def add_many(self, texts: Iterable[str]) -> list[Memory]:
        """Batch-embed and append several memories.

        Faster than looping :meth:`add` because it issues one
        :meth:`Embedder.embed_batch` call.
        """
        text_list = list(texts)
        if not text_list:
            return []
        embeddings = self.embedder.embed_batch(text_list)
        created: list[Memory] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for text, embedding in zip(text_list, embeddings, strict=True):
                memory = Memory(
                    text=text,
                    user_id=self.user_id,
                    metadata={},
                    embedding=embedding,
                    created_at=_utcnow_iso(),
                )
                fh.write(json.dumps(memory.to_jsonl_dict(), ensure_ascii=False) + "\n")
                self._memories.append(memory)
                created.append(memory)
        self._rebuild_matrix()
        return created

    def search(self, query: str, *, k: int) -> list[Memory]:
        """Return the ``k`` stored memories most similar to ``query``.

        Returns an empty list when the store is empty or ``k <= 0``.
        Results are sorted by descending cosine similarity.
        """
        if not self._memories or k <= 0:
            return []
        query_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        indices = _cosine_top_k(query_vec, self._matrix, k)
        return [self._memories[i] for i in indices]

    def all(self) -> list[Memory]:
        """Return every stored memory in insertion order (defensive copy)."""
        return list(self._memories)

    def __len__(self) -> int:
        return len(self._memories)
