"""Private memory store and dense RAG retrieval (WP1.5).

Three small modules layered for forward compatibility:

* :mod:`embedder` wraps the sentence-transformers model used to vectorise
  both stored memories and queries.
* :mod:`store` defines the :class:`Memory` record and the per-user
  :class:`MemoryStore` (JSONL on disk, numpy cosine search in memory).
* :mod:`pipeline` runs the WP1.5 memory-creation step: ask the chat
  LLM whether the just-finished run contains anything worth keeping,
  with the top-5 most similar existing memories appended for dedup.

WP1.6 (ReasoningBank) will replace only :meth:`MemoryPipeline.create_from_run`;
WP2.4 will instantiate a second :class:`MemoryStore` for the shared
cross-user store. Nothing else in this package should need to change.
"""

from .embedder import Embedder
from .pipeline import MemoryPipeline
from .store import Memory, MemoryStore

__all__ = ["Embedder", "Memory", "MemoryStore", "MemoryPipeline"]
