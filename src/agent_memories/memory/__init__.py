"""Private memory store and dense RAG retrieval (WP1.6).

Three small modules layered for forward compatibility:

* :mod:`embedder` wraps the sentence-transformers model used to vectorise
  both stored task queries and retrieval queries.
* :mod:`store` defines the :class:`MemoryItem` and :class:`MemoryEntry`
  records and the per-user :class:`MemoryStore` (JSONL on disk, numpy
  cosine search in memory over query embeddings).
* :mod:`pipeline` runs the WP1.6 ReasoningBank memory-creation step:
  an LLM-as-Judge classifies the trajectory, then a success- or
  failure-specific extractor distils up to three structured items.

WP2.4 will instantiate a second :class:`MemoryStore` for the shared
cross-user store. Nothing else in this package should need to change.
"""

from .embedder import Embedder
from .store import MemoryEntry, MemoryItem, MemoryStore

__all__ = ["Embedder", "MemoryItem", "MemoryEntry", "MemoryStore", "MemoryPipeline"]


def __getattr__(name: str) -> object:
    """Load :class:`MemoryPipeline` only when a caller asks for it.

    Importing ``MemoryEntry`` (Step 1) must not pull LangGraph via the
    WP1.6 pipeline module.
    """
    if name == "MemoryPipeline":
        from .pipeline import MemoryPipeline

        return MemoryPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
