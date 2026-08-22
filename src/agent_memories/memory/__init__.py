"""Private JSONL store, dense embedder, and ReasoningBank pipeline.

The shared store reuses MemoryStore on a different path.
"""

from .embedder import Embedder
from .store import MemoryEntry, MemoryItem, MemoryStore

__all__ = ["Embedder", "MemoryItem", "MemoryEntry", "MemoryStore", "MemoryPipeline"]


def __getattr__(name: str) -> object:
    """Load MemoryPipeline only when a caller asks for it.

    MemoryEntry should not pull in the judge and extractor prompts.
    """
    if name == "MemoryPipeline":
        from .pipeline import MemoryPipeline

        return MemoryPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
