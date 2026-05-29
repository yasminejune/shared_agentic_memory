"""Dense embedder used by the WP1.5 RAG retrieval pipeline.

Thin wrapper around ``sentence_transformers.SentenceTransformer`` so the
rest of the codebase never imports the library directly. The default
model is ``all-MiniLM-L6-v2``: ~80 MB, 384-dimensional, deterministic
offline, and the de facto standard for academic dense-retrieval demos.

The model is loaded eagerly in ``__init__`` so the cost shows up at
construction time rather than on the first ``embed`` call inside a
Think turn. Tests construct one :class:`Embedder` and reuse it.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps a sentence-transformers model with a list-based interface.

    Returns plain ``list[float]`` rather than numpy arrays so callers
    (including the JSONL serialiser in :mod:`store`) do not have to
    care about numpy types. The :class:`MemoryStore` rebuilds a numpy
    matrix internally for cosine search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Return a 1-D embedding vector for a single string."""
        vector: Any = self._model.encode(text, convert_to_numpy=True)
        return [float(x) for x in vector.tolist()]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input string.

        Batches via the underlying model's ``encode`` call so embedding
        many strings at once is much faster than calling :meth:`embed`
        in a loop.
        """
        if not texts:
            return []
        matrix: Any = self._model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in row] for row in matrix.tolist()]
