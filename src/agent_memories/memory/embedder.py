"""Sentence-transformers wrapper. Default model is all-MiniLM-L6-v2.

The rest of the codebase never imports the library directly. The model
loads in __init__ so the cost is paid at construction, not mid-turn.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Dense embedder. Returns list[float], not numpy arrays.

    MemoryStore rebuilds a numpy matrix from those lists for cosine search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed one string."""
        vector: Any = self._model.encode(text, convert_to_numpy=True)
        return [float(x) for x in vector.tolist()]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings in one encode call."""
        if not texts:
            return []
        matrix: Any = self._model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in row] for row in matrix.tolist()]
