"""Smoke tests for the sentence-transformers :class:`Embedder` wrapper.

These tests load the real ``all-MiniLM-L6-v2`` model and are marked
``slow`` so the default ``make test`` run does not pay the model-load
cost on every invocation. Run with ``pytest -m slow`` to include them.
"""

from __future__ import annotations

import pytest

from agent_memories.memory import Embedder


@pytest.mark.slow
@pytest.mark.unit
def test_embed_returns_a_fixed_length_vector_of_floats() -> None:
    """A single ``embed`` call must produce a non-empty list of floats."""
    embedder = Embedder()

    vector = embedder.embed("hello world")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.slow
@pytest.mark.unit
def test_same_text_gives_same_vector() -> None:
    """Determinism: embedding the same string twice must yield identical output."""
    embedder = Embedder()

    v1 = embedder.embed("the quick brown fox")
    v2 = embedder.embed("the quick brown fox")

    assert v1 == v2


@pytest.mark.slow
@pytest.mark.unit
def test_embed_batch_matches_single_embed() -> None:
    """``embed_batch`` is just batched ``embed``; results must agree element-wise."""
    embedder = Embedder()

    texts = ["alpha", "beta", "gamma"]
    batch = embedder.embed_batch(texts)
    singles = [embedder.embed(t) for t in texts]

    assert batch == singles


@pytest.mark.slow
@pytest.mark.unit
def test_embed_batch_on_empty_input() -> None:
    """An empty input list must short-circuit to an empty output list."""
    embedder = Embedder()

    assert embedder.embed_batch([]) == []
