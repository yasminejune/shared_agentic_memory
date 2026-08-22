"""Slow tests that load the real all-MiniLM-L6-v2 Embedder."""

from __future__ import annotations

import pytest

from agent_memories.memory import Embedder


@pytest.mark.slow
@pytest.mark.unit
def test_embed_returns_a_fixed_length_vector_of_floats() -> None:
    embedder = Embedder()

    vector = embedder.embed("hello world")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.slow
@pytest.mark.unit
def test_same_text_gives_same_vector() -> None:
    embedder = Embedder()

    v1 = embedder.embed("the quick brown fox")
    v2 = embedder.embed("the quick brown fox")

    assert v1 == v2


@pytest.mark.slow
@pytest.mark.unit
def test_embed_batch_matches_single_embed() -> None:
    embedder = Embedder()

    texts = ["alpha", "beta", "gamma"]
    batch = embedder.embed_batch(texts)
    singles = [embedder.embed(t) for t in texts]

    assert batch == singles


@pytest.mark.slow
@pytest.mark.unit
def test_embed_batch_on_empty_input() -> None:
    embedder = Embedder()

    assert embedder.embed_batch([]) == []
