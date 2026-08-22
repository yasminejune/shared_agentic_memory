"""Tests for agent_memories.config.reproducibility."""

from __future__ import annotations

import random

import numpy as np
import pytest

from agent_memories.config.reproducibility import load_random_seed, set_global_seed


@pytest.mark.unit
def test_load_random_seed_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANDOM_SEED", "3006")
    assert load_random_seed() == 3006


@pytest.mark.unit
def test_load_random_seed_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANDOM_SEED", "  42  ")
    assert load_random_seed() == 42


@pytest.mark.unit
def test_load_random_seed_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    with pytest.raises(ValueError, match="RANDOM_SEED is not set"):
        load_random_seed()


@pytest.mark.unit
def test_load_random_seed_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANDOM_SEED", "   ")
    with pytest.raises(ValueError, match="RANDOM_SEED is not set"):
        load_random_seed()


@pytest.mark.unit
def test_load_random_seed_raises_when_not_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANDOM_SEED", "abc")
    with pytest.raises(ValueError, match="must be an integer"):
        load_random_seed()


@pytest.mark.unit
def test_set_global_seed_affects_random_and_numpy() -> None:
    set_global_seed(123)
    a_random = random.random()
    a_numpy = float(np.random.random())
    set_global_seed(123)
    assert random.random() == a_random
    assert float(np.random.random()) == a_numpy
