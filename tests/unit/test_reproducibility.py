"""Seed loading and global seeding for reproducible runs."""

from __future__ import annotations

import random

import numpy as np
import pytest

from agent_memories.config.reproducibility import load_random_seed, set_global_seed

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw", ["3006", "  3006  "])
def test_load_random_seed_reads_env(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("RANDOM_SEED", raw)
    assert load_random_seed() == 3006


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (None, "RANDOM_SEED is not set"),
        ("   ", "RANDOM_SEED is not set"),
        ("abc", "must be an integer"),
    ],
)
def test_load_random_seed_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, message: str
) -> None:
    if raw is None:
        monkeypatch.delenv("RANDOM_SEED", raising=False)
    else:
        monkeypatch.setenv("RANDOM_SEED", raw)

    with pytest.raises(ValueError, match=message):
        load_random_seed()


def test_set_global_seed_affects_random_and_numpy() -> None:
    set_global_seed(123)
    a_random = random.random()
    a_numpy = float(np.random.random())

    set_global_seed(123)

    assert random.random() == a_random
    assert float(np.random.random()) == a_numpy
