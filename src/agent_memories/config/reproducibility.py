"""Explicit random-seed helpers for reproducible evaluation runs.

``RANDOM_SEED`` must be set in the environment (typically via ``.env``).
There is no numeric default: a missing or invalid value fails fast so a
run cannot silently proceed without a recorded seed.
"""

from __future__ import annotations

import os
import random

import numpy as np


def load_random_seed() -> int:
    """Read ``RANDOM_SEED`` from the environment.

    Raises:
        ValueError: If ``RANDOM_SEED`` is missing, empty, or not an int.
    """
    raw = os.environ.get("RANDOM_SEED")
    if raw is None or not str(raw).strip():
        raise ValueError(
            "RANDOM_SEED is not set. Add it to .env (e.g. RANDOM_SEED=3006) "
            "before running evaluation scripts."
        )
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(
            f"RANDOM_SEED must be an integer, got {raw!r}. "
            "Set it in .env (e.g. RANDOM_SEED=3006)."
        ) from exc


def set_global_seed(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
