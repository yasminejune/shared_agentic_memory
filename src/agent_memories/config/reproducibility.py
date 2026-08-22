"""Read RANDOM_SEED from the environment and apply it.

There is no numeric default. A missing or invalid value fails so a
run cannot proceed without a recorded seed.
"""

from __future__ import annotations

import os
import random

import numpy as np


def load_random_seed() -> int:
    """Read RANDOM_SEED from the environment.

    Raises ValueError if it is missing, empty, or not an int.
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
    """Seed random, numpy, and torch if it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
