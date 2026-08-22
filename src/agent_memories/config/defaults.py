"""Repo root, default memory directory, and default user id.

AGENT_MEMORY_USER_ID overrides the default user_a. The OS login is
not used: user ids here are simulated users, not whoever is logged in.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

DEFAULT_USER_ID: str = os.getenv("AGENT_MEMORY_USER_ID", "user_a")
DEFAULT_MEMORY_DIR: Path = REPO_ROOT / "data" / "memories"
