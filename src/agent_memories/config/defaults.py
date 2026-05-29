"""Project-wide defaults for any memory-touching code.

Centralising these here keeps the WP1.5+ runner, the WP1.6 ReasoningBank
script (future), and the WP2 shared-store glue from each redefining the
same constants. Tests that need a different value should construct
:class:`agent_memories.memory.MemoryStore` directly with an explicit
``path`` and ``user_id`` rather than monkey-patching this module.

The single env-var hook (``AGENT_MEMORY_USER_ID``) lets a developer flip
to a different default without code edits -- useful on shared dev
machines -- but the *project* default is the literal ``"user_a"``. The
OS user is intentionally *not* consulted: the user-id semantics in this
project refer to simulated users for the WP2 privacy proof, not to
whoever happens to be logged in.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

DEFAULT_USER_ID: str = os.getenv("AGENT_MEMORY_USER_ID", "user_a")
DEFAULT_MEMORY_DIR: Path = REPO_ROOT / "data" / "memories"
