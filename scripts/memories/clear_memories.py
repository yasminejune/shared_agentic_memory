"""Delete every per-user JSONL file under ``data/memories/``.

A scratch-reset helper for the WP1.5 memory store. The runner
(``scripts/memories/WP1_5.py``) will recreate the file on its next
run, so wiping the directory is enough -- there is no schema or
index to keep in sync.

Usage:
    python scripts/memories/clear_memories.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    args = parser.parse_args(argv)

    if not args.memory_dir.exists():
        print(f"[Memory] Nothing to clear: {args.memory_dir} does not exist.")
        return

    removed = 0
    for path in args.memory_dir.glob("*.jsonl"):
        path.unlink()
        print(f"[Memory] Deleted {path}")
        removed += 1

    print(f"[Memory] Cleared {removed} memory file(s) from {args.memory_dir}.")


if __name__ == "__main__":
    main()
