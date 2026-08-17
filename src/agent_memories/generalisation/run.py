"""InvisibleInk shared-memory pipeline: run Steps 1–4 in order.

Counterpart to Amin ``scripts/memories/WP2_pipeline.py``. Each step is
still independently runnable via ``python -m agent_memories.generalisation.stepN_*``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memories.generalisation.io import (
    DEFAULT_SHARED_STORE,
    add_io_arguments,
    resolve_io_paths,
)
from agent_memories.generalisation.step1_labels import main as step1_main
from agent_memories.generalisation.step2_assignment import main as step2_main
from agent_memories.generalisation.step3_content import main as step3_main
from agent_memories.generalisation.step4_store import main as step4_main


def _forward(args: argparse.Namespace, extra: list[str] | None = None) -> list[str]:
    """Rebuild the shared CLI flags so each step sees the same paths."""
    forwarded = ["--run", str(args.run)]
    if args.memories_csv is not None:
        forwarded.extend(["--memories-csv", str(args.memories_csv)])
    if args.work_dir is not None:
        forwarded.extend(["--work-dir", str(args.work_dir)])
    if extra:
        forwarded.extend(extra)
    return forwarded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_io_arguments(parser)
    parser.add_argument(
        "--shared-store",
        type=Path,
        default=DEFAULT_SHARED_STORE,
        help=f"Shared MemoryStore JSONL (default: {DEFAULT_SHARED_STORE}).",
    )
    args = parser.parse_args(argv)

    csv_path, work_dir = resolve_io_paths(args)
    print(f"[run] memories CSV: {csv_path}")
    print(f"[run] work dir:     {work_dir}")
    print(f"[run] shared store: {args.shared_store}")

    shared = _forward(args)
    step1_main(shared)
    step2_main(shared)
    step3_main(shared)
    step4_main([*shared, "--shared-store", str(args.shared_store)])
    print("[run] Steps 1–4 complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
