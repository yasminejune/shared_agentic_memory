"""Multi-trajectory smoke runner for the WP1.5 / WP1.6 agent loop.

Repeats ``scripts/memories/WP1_5.py`` ``N`` times against the same
aim, varying the ``--user-id`` so each run writes to a distinct
``data/memories/<user_id>.jsonl``. After the loop, lists every
per-user JSONL touched and the number of entries it now holds, so the
human can eyeball whether each trajectory produced a private memory
before the WP2 shared-memory pipeline (``scripts/memories/WP2_pipeline.py``)
consumes them.

This is the Phase 1.1 standalone harness for the multi-trajectory
orchestration step: WP1.5 itself only runs one trajectory per
invocation, so this loop is the smallest stand-alone exerciser of
"N trajectories produce N per-user stores" without yet adding the
round-1 / round-2 / buffer logic that the WP2 orchestrator layers on
top. Each trajectory runs as its own ``python scripts/memories/WP1_5.py``
subprocess so the Playwright browser, the LLM clients, and the
sentence-transformers Embedder are all torn down cleanly between
runs and the failure of one trajectory does not poison the next.

Defaults to ``N = 2`` for the smoke test; pass ``--n 10`` to scale
to the WP2 trigger-window size.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR
from agent_memories.memory import Embedder, MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[2]
WP1_5_SCRIPT = REPO_ROOT / "scripts" / "memories" / "WP1_5.py"

DEFAULT_N = 2
DEFAULT_AIM = "Find amazon results for paper"
DEFAULT_URL = "https://www.amazon.co.uk"
DEFAULT_USER_PREFIX = "user_"
DEFAULT_MAX_STEPS = 10


def _user_id_for(prefix: str, idx: int) -> str:
    """Render the ``user_id`` used for trajectory ``idx``.

    Zero-padded to two digits so a 10-trajectory run sorts in
    insertion order on the filesystem (``user_00.jsonl`` before
    ``user_10.jsonl``) and stays readable in the per-user store
    summary printed at the end.
    """
    return f"{prefix}{idx:02d}"


def _run_one(
    *,
    aim: str,
    url: str,
    user_id: str,
    max_steps: int,
    headless: bool,
    model: str,
    memory_dir: Path,
) -> int:
    """Spawn one WP1.5 subprocess and return its exit code.

    Streaming stdout / stderr to the parent terminal (no ``capture``)
    keeps the per-trajectory ``[Observe]: / [Think]: / [Act]:`` trace
    visible during the run, matching the conventions decision in
    ``.claude/memory/decisions.md`` ("terminal output for a WP1 run
    is exactly three trace prefixes").
    """
    cmd = [
        sys.executable,
        str(WP1_5_SCRIPT),
        "--aim",
        aim,
        "--url",
        url,
        "--user-id",
        user_id,
        "--max-steps",
        str(max_steps),
        "--model",
        model,
        "--memory-dir",
        str(memory_dir),
    ]
    if headless:
        cmd.append("--headless")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _summarise_stores(memory_dir: Path, user_ids: list[str]) -> None:
    """Print one line per per-user store written by the loop.

    Loads each :class:`MemoryStore` through the regular
    :meth:`MemoryStore.load` path so we get the same entry-count the
    WP2 orchestrator will see when it reads them back. Stores with
    zero entries (the trajectory's LLM extractor produced no
    parseable items) are listed too so the user knows which
    trajectories silently dropped.
    """
    embedder = Embedder()
    print()
    print("=" * 72)
    print(f"Per-user store summary under {memory_dir}:")
    print("=" * 72)
    for user_id in user_ids:
        store_path = memory_dir / f"{user_id}.jsonl"
        if not store_path.exists():
            print(f"  {user_id}: no JSONL written (trajectory may have errored).")
            continue
        store = MemoryStore.load(store_path, user_id=user_id, embedder=embedder)
        print(f"  {user_id}: {len(store)} entry/entries at {store_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of trajectories to run (default: {DEFAULT_N}).",
    )
    parser.add_argument("--aim", default=DEFAULT_AIM)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--user-prefix",
        default=DEFAULT_USER_PREFIX,
        help=(
            "Prefix for the per-trajectory user_id; the loop appends a "
            "two-digit index so 10 trajectories sort as user_00..user_09."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    args = parser.parse_args(argv)

    if args.n < 1:
        parser.error("--n must be >= 1.")

    user_ids = [_user_id_for(args.user_prefix, i) for i in range(args.n)]

    print(f"Running {args.n} trajector{'y' if args.n == 1 else 'ies'} of aim={args.aim!r}")
    print(f"user_ids: {user_ids}")
    print()

    exit_codes: list[int] = []
    for idx, user_id in enumerate(user_ids):
        print("=" * 72)
        print(f"Trajectory {idx + 1} / {args.n} (user_id={user_id})")
        print("=" * 72)
        code = _run_one(
            aim=args.aim,
            url=args.url,
            user_id=user_id,
            max_steps=args.max_steps,
            headless=args.headless,
            model=args.model,
            memory_dir=args.memory_dir,
        )
        exit_codes.append(code)
        if code != 0:
            print(f"[run_multi_trajectory] WP1_5.py exited {code} for {user_id}; continuing.")

    _summarise_stores(args.memory_dir, user_ids)

    print()
    n_ok = sum(1 for c in exit_codes if c == 0)
    print(f"Done: {n_ok} / {args.n} trajectories returned exit code 0.")


if __name__ == "__main__":
    main()
