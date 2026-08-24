"""Run WP1.5 N times, one private store per fake user.

Same aim each time, different ``--user-id``, so each trajectory writes
``data/memories/<user_id>.jsonl``. After the loop it lists every JSONL
and how many entries it holds. I look at that before feeding the
stores into anything shared.

WP1_5.py itself is one trajectory per process. This wrapper is the
smallest way to get N stores without pulling in the shared-memory
pipeline. Each run is a subprocess so Playwright, the LLM client,
and the embedder get torn down between users; one crash does not
take the rest with it.

N=2 by default. Pass ``--n 10`` if you want a fuller window.
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
    """``user_id`` for trajectory ``idx``, zero-padded to two digits.

    So a 10-run batch sorts as ``user_00.jsonl`` ... ``user_09.jsonl``
    and the summary at the end is readable.
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

    Stdout/stderr stream to this terminal (no capture) so the usual
    ``[Observe]:`` / ``[Think]:`` / ``[Act]:`` trace stays visible.
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
    """Print one line per per-user store the loop touched.

    Loaded through MemoryStore.load, so the count matches what the
    shared-memory pipeline will see. Zero-entry stores are listed
    too (extractor produced nothing parseable).
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
