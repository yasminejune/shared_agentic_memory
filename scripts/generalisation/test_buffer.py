"""Smoke X-gating and the carry-over buffer, no LLM.

select_round2_inputs is index lists; write_buffer / read_buffer
is versioned JSONL. Three checks:

1. A small (bucket_sizes, X) grid so the policy is visible:
   full buckets go to round 2, the rest are set aside.
2. Round-trip a synthetic carry-over list through a temp file
   under data/memories/.shared_buffer_test.jsonl.
3. Fake assignments + fake entries, resolve carry_over indices
   back to MemoryEntry objects, persist, reload.

Temp file is deleted unless --keep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR
from agent_memories.generalisation import (
    GatingResult,
    entry_id,
    read_buffer,
    select_round2_inputs,
    write_buffer,
)
from agent_memories.memory import MemoryEntry, MemoryItem

DEFAULT_BUFFER_PATH = DEFAULT_MEMORY_DIR / ".shared_buffer_test.jsonl"


def _fake_entry(user_id: str, query: str, *, dim: int = 4) -> MemoryEntry:
    """Deterministic MemoryEntry for buffer tests.

    Tiny embedding seeded by hash(query) so the JSONL serialiser has
    something to write without standing up an Embedder. Title /
    description / content are dummy strings so the persisted file
    still reads like a memory if you open it.
    """
    seed = abs(hash(query)) % 997
    embedding = [float((seed + i) % 13) / 13.0 for i in range(dim)]
    return MemoryEntry(
        user_id=user_id,
        query=query,
        outcome="successful",
        items=[
            MemoryItem(
                title=f"Lesson from {query!r}",
                description="Synthetic test entry.",
                content=f"Insight derived from {query!r} for user {user_id}.",
            )
        ],
        embedding=embedding,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _format_gating(result: GatingResult) -> str:
    """One-line dump of label_inputs and carry_over."""
    parts: list[str] = []
    for label_idx, inputs in enumerate(result.label_inputs):
        if inputs is None:
            parts.append(f"L{label_idx}=skip")
        else:
            parts.append(f"L{label_idx}={inputs}")
    return (
        f"  triggered={result.triggered_labels}  {'  '.join(parts)}  carry_over={result.carry_over}"
    )


def _run_gating_grid() -> None:
    """Print the gating policy on a handful of cases."""
    print("=" * 72)
    print("X-gating policy on a representative grid:")
    print("=" * 72)
    grid: list[tuple[list[list[int]], int]] = [
        ([[0, 1, 2], [3, 4], [5]], 3),
        ([[0, 1, 2, 3, 4], [5, 6, 7], [8]], 3),
        ([[0], [1], [2]], 3),
        ([[0, 1, 2, 3, 4, 5]], 2),
        ([[], [0, 1, 2], []], 2),
    ]
    for buckets, x in grid:
        print(f"buckets={buckets}  X={x}")
        result = select_round2_inputs(buckets, x_per_label=x)
        print(_format_gating(result))
        print()


def _run_roundtrip(path: Path) -> bool:
    """Write a synthetic carry-over list, reload it, check equality."""
    print("=" * 72)
    print(f"Persistence round-trip via {path}:")
    print("=" * 72)
    original = [
        _fake_entry("user_00", "Find cheapest hairbrush on Amazon"),
        _fake_entry("user_01", "Browse search results for hairbrush"),
        _fake_entry("user_02", "Look up reviews for hairbrush"),
    ]
    write_buffer(original, path)
    loaded = read_buffer(path)

    matched = len(loaded) == len(original) and all(
        a.to_jsonl_dict() == b.to_jsonl_dict() for a, b in zip(loaded, original, strict=True)
    )
    print(f"Wrote {len(original)} entries; reloaded {len(loaded)}; equal={matched}")
    for entry in loaded:
        print(
            f"  user={entry.user_id} query={entry.query!r} "
            f"outcome={entry.outcome} entry_id={entry_id(entry)!r} "
            f"embedding_dim={len(entry.embedding or [])}"
        )
    return matched


def _run_orchestrator_glue(path: Path) -> None:
    """End-of-trigger glue: gating -> carry_over indices -> persist."""
    print()
    print("=" * 72)
    print("End-of-trigger orchestrator glue (gating -> carry_over -> persist):")
    print("=" * 72)
    entries = [
        _fake_entry(f"user_{i:02d}", f"Task {i}: search the web for some product") for i in range(7)
    ]
    buckets = [[0, 1, 2, 3], [4, 5], [6]]  # 3 labels, sizes 4/2/1
    x = 3
    result = select_round2_inputs(buckets, x_per_label=x)
    print(f"Labels triggered (bucket size >= X={x}): {result.triggered_labels}")
    print(f"Carry-over entry indices: {result.carry_over}")

    carry_entries = [entries[i] for i in result.carry_over]
    write_buffer(carry_entries, path)
    reloaded = read_buffer(path)
    print(f"Persisted {len(carry_entries)} carry-over entries; reloaded {len(reloaded)}.")
    for entry in reloaded:
        print(f"  carry-over: user={entry.user_id} query={entry.query!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--buffer-path",
        type=Path,
        default=DEFAULT_BUFFER_PATH,
        help=f"Temp JSONL for the round-trip check (default: {DEFAULT_BUFFER_PATH}).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete the temp buffer file at the end.",
    )
    args = parser.parse_args(argv)

    _run_gating_grid()
    matched = _run_roundtrip(args.buffer_path)
    _run_orchestrator_glue(args.buffer_path)

    if not args.keep and args.buffer_path.exists():
        args.buffer_path.unlink()
        print()
        print(f"Removed temp buffer at {args.buffer_path}.")

    print()
    print(f"Round-trip equality: {matched}")
    if not matched:
        raise SystemExit("Round-trip equality check failed.")


if __name__ == "__main__":
    main()
