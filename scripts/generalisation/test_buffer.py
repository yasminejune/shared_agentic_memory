"""Standalone smoke for the WP2 X-gating + carry-over buffer.

No LLM and no embedder are involved here -- the gating policy is
pure logic over batch-index lists, and the persistence layer is
versioned JSONL round-tripping of entry-level records. This is the
Phase 1.3 exerciser of :func:`agent_memories.generalisation.select_round2_inputs`
and the :func:`write_buffer` / :func:`read_buffer` pair, before the
WP2 orchestrator wires them in.

The script runs three things and prints the result of each:

1. A small grid of ``(bucket_sizes, X)`` cases through
   :func:`select_round2_inputs` so the gating policy ("pass every
   entry from a triggered label to round 2; set aside every bucket
   that did not hit ``X``") is visible at a glance.
2. A round-trip of a synthetic carry-over list through
   :func:`write_buffer` / :func:`read_buffer` against a temp path
   under ``data/memories/.shared_buffer_test.jsonl``, checking that
   the loaded entries equal the input on every field.
3. The orchestrator's end-of-trigger glue: given fake assignments
   and a fake list of memory entries, resolve the
   :attr:`GatingResult.carry_over` indices back to
   :class:`MemoryEntry` objects, persist them, reload them, and
   print the resulting list.

The temp persistence file is removed at the end of the script so
repeated runs stay deterministic. Pass ``--keep`` to leave it in
place for manual inspection.
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
    """Construct a deterministic entry-level record for buffer tests.

    The embedding is a tiny ``dim``-vector seeded by ``hash(query)``
    so the JSONL serialiser has something to write but the test
    does not depend on a real :class:`Embedder`. The entry carries
    plausible-looking title / description / content strings so a
    visual inspection of the persisted JSONL reads naturally.
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
    """One-line dump of label_inputs and carry_over for the gating grid."""
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
    """Print the gating policy on a handful of representative cases."""
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
    """Persist a synthetic carry-over and reload it; return ``True`` on equality."""
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
    """Simulate the orchestrator's end-of-trigger carry-over flow."""
    print()
    print("=" * 72)
    print("End-of-trigger orchestrator glue (gating -> carry_over -> persist):")
    print("=" * 72)
    entries = [
        _fake_entry(f"user_{i:02d}", f"Task {i}: search the web for some product")
        for i in range(7)
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
