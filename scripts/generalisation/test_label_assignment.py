"""Smoke the round-1 label -> round-2 cosine assignment.

Reads K labels from scripts/amin_et_al/outputs/labels.jsonl
(WP2_8.py) and every user_*.jsonl under data/memories/, then
assign_memories_to_labels. Embeddings on the entries are already
there; labels get a fresh all-MiniLM-L6-v2 pass.

No privacy cost: labels are already released, assignment is
post-processing. Prints bucket sizes and (user, query, label, sim)
so I can see whether related memories land together. Nothing is
written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR
from agent_memories.generalisation import (
    LabelAssignment,
    assign_memories_to_labels,
    group_by_label,
)
from agent_memories.memory import Embedder, MemoryEntry, MemoryStore

DEFAULT_LABELS_PATH = Path("scripts/amin_et_al/outputs/labels.jsonl")


def _load_labels(path: Path) -> list[str]:
    """Read one label string per line from WP2_8.py's JSONL.

    Schema is ``{"index", "label", "parsed", "created_at"}``; only
    ``label`` is used. Fallback ``label_<i>`` rows (parsed=false) stay
    in the list because round 2 still expects K labels, and a dull
    fallback is still a cosine target.
    """
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels.append(str(record["label"]))
    return labels


def _load_all_user_entries(
    memory_dir: Path,
    embedder: Embedder,
) -> list[MemoryEntry]:
    """Every trajectory entry under ``memory_dir``.

    Each entry already carries ``user_id`` from MemoryStore.add_entry,
    so this reads it off the record rather than from the filename.

    Only ``user_*.jsonl`` (same allow-list as WP2_pipeline.py).
    shared.jsonl and the hidden sidecars do not match that prefix.
    """
    entries: list[MemoryEntry] = []
    for path in sorted(memory_dir.glob("user_*.jsonl")):
        store = MemoryStore.load(path, user_id=path.stem, embedder=embedder)
        entries.extend(store.all())
    return entries


def _print_summary(
    labels: list[str],
    entries: list[MemoryEntry],
    assignments: list[LabelAssignment],
) -> None:
    """Bucket sizes first, then per-entry assignments."""
    buckets = group_by_label(assignments, n_labels=len(labels))

    print()
    print("=" * 72)
    print(f"Per-label bucket sizes (K={len(labels)}, N_entries={len(entries)}):")
    print("=" * 72)
    for label_idx, bucket in enumerate(buckets):
        print(f"  [{label_idx}] ({len(bucket):>2}) {labels[label_idx]!r}")

    print()
    print("=" * 72)
    print("Per-entry assignments:")
    print("=" * 72)
    for assignment in assignments:
        entry = entries[assignment.item_index]
        label_str = labels[assignment.label_index]
        query_preview = (entry.query[:60] + "...") if len(entry.query) > 60 else entry.query
        print(
            f"  user={entry.user_id:<12} "
            f"n_items={len(entry.items)} "
            f"label_idx={assignment.label_index} "
            f"sim={assignment.similarity:+.3f} "
            f"label={label_str!r:<40} "
            f"query={query_preview!r}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=(
            "JSONL produced by scripts/amin_et_al/WP2_8.py; one record per "
            f"label (default: {DEFAULT_LABELS_PATH})."
        ),
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=DEFAULT_MEMORY_DIR,
        help="Directory holding per-user <user_id>.jsonl stores.",
    )
    args = parser.parse_args(argv)

    if not args.labels_path.exists():
        parser.error(
            f"--labels-path does not exist: {args.labels_path}. "
            "Run scripts/amin_et_al/WP2_8.py first to produce it."
        )

    labels = _load_labels(args.labels_path)
    if not labels:
        parser.error(f"No labels parsed from {args.labels_path}.")

    embedder = Embedder()
    entries = _load_all_user_entries(args.memory_dir, embedder)
    if not entries:
        parser.error(
            f"No per-user memory entries found under {args.memory_dir}. "
            "Run scripts/memories/run_multi_trajectory.py (or WP1_5.py) "
            "first to populate the per-user stores."
        )

    print(f"Loaded {len(labels)} label(s) from {args.labels_path}")
    print(
        f"Loaded {len(entries)} memory entries across "
        f"{len({entry.user_id for entry in entries})} user store(s)"
    )

    assignments = assign_memories_to_labels(entries, labels, embedder=embedder)
    _print_summary(labels, entries, assignments)


if __name__ == "__main__":
    main()
