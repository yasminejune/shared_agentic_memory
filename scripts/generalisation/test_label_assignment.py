"""Standalone smoke for the WP2 round-1 label -> round-2 batch assignment.

Reads ``K`` DP-released label strings from
``scripts/amin_et_al/outputs/labels.jsonl`` (produced by
``scripts/amin_et_al/WP2_8.py``) and every per-user
:class:`MemoryEntry` it can find under ``data/memories/<user_id>.jsonl``,
then runs :func:`agent_memories.generalisation.assign_memories_to_labels`
to assign each memory to its nearest label by cosine similarity over
the WP1.6 ``query`` embedding (already stored on each entry) against
freshly-computed label embeddings (same ``all-MiniLM-L6-v2`` model).

This is the Phase 1.2 stand-alone exerciser of the assignment
function the WP2 orchestrator depends on (WP2-plan §3.2 / §4.3
round-2 batch assignment). No DP cost is touched here -- the labels
are already DP-released and the assignment is pure post-processing.

Output: the per-label bucket sizes and the (user_id, query,
assigned_label_index, label_string, similarity) tuples printed to
stdout, so the human can spot-check that semantically related
memories cluster under the same label before wiring the orchestrator
to it. No JSONL is written.
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
    """Read one label string per line from ``WP2_8.py``'s JSONL output.

    The JSONL schema is ``{"index": int, "label": str, "parsed":
    bool, "created_at": str}`` per ``WP2_8.parse_labels``; we only
    need the ``label`` field for assignment. Fallback ``label_<i>``
    entries (``parsed = false``) are kept in the list because the
    round-2 pipeline still receives ``k`` labels and a fallback
    label is a legitimate cosine-similarity target even if it is
    not very informative.
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
    """Return every per-user :class:`MemoryEntry` discovered under ``memory_dir``.

    Each entry already carries its own ``user_id`` (set by
    :meth:`MemoryStore.add_entry` from the store's own ``self.user_id``),
    so the smoke-test reads it off the entry directly rather than
    re-deriving it from the filename.

    Only ``<memory_dir>/user_*.jsonl`` is considered, matching the
    allow-list pattern used by ``scripts/memories/WP2_pipeline.py``'s
    ``_load_per_user_entries``. The WP2 shared cross-user store
    (``shared.jsonl``) and its hidden sidecars (``.shared_audit.jsonl``,
    ``.shared_buffer.jsonl``, ``.shared_state.json``) all lack the
    ``user_`` prefix and are therefore excluded automatically.
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
    """Pretty-print buckets, then per-memory assignments, to stdout."""
    buckets = group_by_label(assignments, n_labels=len(labels))

    print()
    print("=" * 72)
    print(f"Per-label bucket sizes (K={len(labels)}, N_memories={len(entries)}):")
    print("=" * 72)
    for label_idx, bucket in enumerate(buckets):
        print(f"  [{label_idx}] ({len(bucket):>2}) {labels[label_idx]!r}")

    print()
    print("=" * 72)
    print("Per-memory assignments:")
    print("=" * 72)
    for assignment in assignments:
        entry = entries[assignment.entry_index]
        label_str = labels[assignment.label_index]
        query_preview = (entry.query[:60] + "...") if len(entry.query) > 60 else entry.query
        print(
            f"  user={entry.user_id:<12} "
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
        f"Loaded {len(entries)} memory entry/entries across "
        f"{len({entry.user_id for entry in entries})} user store(s)"
    )

    assignments = assign_memories_to_labels(entries, labels, embedder=embedder)
    _print_summary(labels, entries, assignments)


if __name__ == "__main__":
    main()
