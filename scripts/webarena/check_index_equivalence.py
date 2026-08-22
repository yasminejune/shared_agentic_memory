"""One-off check: the unified MemoryIndex retrieves what the old runners did.

Compares, on real intents from the condition A CSV:

  B  old PrivateMemoryIndex   vs  MemoryIndex over private records
  C  old MemoryStore.search   vs  MemoryIndex over shared records
  D  old CombinedMemoryIndex  vs  MemoryIndex over private + shared

Loads the embedder, so run it only when no WebArena batch is in flight.
Delete this script once the unified runner is trusted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_WEBARENA_DIR = Path(__file__).resolve().parent
if str(_WEBARENA_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBARENA_DIR))

import run_webarena_with_private_memories as old_b
import run_webarena_with_private_shared_memories as old_d
from common import (
    DEFAULT_MEMORIES_CSV,
    DEFAULT_SHARED_STORE,
    DEFAULT_TRAJECTORIES_CSV,
    MemoryIndex,
    audit_ids,
    load_memory_entries_from_csv,
    load_private_records,
    load_shared_records,
)

from agent_memories.memory import Embedder, MemoryStore

K = 3
N_INTENTS = 25


def sample_intents(csv_path: Path, limit: int) -> list[str]:
    """Real intents from the condition A trajectories CSV."""
    csv.field_size_limit(sys.maxsize)
    intents: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            intent = row.get("intent", "").strip()
            if intent and not intent.startswith("(") and intent not in intents:
                intents.append(intent)
            if len(intents) >= limit:
                break
    return intents


def report(name: str, old: list[list], new: list[list]) -> bool:
    mismatches = [(o, n) for o, n in zip(old, new) if o != n]
    if not mismatches:
        print(f"{name}: EQUIVALENT across {len(old)} intents")
        return True
    print(f"{name}: {len(mismatches)} of {len(old)} DIFFER")
    for o, n in mismatches[:5]:
        print(f"    old={o}\n    new={n}")
    return False


def main() -> None:
    intents = sample_intents(DEFAULT_TRAJECTORIES_CSV, N_INTENTS)
    print(f"Comparing top-{K} retrieval on {len(intents)} real intents\n", flush=True)

    embedder = Embedder()
    private_pairs = load_memory_entries_from_csv(DEFAULT_MEMORIES_CSV)
    shared_records = load_shared_records(DEFAULT_SHARED_STORE, embedder)
    shared_entries = [entry for _, _, entry in shared_records]

    ok = True

    # B: private only
    old_index = old_b.PrivateMemoryIndex(private_pairs, embedder)
    new_index = MemoryIndex(load_private_records(DEFAULT_MEMORIES_CSV), embedder)
    ok &= report(
        "B private",
        [[tid for tid, _ in old_index.search(q, k=K)] for q in intents],
        [audit_ids(new_index.search(q, k=K)) for q in intents],
    )

    # C: shared only
    store = MemoryStore.load(DEFAULT_SHARED_STORE, user_id="shared", embedder=embedder)
    new_index = MemoryIndex(shared_records, embedder)
    ok &= report(
        "C shared",
        [[entry.query for entry in store.search(q, k=K)] for q in intents],
        [audit_ids(new_index.search(q, k=K)) for q in intents],
    )

    # D: private + shared
    old_index = old_d.CombinedMemoryIndex(private_pairs, shared_entries, embedder)
    new_index = MemoryIndex(
        load_private_records(DEFAULT_MEMORIES_CSV) + shared_records,
        embedder,
    )
    ok &= report(
        "D combined",
        [old_d._audit_ids(old_index.search(q, k=K)) for q in intents],
        [audit_ids(new_index.search(q, k=K)) for q in intents],
    )

    print("\nAll conditions equivalent." if ok else "\nMISMATCH - do not switch over yet.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
