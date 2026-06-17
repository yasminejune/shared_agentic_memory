"""Standalone round-trip for the WP2 shared :class:`MemoryStore`.

Phase 1.4 exerciser for the additive change in
:mod:`agent_memories.memory.store` that extends the ``Outcome``
literal to include ``"shared"`` (WP2-plan §7.1 / §8.1). The shared
store is the same :class:`MemoryStore` class as the per-user one,
pointed at ``data/memories/shared.jsonl`` with the reserved
``user_id="shared"`` and ``query=<round-1 label string>`` (so cosine
retrieval matches a new agent query against the label, see
WP2-plan §7.1).

This script writes one entry through ``add_entry`` (which embeds the
label via the same :class:`Embedder` the per-user stores use), then
constructs a *second* store instance against the same path so the
reload path goes through ``_load_from_disk`` and exercises
:func:`_coerce_outcome` on the persisted ``"shared"`` outcome. It
then runs ``store.search(query=label, k=1)`` to confirm cosine
retrieval over the label embedding finds the round-tripped entry.

The test JSONL is removed at the end so repeated runs stay
deterministic; pass ``--keep`` to leave it in place for manual
inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR
from agent_memories.memory import Embedder, MemoryItem, MemoryStore

DEFAULT_SHARED_PATH = DEFAULT_MEMORY_DIR / "shared_test.jsonl"
DEFAULT_LABEL = "attending a recent event"
DEFAULT_ITEM = MemoryItem(
    title="Verify Venue and Date Before Payment",
    description=(
        "Confirm details on the issuer's page to avoid stale "
        "availability from third-party aggregators."
    ),
    content=(
        "When booking event tickets through a third-party platform, "
        "confirm the venue and date on the issuer's own page before "
        "paying, because the aggregator may show stale availability."
    ),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shared-path",
        type=Path,
        default=DEFAULT_SHARED_PATH,
        help=f"Temp shared-store JSONL (default: {DEFAULT_SHARED_PATH}).",
    )
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete the temp shared-store file at the end.",
    )
    args = parser.parse_args(argv)

    if args.shared_path.exists():
        args.shared_path.unlink()

    embedder = Embedder()

    print(f"Constructing writer store at {args.shared_path} ...")
    writer = MemoryStore.load(args.shared_path, user_id="shared", embedder=embedder)
    print(f"  initial size: {len(writer)}")

    print()
    print("Adding one shared entry (outcome='shared', query=label) ...")
    written = writer.add_entry(
        query=args.label,
        outcome="shared",
        items=[DEFAULT_ITEM],
    )
    print(
        f"  wrote: query={written.query!r} outcome={written.outcome} "
        f"items={len(written.items)} created_at={written.created_at}"
    )
    print(f"  writer size now: {len(writer)}")

    print()
    print("Re-loading the store from disk to exercise _coerce_outcome('shared') ...")
    reader = MemoryStore.load(args.shared_path, user_id="shared", embedder=embedder)
    print(f"  reader size: {len(reader)}")
    if len(reader) != 1:
        raise SystemExit(f"Expected exactly 1 reloaded entry, got {len(reader)}.")

    reloaded = reader.all()[0]
    print(
        f"  reloaded: user_id={reloaded.user_id} outcome={reloaded.outcome} "
        f"query={reloaded.query!r}"
    )
    if reloaded.outcome != "shared":
        raise SystemExit(f"Outcome did not round-trip as 'shared'; got {reloaded.outcome!r}.")

    print()
    print(f"Searching for top-1 entry against query={args.label!r} ...")
    hits = reader.search(args.label, k=1)
    print(f"  search returned {len(hits)} hit(s)")
    if not hits:
        raise SystemExit("Search returned no hits; cosine retrieval failed.")
    top = hits[0]
    print(
        f"  top hit: query={top.query!r} outcome={top.outcome} "
        f"item_title={top.items[0].title!r}"
    )
    if top.query != args.label or top.outcome != "shared":
        raise SystemExit("Top-1 search hit does not match the written entry.")

    if not args.keep and args.shared_path.exists():
        args.shared_path.unlink()
        print()
        print(f"Removed temp shared-store JSONL at {args.shared_path}.")

    print()
    print("Round-trip OK: write -> reload -> search all succeed for outcome='shared'.")


if __name__ == "__main__":
    main()
