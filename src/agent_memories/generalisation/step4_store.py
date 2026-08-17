"""Step 4: title and description from DP content, then shared MemoryStore write.

Off the privacy path. The stored embedding is over the DP label
(dissertation §4.1.3).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_memories.config import load_random_seed
from agent_memories.generalisation.io import (
    DEFAULT_SHARED_STORE,
    QWEN_MODEL,
    add_io_arguments,
    contents_path,
    read_jsonl,
    resolve_io_paths,
)
from agent_memories.generalisation.post_processing import title_and_description
from agent_memories.memory import Embedder, MemoryItem, MemoryStore
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient


def run_store_write(
    records: Sequence[dict[str, Any]],
    *,
    shared_path: Path,
    client: ChatClient,
    embedder: Embedder,
) -> list[dict[str, str]]:
    """Post-process each content record and append to the shared store."""
    store = MemoryStore.load(shared_path, user_id="shared", embedder=embedder)
    written: list[dict[str, str]] = []
    for record in records:
        label = str(record["label"])
        content = str(record["content"])
        title, description = title_and_description(content, label, client=client)
        item = MemoryItem(title=title, description=description, content=content)
        store.add_entry(query=label, outcome="shared", items=[item])
        written.append({"label": label, "title": title, "description": description})
    return written


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

    _, work_dir = resolve_io_paths(args)
    records = read_jsonl(contents_path(work_dir))
    if not records:
        print(f"[Step 4] No content records in {contents_path(work_dir)}; nothing to write.")
        return

    seed = load_random_seed()
    client = OllamaClient(model=QWEN_MODEL, seed=seed)
    embedder = Embedder()
    written = run_store_write(
        records,
        shared_path=args.shared_store,
        client=client,
        embedder=embedder,
    )
    for row in written:
        print(f"[Step 4] label={row['label']!r} title={row['title']!r}")
    print(f"[Step 4] Wrote {len(written)} shared entries to {args.shared_store}")


if __name__ == "__main__":
    main(sys.argv[1:])
