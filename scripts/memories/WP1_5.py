"""Runnable entry point for the WP1.5 memory-augmented Observe-Think-Act loop.

Mirrors ``scripts/langgraph/WP1_3.py`` exactly except for three additions:

1. **Per-user memory store on disk.** A JSONL file at
   ``data/memories/<user_id>.jsonl`` is loaded (or created lazily) on
   startup.
2. **Optional seed file.** ``--seed-memories <path>`` reads a JSON list
   of strings and appends them to the user's store before the run, so
   the "manually seeded memories" controlled-task test from tasks.md
   reduces to a single command-line invocation.
3. **Memory injection + creation.** The top-``k`` memories most similar
   to the aim are read once into ``state['memories']`` before
   ``graph.invoke``; after the run completes, the curator LLM is asked
   whether the trajectory is worth distilling into a new memory.

The script returns the final :class:`AgentState` so a higher-level
WP1.6 / WP2 driver can consume ``state['history']`` and the updated
store directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.nodes import DEFAULT_STUCK_THRESHOLD, make_think
from agent_memories.agent.state import AgentState
from agent_memories.config import DEFAULT_MEMORY_DIR, DEFAULT_USER_ID, default_seed_path
from agent_memories.memory import Embedder, MemoryPipeline, MemoryStore
from agent_memories.services.mistral_client import MistralClient
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"
MISTRAL_MODEL = "mistral-small-latest"

DEFAULT_WEBSITE = "https://www.amazon.co.uk"
DEFAULT_AIM = "Find amazon results for paper"
DEFAULT_MAX_STEPS = 10
DEFAULT_K = 1


def _build_client(name: str) -> ChatClient:
    if name == "qwen":
        return OllamaClient(model=QWEN_MODEL)
    return MistralClient(model=MISTRAL_MODEL)


def _load_seed_memories(path: Path) -> list[str]:
    """Read a JSON file of seed memories.

    Accepted shapes:

    * ``["memory text", ...]`` -- a plain list of strings.
    * ``[{"text": "memory text"}, ...]`` -- a list of objects with a
      ``text`` field; other fields are ignored (forward-compat with
      WP1.6's richer schema).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Seed file {path} must contain a JSON list, got {type(raw).__name__}")
    texts: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            texts.append(entry)
        elif isinstance(entry, dict) and "text" in entry:
            texts.append(str(entry["text"]))
        else:
            raise ValueError(
                f"Seed file {path} entries must be strings or objects with 'text'; got {entry!r}"
            )
    return texts


def _seed_store_if_empty(store: MemoryStore, seed_path: Path | None) -> int:
    """Seed an empty store from ``seed_path``; return the number seeded.

    Re-running the script with the same ``--seed-memories`` file should
    not double-seed, so seeding is skipped when the store is already
    non-empty. Delete the JSONL file to re-seed from scratch.
    """
    if seed_path is None or len(store) > 0:
        return 0
    texts = _load_seed_memories(seed_path)
    store.add_many(texts)
    return len(texts)


def main(argv: list[str] | None = None) -> AgentState:
    """Drive one memory-augmented TAO session and return the final state."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    parser.add_argument("--url", default=DEFAULT_WEBSITE)
    parser.add_argument("--aim", default=DEFAULT_AIM)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument(
        "--seed-memories",
        type=Path,
        default=None,
        help="Optional JSON file of seed memories to load into an empty store.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="How many memories to inject into the Think prompt (default: 1).",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=DEFAULT_MEMORY_DIR,
        help="Directory containing <user_id>.jsonl files.",
    )
    parser.add_argument(
        "--stuck-threshold",
        type=int,
        default=DEFAULT_STUCK_THRESHOLD,
        help=(
            "Abort the run as 'stuck' when this many consecutive Think turns "
            "produce identical replies (default: 5)."
        ),
    )
    args = parser.parse_args(argv)

    load_dotenv()

    embedder = Embedder()
    store_path = args.memory_dir / f"{args.user_id}.jsonl"
    store = MemoryStore.load(store_path, user_id=args.user_id, embedder=embedder)
    print(f"[Memory] Store at {store_path} has {len(store)} memories.", flush=True)

    # Fall back to the canonical seed fixture for this user when the caller did
    # not pass --seed-memories. Keeps "first run" reproducible without
    # requiring the flag on every invocation; idempotent on subsequent runs
    # because _seed_store_if_empty no-ops once the store is populated.
    seed_path = args.seed_memories
    if seed_path is None:
        candidate = default_seed_path(args.user_id)
        if candidate.exists():
            seed_path = candidate

    seeded = _seed_store_if_empty(store, seed_path)
    if seeded:
        print(f"[Memory] Seeded {seeded} memories into {store_path}", flush=True)

    retrieved = store.search(args.aim, k=args.k)
    state = new_state(aim=args.aim)
    state["memories"] = [m.text for m in retrieved]
    print(
        f"[Memory] Retrieved {len(retrieved)} memories for aim "
        f"(user_id={args.user_id}, k={args.k})",
        flush=True,
    )

    client = _build_client(args.model)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        try:
            page = browser.new_page()
            page.goto(args.url)
            graph = build_graph(
                page,
                make_think(client, stuck_threshold=args.stuck_threshold),
                max_steps=args.max_steps,
            )
            result: AgentState = graph.invoke(state)
        finally:
            browser.close()

    pipeline = MemoryPipeline(client=client, store=store)
    created = pipeline.create_from_run(result)
    if created is None:
        print("[Memory] No new memory created (curator returned 'None').", flush=True)
    else:
        print(f"[Memory] Created new memory: {created.text!r}", flush=True)

    return result


if __name__ == "__main__":
    main()
