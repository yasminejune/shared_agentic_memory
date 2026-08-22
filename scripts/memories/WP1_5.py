"""Private ReasoningBank loop: Observe-Think-Act plus memory write.

Same driver as scripts/langgraph/WP1_3.py, with three extras:

1. Per-user JSONL at data/memories/<user_id>.jsonl, loaded or created
   on startup.
2. Top-k memories for the aim go into state['memories'] before
   graph.invoke. Empty store means the agent runs without them.
3. After the loop, MemoryPipeline judges the trajectory and distils
   a private memory. The last observation's ARIA YAML is passed as
   final_state so the judge sees the page the agent left on.

Returns the final AgentState so a caller can read history and the
updated store without scraping stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.nodes import (
    DEFAULT_STUCK_THRESHOLD,
    OBSERVATION_CHAR_BUDGET,
    make_think,
)
from agent_memories.agent.state import AgentState
from agent_memories.config import DEFAULT_MEMORY_DIR, DEFAULT_USER_ID, load_random_seed
from agent_memories.memory import Embedder, MemoryPipeline, MemoryStore
from agent_memories.memory.store import MemoryEntry
from agent_memories.services.mistral_client import MistralClient
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"
MISTRAL_MODEL = "mistral-small-latest"

DEFAULT_WEBSITE = "https://www.amazon.co.uk"
DEFAULT_AIM = "Find amazon results for paper"
DEFAULT_MAX_STEPS = 10
DEFAULT_K = 1


def _build_client(name: str, seed: int) -> ChatClient:
    if name == "qwen":
        return OllamaClient(model=QWEN_MODEL, seed=seed)
    return MistralClient(model=MISTRAL_MODEL)


def _flatten_entries_for_think(entries: list[MemoryEntry]) -> list[dict[str, str]]:
    """Flatten retrieved entries to the ``{title, content}`` list Think expects.

    Default ``k=1`` is one trajectory's worth of lessons (usually 1-3
    items). Description is dropped: ReasoningBank's agent prompt only
    uses title and content (Appendix A.2).
    """
    flat: list[dict[str, str]] = []
    for entry in entries:
        for item in entry.items:
            flat.append({"title": item.title, "content": item.content})
    return flat


def _truncate_observation(tree_yaml: str, budget: int) -> str:
    """Cap tree_yaml to ``budget`` characters, with a visible cut marker.

    Same shape as ``nodes._truncate_observation``, so the judge sees
    what Think saw, not the full ARIA dump.
    """
    if budget <= 0 or len(tree_yaml) <= budget:
        return tree_yaml
    marker = "\n# ... <observation truncated to fit context window> ..."
    head = tree_yaml[: max(0, budget - len(marker))]
    return head + marker


def main(argv: list[str] | None = None) -> AgentState:
    """Run one memory-augmented TAO session; return the final state."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    parser.add_argument("--url", default=DEFAULT_WEBSITE)
    parser.add_argument("--aim", default=DEFAULT_AIM)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="How many memory entries to inject into the Think prompt (default: 1).",
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
    print(f"[Memory] Store at {store_path} has {len(store)} entries.", flush=True)

    retrieved = store.search(args.aim, k=args.k)
    flat_memories = _flatten_entries_for_think(retrieved)
    state = new_state(aim=args.aim)
    state["memories"] = flat_memories
    print(
        f"[Memory] Retrieved {len(retrieved)} entries ({len(flat_memories)} item(s)) for aim "
        f"(user_id={args.user_id}, k={args.k})",
        flush=True,
    )

    client = _build_client(args.model, load_random_seed())

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

    final_state_yaml = _truncate_observation(
        result.get("observation", {}).get("tree_yaml", ""),
        OBSERVATION_CHAR_BUDGET,
    )
    pipeline = MemoryPipeline(client=client, store=store)
    created = pipeline.create_from_run(result, final_state=final_state_yaml)
    if created is None:
        print("[Memory] No new memory created (0 items parsed from extractor).", flush=True)
    else:
        print(
            f"[Memory] Created entry for query={created.query!r} "
            f"outcome={created.outcome} items={len(created.items)}",
            flush=True,
        )

    return result


if __name__ == "__main__":
    main()
