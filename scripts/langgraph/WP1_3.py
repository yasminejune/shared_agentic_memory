"""WP1.3 / WP1.4 Observe-Think-Act on a live Chromium page.

Think is an LLM (local Qwen via Ollama, or Mistral on the cloud).
Stops on the model's ``stop`` action or ``--max-steps``. WP1.3 is the
single-action grammar; WP1.4 is the multi-step loop. Both live here.

Stdout is three prefixes only: ``[Observe]:``, ``[Think]:``, ``[Act]:``.
The trajectory sits in ``state['history']`` and ``main`` returns it so
a later memory pipeline can consume a finished run in-process.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.nodes import DEFAULT_STUCK_THRESHOLD, make_think
from agent_memories.agent.state import AgentState
from agent_memories.config import load_random_seed
from agent_memories.services.mistral_client import MistralClient
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"
MISTRAL_MODEL = "mistral-small-latest"

DEFAULT_WEBSITE = "https://www.amazon.co.uk"
DEFAULT_AIM = "Find amazon results for paper"
DEFAULT_MAX_STEPS = 30


def _build_client(name: str, seed: int) -> ChatClient:
    if name == "qwen":
        return OllamaClient(model=QWEN_MODEL, seed=seed)
    return MistralClient(model=MISTRAL_MODEL)


def main(argv: list[str] | None = None) -> AgentState:
    """Run one TAO session and return the terminal AgentState.

    Same dict ``graph.invoke`` returns, so a caller can read
    ``result['history']`` without scraping stdout. The ``finally``
    closes the browser even if the chat client blows up.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    parser.add_argument("--url", default=DEFAULT_WEBSITE)
    parser.add_argument("--aim", default=DEFAULT_AIM)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headless", action="store_true")
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
    client = _build_client(args.model, load_random_seed())
    state = new_state(aim=args.aim)
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
            return graph.invoke(state)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
