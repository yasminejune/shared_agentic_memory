"""Runnable entry point for the WP1.3 + WP1.4 Observe-Think-Act loop.

Drives one TAO session against a real Playwright Chromium page. Think
is LLM-driven (Qwen via local Ollama by default, Mistral via cloud API
optionally). The loop terminates on the LLM's ``stop`` action or when
``--max-steps`` is reached, whichever comes first -- WP1.3 covers the
single-action grammar, WP1.4 covers the multi-step termination, and
both ship in this one runner.

Terminal output is intentionally three trace prefixes only:
``[Observe]:``, ``[Think]:``, ``[Act]:``. The trajectory itself lives
in ``state['history']`` and is returned from :func:`main` so the
WP1.5+ memory pipeline can consume it in-process.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.nodes import make_think
from agent_memories.agent.state import AgentState
from agent_memories.services.mistral_client import MistralClient
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"
MISTRAL_MODEL = "mistral-small-latest"

DEFAULT_WEBSITE = "https://www.amazon.co.uk"
DEFAULT_AIM = "Find amazon results for paper"
DEFAULT_MAX_STEPS = 30


def _build_client(name: str) -> ChatClient:
    if name == "qwen":
        return OllamaClient(model=QWEN_MODEL)
    return MistralClient(model=MISTRAL_MODEL)


def main(argv: list[str] | None = None) -> AgentState:
    """Drive one Think-Act-Observe session and return the terminal state.

    Returns the final ``AgentState`` (the same dict ``graph.invoke``
    returns) so a higher-level memory-extraction pipeline can read
    ``result['history']`` directly. The browser is always closed via
    the ``finally`` block, including when a fatal chat exception
    propagates.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    parser.add_argument("--url", default=DEFAULT_WEBSITE)
    parser.add_argument("--aim", default=DEFAULT_AIM)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    client = _build_client(args.model)
    state = new_state(aim=args.aim)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        try:
            page = browser.new_page()
            page.goto(args.url)
            graph = build_graph(page, make_think(client), max_steps=args.max_steps)
            return graph.invoke(state)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
