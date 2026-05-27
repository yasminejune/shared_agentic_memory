"""Runnable entry point for the WP1.2 TAO loop with accessibility-tree observations.

Loads google.com, runs the Observe -> Think -> Act loop with a scripted
two-step plan (fill the search box, then click the search button), and
prints the final state. The loop stops after two rounds because the
scripted Think runs out of actions; the graph itself has no hard cap,
so swapping in the WP1.3 Mistral-driven Think will let it run as long
as the LLM keeps emitting actions.
"""

from __future__ import annotations

from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.nodes import make_think_scripted

WEBSITE = "https://www.amazon.co.uk"


def main() -> None:
    stub_actions = [
        {"type": "fill", "selector": "#twotabsearchtextbox", "value": "Paper"},
        {"type": "click", "selector": "#nav-search-submit-button"},
    ]
    state = new_state(aim="Find amazon results for paper")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        try:
            page = browser.new_page()
            page.goto(WEBSITE)
            graph = build_graph(page, make_think_scripted(stub_actions))
            result = graph.invoke(state)
            print(result)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
