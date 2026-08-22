"""WP1.2 TAO loop with accessibility-tree observations, still scripted.

Amazon this time, not the fixture. Fill the search box, click search,
print the final state. Stops when the stub Think runs out of actions;
the graph has no step cap yet. WP1.3 puts an LLM behind Think.
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
