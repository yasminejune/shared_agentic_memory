"""WP1.1 TAO scaffold. One scripted click on the fixture page.

Loads tests/data/single_step.html over file://, stubs Think, runs one
cycle, prints the state. Pre-BrowserGym; the nodes themselves live in
src/agent_memories/agent/.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state
from agent_memories.agent.playwright.nodes import make_think_scripted

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "data" / "single_step.html"


def main() -> None:
    stub_actions = [{"type": "click", "selector": "#go"}]
    state = new_state(aim="Click the Go button on the fixture page.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(FIXTURE.as_uri())
            graph = build_graph(page, make_think_scripted(stub_actions))
            result = graph.invoke(state)
            print(result)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
