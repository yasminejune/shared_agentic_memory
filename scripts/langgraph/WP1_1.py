"""Runnable entry point for the WP1.1 Think-Act-Observe scaffold.

Loads the controlled fixture page over ``file://``, builds the graph
with a stubbed Think node, runs one cycle, prints the final state. The
real logic lives in ``src/agent_memories/agent/``; this file only
wires it to a browser and a fixture path.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state

REPO_ROOT = Path(__file__).resolve().parents[2]  # Name of this project route
FIXTURE = REPO_ROOT / "tests" / "data" / "single_step.html"  # Path to fake website


def main() -> None:
    stub_action = {"type": "click", "selector": "#go"}  # Predefined action to click the Go button
    state = new_state(
        aim="Click the Go button on the fixture page."
    )  # Aim is to click the Go button on the fake website

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Launch a headless browser
        try:
            page = browser.new_page()
            page.goto(FIXTURE.as_uri())
            graph = build_graph(page, stub_action)
            result = graph.invoke(state)
            print(result)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
