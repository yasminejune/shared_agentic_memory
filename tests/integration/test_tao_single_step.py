"""Integration test for the WP1.1 TAO scaffold.

Launches a real Chromium via Playwright, loads the local fixture page
over ``file://``, runs one cycle of the graph and checks that:

  * the graph reaches the terminal state in a single step;
  * the stubbed action was carried out (the success div is visible);
  * the Observe node captured a non-empty observation.

Marked ``integration`` because it launches a browser; it is excluded
from the default unit-test run via ``-m 'not integration'`` if needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from agent_memories.agent import build_graph, new_state

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "data" / "single_step.html"


@pytest.mark.integration
def test_single_step_clicks_button_and_reveals_success() -> None:
    stub_action = {"type": "click", "selector": "#go"}
    state = new_state(aim="Click the Go button on the fixture page.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(FIXTURE.as_uri())

            graph = build_graph(page, stub_action)
            result = graph.invoke(state)

            assert result["done"] is True
            assert result["step"] == 1
            assert result["action"] == stub_action
            assert result["thought"].startswith("Stubbed think")

            assert result["observation"]["title"] == "TAO Single Step Fixture"
            assert "TAO Single Step Fixture" in result["observation"]["body_text"]

            assert page.locator("#success").is_visible()
        finally:
            browser.close()
