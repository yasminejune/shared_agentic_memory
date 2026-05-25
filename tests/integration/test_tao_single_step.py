"""Integration test for the WP1.1 TAO scaffold extended with the WP1.2 loop.

Launches a real Chromium via Playwright, loads the local fixture page
over ``file://``, runs the loop with a single scripted action and checks
that:

  * the graph reaches the terminal state in one round (one action -> done);
  * the stubbed action was carried out (the success div is visible);
  * the Observe node produced an ARIA-snapshot YAML containing the
    fixture's button and the ``[ref=eN]`` references WP1.3 will resolve.

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
    stub_actions = [{"type": "click", "selector": "#go"}]
    state = new_state(aim="Click the Go button on the fixture page.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(FIXTURE.as_uri())

            graph = build_graph(page, stub_actions)
            result = graph.invoke(state)

            assert result["done"] is True
            assert result["step"] == 1
            assert result["action"] == stub_actions[0]
            assert result["thought"].startswith("Scripted action")

            assert result["observation"]["title"] == "TAO Single Step Fixture"

            tree_yaml = result["observation"]["tree_yaml"]
            assert tree_yaml, "tree_yaml should be non-empty"
            assert "button" in tree_yaml
            assert "Go" in tree_yaml
            assert "[ref=e" in tree_yaml, "AI mode should embed [ref=eN] markers"

            assert page.locator("#success").is_visible()
        finally:
            browser.close()
