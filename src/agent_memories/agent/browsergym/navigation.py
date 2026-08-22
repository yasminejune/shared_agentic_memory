"""goto and go_home closed under the local WebArena instance."""

from __future__ import annotations

from typing import Any

# BrowserGym injects the live Playwright page into this name at exec time.
page: Any = None


def goto(url: str) -> None:
    """
    Navigate to a url on the local WebArena instance. Off-instance hosts
    are rewritten or sent to the homepage.

    Examples:
        goto('http://localhost:9999/f/books')
    """
    from agent_memories.agent.browsergym.urls import instance_urls, resolve_local_url

    site_urls, home_url = instance_urls()
    page.goto(resolve_local_url(url, site_urls, home_url))


def go_home() -> None:
    """
    Navigate to the WebArena homepage. Use this for cross-site tasks.

    Examples:
        go_home()
    """
    from agent_memories.agent.browsergym.urls import instance_urls

    _, home_url = instance_urls()
    page.goto(home_url)
