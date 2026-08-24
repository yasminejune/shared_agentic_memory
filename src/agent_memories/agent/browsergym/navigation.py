"""Added functions for browsergym: goto and go_home given the local WebArena instance."""

from __future__ import annotations

from typing import Any

# BrowserGym injects the live Playwright page into this name at exec time.
page: Any = None


# This makes sure that the page doesn't go to the real website but the local copy of it
def goto(url: str) -> None:
    """
    Navigate to a url on the local WebArena instance.

    Examples:
        goto('http://localhost:9999/f/books')
    """
    from agent_memories.agent.browsergym.urls import instance_urls, resolve_local_url

    site_urls, home_url = instance_urls()
    page.goto(resolve_local_url(url, site_urls, home_url))


def go_home() -> None:
    """
    Navigate to the WebArena homepage. Used for cross-site tasks.

    Examples:
        go_home()
    """
    from agent_memories.agent.browsergym.urls import instance_urls

    _, home_url = instance_urls()
    page.goto(home_url)
