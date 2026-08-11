"""Custom BrowserGym navigation actions closed under the WebArena instance."""

from __future__ import annotations

from typing import Any

# Placeholder for BrowserGym's exec globals (see browsergym.core.action.functions).
# At action time BrowserGym injects the live Playwright page into this name.
page: Any = None


def goto(url: str) -> None:
    """
    Navigate to a url within the WebArena instance. Urls outside the instance
    are redirected to the WebArena homepage.

    Examples:
        goto('http://localhost:9999/f/books')
    """
    from agent_memories.agent.browsergym.urls import instance_urls, resolve_local_url

    site_urls, home_url = instance_urls()
    page.goto(resolve_local_url(url, site_urls, home_url))


def go_home() -> None:
    """
    Navigate to the WebArena homepage, which lists every website available in
    the environment. Use this to reach a different website.

    Examples:
        go_home()
    """
    from agent_memories.agent.browsergym.urls import instance_urls

    _, home_url = instance_urls()
    page.goto(home_url)
