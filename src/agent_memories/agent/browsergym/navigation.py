"""Custom BrowserGym navigation actions closed under the WebArena instance."""

# Placeholder for BrowserGym's exec globals (see browsergym.core.action.functions).
page = None  # type: ignore[assignment]


def goto(url: str):
    """
    Navigate to a url within the WebArena instance. Urls outside the instance
    are redirected to the WebArena homepage.

    Examples:
        goto('http://localhost:9999/f/books')
    """
    from agent_memories.agent.browsergym.urls import instance_urls, resolve_local_url

    site_urls, home_url = instance_urls()
    page.goto(resolve_local_url(url, site_urls, home_url))


def go_home():
    """
    Navigate to the WebArena homepage, which lists every website available in
    the environment. Use this to reach a different website.

    Examples:
        go_home()
    """
    from agent_memories.agent.browsergym.urls import instance_urls

    _, home_url = instance_urls()
    page.goto(home_url)
