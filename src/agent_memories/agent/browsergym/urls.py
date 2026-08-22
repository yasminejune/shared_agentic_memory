"""Rewrite public WebArena hosts onto the local instance."""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import urlparse, urlunparse

# Public WebArena hosts plus aliases from the no-memory baseline.
SITE_HOSTS: dict[str, str] = {
    "reddit.com": "reddit",
    "onestopmarket.com": "shopping",
    "luma.com": "shopping_admin",
    "gitlab.com": "gitlab",
    "wikipedia.org": "wikipedia",
    "openstreetmap.org": "map",
    "homepage.com": "home",
    # Aliases the baseline agent actually emitted.
    "github.com": "gitlab",
    "amazon.com": "shopping",
    "ebay.com": "shopping",
    "onestopshopping.com": "shopping",
    "en.wikipedia.org": "wikipedia",
    "about.gitlab.com": "gitlab",
    "docs.gitlab.com": "gitlab",
}

_SITE_ENV_KEYS: dict[str, str] = {
    "reddit": "WA_REDDIT",
    "shopping": "WA_SHOPPING",
    "shopping_admin": "WA_SHOPPING_ADMIN",
    "gitlab": "WA_GITLAB",
    "wikipedia": "WA_WIKIPEDIA",
    "map": "WA_MAP",
}


def _normalise_host(netloc: str) -> str:
    host = netloc.lower().split("@")[-1]
    if host.startswith("[") and "]" in host:
        # IPv6 literal: keep as-is up to the closing bracket / port.
        return host
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _origin_netloc(url: str) -> str:
    return urlparse(url).netloc.lower()


def _is_on_instance(netloc: str, site_urls: dict[str, str], home_url: str) -> bool:
    target = netloc.lower()
    authorised = {_origin_netloc(u) for u in site_urls.values()}
    authorised.add(_origin_netloc(home_url))
    return target in authorised


def resolve_local_url(url: str, site_urls: dict[str, str], home_url: str) -> str:
    """Map a navigation target onto the local WebArena instance.

    URLs already on the instance pass through. Hosts in SITE_HOSTS keep path
    and query on the matching site origin. Anything else becomes home_url.
    """
    parsed = urlparse(url)
    if not parsed.netloc:
        return home_url
    if _is_on_instance(parsed.netloc, site_urls, home_url):
        return url

    host = _normalise_host(parsed.netloc)
    site_key = SITE_HOSTS.get(host)
    if site_key == "home":
        return home_url
    if site_key is None or site_key not in site_urls:
        return home_url

    local = urlparse(site_urls[site_key])
    return urlunparse(
        (
            local.scheme or "http",
            local.netloc,
            parsed.path or "/",
            "",
            parsed.query,
            "",
        )
    )


@lru_cache(maxsize=1)
def instance_urls() -> tuple[dict[str, str], str]:
    """Return (site_urls, home_url) from the WA_* environment variables."""
    missing = [env for env in _SITE_ENV_KEYS.values() if not os.environ.get(env)]
    if not os.environ.get("WA_HOMEPAGE"):
        missing.append("WA_HOMEPAGE")
    if missing:
        raise RuntimeError(
            "Missing WebArena environment variables for URL resolution: " + ", ".join(missing)
        )
    site_urls = {key: os.environ[env] for key, env in _SITE_ENV_KEYS.items()}
    return site_urls, os.environ["WA_HOMEPAGE"]
