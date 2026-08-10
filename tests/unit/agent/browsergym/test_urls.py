"""Unit tests for sandboxed WebArena URL resolution."""

from __future__ import annotations

import pytest

from agent_memories.agent.browsergym.urls import instance_urls, resolve_local_url

SITE_URLS = {
    "reddit": "http://localhost:9999",
    "shopping": "http://localhost:7770",
    "shopping_admin": "http://localhost:7780/admin",
    "gitlab": "http://localhost:8023",
    "wikipedia": "http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    "map": "http://localhost:3000",
}
HOME_URL = "http://localhost:4399"


def test_on_instance_passthrough() -> None:
    url = "http://localhost:9999/f/books"
    assert resolve_local_url(url, SITE_URLS, HOME_URL) == url


def test_www_and_https_normalisation() -> None:
    assert (
        resolve_local_url("https://www.reddit.com/r/books/", SITE_URLS, HOME_URL)
        == "http://localhost:9999/r/books/"
    )


def test_preserves_path_and_query() -> None:
    assert (
        resolve_local_url(
            "https://gitlab.com/byteblaze/a11y-testing?tab=issues",
            SITE_URLS,
            HOME_URL,
        )
        == "http://localhost:8023/byteblaze/a11y-testing?tab=issues"
    )


@pytest.mark.parametrize(
    ("public", "expected"),
    [
        ("https://github.com/org/repo", "http://localhost:8023/org/repo"),
        ("https://www.amazon.com/dp/B00", "http://localhost:7770/dp/B00"),
        ("https://www.ebay.com/itm/1", "http://localhost:7770/itm/1"),
        ("https://onestopshopping.com/", "http://localhost:7770/"),
        ("https://en.wikipedia.org/wiki/X", "http://localhost:8888/wiki/X"),
        ("https://about.gitlab.com/", "http://localhost:8023/"),
        ("https://docs.gitlab.com/ee/", "http://localhost:8023/ee/"),
        ("http://luma.com/admin/dashboard", "http://localhost:7780/admin/dashboard"),
        ("http://onestopmarket.com/catalog", "http://localhost:7770/catalog"),
        (
            "https://www.openstreetmap.org/search?query=cmu",
            "http://localhost:3000/search?query=cmu",
        ),
    ],
)
def test_aliases_and_canonical_hosts(public: str, expected: str) -> None:
    assert resolve_local_url(public, SITE_URLS, HOME_URL) == expected


def test_unmappable_host_falls_back_to_homepage() -> None:
    assert resolve_local_url("https://www.bestbuy.com/", SITE_URLS, HOME_URL) == HOME_URL
    assert resolve_local_url("https://www.example.com", SITE_URLS, HOME_URL) == HOME_URL


def test_homepage_public_host_maps_to_home() -> None:
    assert resolve_local_url("http://homepage.com/", SITE_URLS, HOME_URL) == HOME_URL


def test_missing_netloc_falls_back_to_homepage() -> None:
    assert resolve_local_url("/relative/path", SITE_URLS, HOME_URL) == HOME_URL


def test_instance_urls_reads_wa_env(monkeypatch: pytest.MonkeyPatch) -> None:
    instance_urls.cache_clear()
    monkeypatch.setenv("WA_REDDIT", "http://localhost:9999")
    monkeypatch.setenv("WA_SHOPPING", "http://localhost:7770")
    monkeypatch.setenv("WA_SHOPPING_ADMIN", "http://localhost:7780/admin")
    monkeypatch.setenv("WA_GITLAB", "http://localhost:8023")
    monkeypatch.setenv("WA_WIKIPEDIA", "http://localhost:8888/wiki")
    monkeypatch.setenv("WA_MAP", "http://localhost:3000")
    monkeypatch.setenv("WA_HOMEPAGE", "http://localhost:4399")

    site_urls, home_url = instance_urls()
    assert home_url == "http://localhost:4399"
    assert site_urls["reddit"] == "http://localhost:9999"
    assert site_urls["shopping_admin"] == "http://localhost:7780/admin"
    instance_urls.cache_clear()


def test_instance_urls_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    instance_urls.cache_clear()
    for key in (
        "WA_REDDIT",
        "WA_SHOPPING",
        "WA_SHOPPING_ADMIN",
        "WA_GITLAB",
        "WA_WIKIPEDIA",
        "WA_MAP",
        "WA_HOMEPAGE",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Missing WebArena environment variables"):
        instance_urls()
    instance_urls.cache_clear()
