"""Accessibility-tree observation for the Observe node.

We expose a compact semantic view of the current page to the LLM via
Playwright's ``page.aria_snapshot(mode="ai")``, which returns a YAML
representation of the accessibility tree with native element references
of the form ``[ref=eN]``. WP1.3's action parser resolves those refs
back to actionable handles via ``page.locator(f"aria-ref={ref}")`` --
no custom target registry is maintained here.

Refs are ephemeral: the snapshot is regenerated on every Observe step,
so an ``eN`` from one observation must not be reused after the next
action.
"""

from __future__ import annotations

from playwright.sync_api import Page


def format_aria_snapshot(page: Page) -> str:
    """Return ``page``'s ARIA snapshot as YAML with element references.

    Uses ``mode="ai"`` so each interactive node carries an ``[ref=eN]``
    that WP1.3 can feed straight into ``page.locator("aria-ref=eN")``.
    Falsy returns (``None`` or empty string) are coerced to ``""`` so
    downstream callers always receive a string.
    """
    return page.aria_snapshot(mode="ai") or ""
