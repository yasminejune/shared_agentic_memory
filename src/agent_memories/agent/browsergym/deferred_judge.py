"""Deferred WebArena LLM judge — capture inputs, score offline.

Patches ``webarena.evaluation_harness.evaluators.llm_fuzzy_match`` and
``.llm_ua_match`` with stubs that return ``1.0`` and record the exact
``(pred, reference, intent)`` triple the harness passed. The live
reward is therefore the deterministic factor only; the real judge
verdict is computed in a separate offline pass via
``scripts/webarena/score_deferred_judge.py``.

The stubs patch the ``evaluators`` module attributes (not
``helper_functions``), because ``evaluators.py`` does
``from .helper_functions import llm_fuzzy_match, llm_ua_match``
at import time, binding those names in its own namespace.

Call :func:`install` after :func:`prepare_webarena` — importing the
harness triggers ``webarena/browser_env/env_config.py``'s assertion
that requires the bare ``SHOPPING``/``REDDIT``/… env vars, which only
exist after ``WebArenaInstance.__init__`` copies them from ``WA_*``.
"""

from __future__ import annotations

from typing import Any

_buffer: list[dict[str, Any]] = []
_installed: bool = False


def _stub_fuzzy_match(pred: str, reference: str, question: str) -> float:
    _buffer.append(
        {
            "kind": "fuzzy",
            "pred": pred,
            "reference": reference,
            "intent": question,
        }
    )
    return 1.0


def _stub_ua_match(pred: str, reference: str, question: str) -> float:
    _buffer.append(
        {
            "kind": "ua",
            "pred": pred,
            "reference": reference,
            "intent": question,
        }
    )
    return 1.0


def install() -> None:
    """Patch the evaluators module to defer LLM judge calls. Idempotent."""
    global _installed
    if _installed:
        return
    from webarena.evaluation_harness import evaluators

    evaluators.llm_fuzzy_match = _stub_fuzzy_match
    evaluators.llm_ua_match = _stub_ua_match
    _installed = True


def reset() -> None:
    """Clear the call buffer (call before each ``env.step``)."""
    _buffer.clear()


def calls() -> list[dict[str, Any]]:
    """Return a copy of the recorded judge calls since last :func:`reset`."""
    return list(_buffer)
