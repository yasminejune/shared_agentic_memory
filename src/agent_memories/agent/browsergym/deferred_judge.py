"""Stub the WebArena LLM judge during a run and record its inputs.

fuzzy_match and ua_match return 1.0 so the harness product is not lost.
The real (pred, reference, intent) triples are written out and scored later.
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
    """Patch evaluators.llm_fuzzy_match / llm_ua_match. Call after prepare_webarena."""
    global _installed
    if _installed:
        return
    from webarena.evaluation_harness import evaluators

    evaluators.llm_fuzzy_match = _stub_fuzzy_match
    evaluators.llm_ua_match = _stub_ua_match
    _installed = True


def reset() -> None:
    """Clear the call buffer. Call before each env.step."""
    _buffer.clear()


def calls() -> list[dict[str, Any]]:
    """Copy of recorded (pred, reference, intent) triples since the last reset."""
    return list(_buffer)
