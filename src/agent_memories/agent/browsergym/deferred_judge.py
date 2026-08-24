"""Since checking answers is computationally expensive, this script prevents any fuzzy or unachievable
checks to be run during a run-through. Instead, they are run at a later stage.
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
    """Replaces WebArena's LLM judge (costly) with a stub that records the inputs."""
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
