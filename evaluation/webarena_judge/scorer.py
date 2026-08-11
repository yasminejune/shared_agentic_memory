"""Score deferred WebArena judge calls using Mistral.

Patches ``webarena.evaluation_harness.helper_functions.generate_from_openai_chat_completion``
to route through :class:`MistralClient` so the harness's own
``llm_fuzzy_match`` and ``llm_ua_match`` execute with their verbatim
prompts and verdict rules but against a pinned Mistral model instead
of ``gpt-4-1106-preview``.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from agent_memories.services.mistral_client import MistralClient

JUDGE_MODEL = "mistral-large-2512"
JUDGE_MAX_TOKENS = 768
JUDGE_TEMPERATURE = 0.0

JUDGE_SCORES_COLUMNS = [
    "task_id",
    "judge_model",
    "n_calls",
    "verdicts",
    "judge_product",
    "deferred_reward",
    "final_reward",
    "final_success",
    "status",
    "scored_at",
]


def install_mistral_judge(client: MistralClient) -> None:
    """Patch the harness to route LLM calls through the given MistralClient."""
    from webarena.evaluation_harness import helper_functions

    def _mistral_chat_completion(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
    ) -> str:
        system_parts = []
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_parts.append(msg["content"])
        return client.chat(
            system="\n".join(system_parts) if system_parts else "You are a helpful assistant",
            user="\n".join(user_parts),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )

    helper_functions.generate_from_openai_chat_completion = _mistral_chat_completion  # type: ignore[assignment]


def score_single_call(
    call: dict[str, Any],
) -> tuple[float, str]:
    """Run one deferred judge call through the patched harness.

    Returns ``(verdict, status)`` where verdict is 0.0 or 1.0 on
    success, and status is ``"scored"``, ``"unparseable"``, or
    ``"judge_error: <detail>"``.
    """
    from webarena.evaluation_harness.helper_functions import (
        llm_fuzzy_match,
        llm_ua_match,
    )

    kind = call["kind"]
    pred = call["pred"]
    reference = call["reference"]
    intent = call["intent"]

    try:
        if kind == "ua":
            verdict = llm_ua_match(pred, reference, intent)
        else:
            verdict = llm_fuzzy_match(pred, reference, intent)
        return verdict, "scored"
    except AssertionError:
        return 0.0, "unparseable"
    except Exception as exc:
        return 0.0, f"judge_error: {type(exc).__name__}: {exc}"


def score_task(
    calls: list[dict[str, Any]],
    deferred_reward: float,
    *,
    max_retries: int = 3,
    sleep_between: float = 1.0,
) -> tuple[list[float], float, float, bool, str]:
    """Score all deferred calls for one task.

    Returns ``(verdicts, judge_product, final_reward, final_success, status)``.
    """
    verdicts: list[float] = []
    worst_status = "scored"

    for call in calls:
        verdict = 0.0
        status = "judge_error: exhausted retries"
        for attempt in range(1, max_retries + 1):
            verdict, status = score_single_call(call)
            if status == "scored" or status == "unparseable":
                break
            delay = sleep_between * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(delay)

        verdicts.append(verdict)
        if status != "scored" and worst_status == "scored":
            worst_status = status
        elif status == "unparseable" and "judge_error" not in worst_status:
            worst_status = status

    judge_product = math.prod(verdicts) if verdicts else 1.0
    final_reward = deferred_reward * judge_product
    final_success = final_reward > 0
    return verdicts, judge_product, final_reward, final_success, worst_status
