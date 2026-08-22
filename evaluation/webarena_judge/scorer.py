"""Score deferred WebArena LLM-judge calls with Mistral.

The live harness reward is the official score. A subset of tasks also
need fuzzy_match or ua_match; those were recorded during the run and
are scored here. The harness match functions keep their original prompts
but talk to a pinned Mistral model instead of gpt-4-1106-preview.
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
# Wait at least this long after a 429 so retries do not burn the next minute.
RATE_LIMIT_RETRY_SLEEP = 60.0

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
    """Patch the harness so judge calls go through this Mistral client."""
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

    helper_functions.generate_from_openai_chat_completion = _mistral_chat_completion


def score_single_call(
    call: dict[str, Any],
) -> tuple[float, str]:
    """Run one recorded fuzzy_match or ua_match call.

    Returns (verdict, status). Verdict is 0.0 or 1.0 when scored;
    status is "scored", "unparseable", or "judge_error: ...".
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


def _is_rate_limited(status: str) -> bool:
    lowered = status.lower()
    return "429" in status or "rate_limited" in lowered


def score_task(
    calls: list[dict[str, Any]],
    deferred_reward: float,
    *,
    max_retries: int = 3,
    sleep_between: float = 1.0,
) -> tuple[list[float], float, float, bool, str]:
    """Score every deferred judge call for one task.

    sleep_between is the gap after every API attempt. A 429 waits at least
    RATE_LIMIT_RETRY_SLEEP seconds so retries do not burn the minute window.
    """
    verdicts: list[float] = []
    worst_status = "scored"

    for call in calls:
        verdict = 0.0
        status = "judge_error: exhausted retries"
        for attempt in range(1, max_retries + 1):
            verdict, status = score_single_call(call)
            done = status == "scored" or status == "unparseable"
            will_retry = not done and attempt < max_retries

            if will_retry:
                base = sleep_between * (2 ** (attempt - 1))
                if _is_rate_limited(status):
                    delay = max(RATE_LIMIT_RETRY_SLEEP, base)
                else:
                    delay = base
                time.sleep(delay + random.uniform(0, 0.5))
            else:
                if sleep_between > 0:
                    time.sleep(sleep_between)
                if done:
                    break

        verdicts.append(verdict)
        if status != "scored" and worst_status == "scored":
            worst_status = status
        elif status == "unparseable" and "judge_error" not in worst_status:
            worst_status = status

    judge_product = math.prod(verdicts) if verdicts else 1.0
    final_reward = deferred_reward * judge_product
    final_success = final_reward > 0
    return verdicts, judge_product, final_reward, final_success, worst_status
