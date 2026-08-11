"""Unit tests for the deferred WebArena LLM judge infrastructure."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1. Shim tests
# ---------------------------------------------------------------------------

from agent_memories.agent.browsergym import deferred_judge


@pytest.fixture(autouse=True)
def _reset_shim():
    """Ensure a clean shim state for each test."""
    deferred_judge.reset()
    yield
    deferred_judge.reset()


@pytest.mark.unit
def test_stub_fuzzy_match_returns_1_and_records() -> None:
    result = deferred_judge._stub_fuzzy_match("pred_val", "ref_val", "intent_val")
    assert result == 1.0
    recorded = deferred_judge.calls()
    assert len(recorded) == 1
    assert recorded[0] == {
        "kind": "fuzzy",
        "pred": "pred_val",
        "reference": "ref_val",
        "intent": "intent_val",
    }


@pytest.mark.unit
def test_stub_ua_match_returns_1_and_records() -> None:
    result = deferred_judge._stub_ua_match("pred_val", "ref_val", "intent_val")
    assert result == 1.0
    recorded = deferred_judge.calls()
    assert len(recorded) == 1
    assert recorded[0]["kind"] == "ua"


@pytest.mark.unit
def test_reset_clears_buffer() -> None:
    deferred_judge._stub_fuzzy_match("a", "b", "c")
    assert len(deferred_judge.calls()) == 1
    deferred_judge.reset()
    assert len(deferred_judge.calls()) == 0


@pytest.mark.unit
def test_calls_returns_copy() -> None:
    deferred_judge._stub_fuzzy_match("a", "b", "c")
    copy = deferred_judge.calls()
    copy.append({"extra": True})
    assert len(deferred_judge.calls()) == 1


@pytest.mark.unit
def test_install_is_idempotent() -> None:
    import types

    fake_harness = types.ModuleType("webarena.evaluation_harness")
    fake_evaluators = types.ModuleType("webarena.evaluation_harness.evaluators")
    fake_evaluators.llm_fuzzy_match = lambda p, r, q: 0.0  # type: ignore
    fake_evaluators.llm_ua_match = lambda p, r, q: 0.0  # type: ignore
    fake_harness.evaluators = fake_evaluators  # type: ignore

    saved = deferred_judge._installed
    try:
        deferred_judge._installed = False
        with patch.dict("sys.modules", {
            "webarena.evaluation_harness": fake_harness,
            "webarena.evaluation_harness.evaluators": fake_evaluators,
        }):
            deferred_judge.install()
            assert fake_evaluators.llm_fuzzy_match is deferred_judge._stub_fuzzy_match
            assert fake_evaluators.llm_ua_match is deferred_judge._stub_ua_match
            deferred_judge.install()
            assert fake_evaluators.llm_fuzzy_match is deferred_judge._stub_fuzzy_match
    finally:
        deferred_judge._installed = saved


# ---------------------------------------------------------------------------
# 2. Scoring arithmetic tests
# ---------------------------------------------------------------------------

from evaluation.webarena_judge.scorer import score_task


class _FakeJudgeClient:
    """Chat client that returns predetermined verdicts."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def chat(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


@pytest.mark.unit
def test_score_task_single_correct() -> None:
    calls = [{"kind": "fuzzy", "pred": "3h30", "reference": "3h 30min", "intent": "How long?"}]

    def fake_score(call):
        return 1.0, "scored"

    with patch("evaluation.webarena_judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(calls, 1.0, max_retries=1, sleep_between=0)
    assert verdicts == [1.0]
    assert product == 1.0
    assert final == 1.0
    assert success is True
    assert status == "scored"


@pytest.mark.unit
def test_score_task_single_incorrect() -> None:
    calls = [{"kind": "fuzzy", "pred": "wrong", "reference": "right", "intent": "q"}]

    def fake_score(call):
        return 0.0, "scored"

    with patch("evaluation.webarena_judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(calls, 1.0, max_retries=1, sleep_between=0)
    assert verdicts == [0.0]
    assert product == 0.0
    assert final == 0.0
    assert success is False


@pytest.mark.unit
def test_score_task_multi_reference_product() -> None:
    calls = [
        {"kind": "fuzzy", "pred": "a", "reference": "r1", "intent": "q"},
        {"kind": "fuzzy", "pred": "a", "reference": "r2", "intent": "q"},
    ]

    results = iter([(1.0, "scored"), (0.0, "scored")])

    def fake_score(call):
        return next(results)

    with patch("evaluation.webarena_judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(calls, 1.0, max_retries=1, sleep_between=0)
    assert verdicts == [1.0, 0.0]
    assert product == 0.0
    assert final == 0.0


@pytest.mark.unit
def test_score_task_deferred_reward_zero_short_circuits() -> None:
    calls = [{"kind": "fuzzy", "pred": "a", "reference": "r", "intent": "q"}]

    def fake_score(call):
        return 1.0, "scored"

    with patch("evaluation.webarena_judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(calls, 0.0, max_retries=1, sleep_between=0)
    assert product == 1.0
    assert final == 0.0
    assert success is False


@pytest.mark.unit
def test_score_task_unparseable_status() -> None:
    calls = [{"kind": "fuzzy", "pred": "a", "reference": "r", "intent": "q"}]

    def fake_score(call):
        return 0.0, "unparseable"

    with patch("evaluation.webarena_judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(calls, 1.0, max_retries=1, sleep_between=0)
    assert status == "unparseable"


# ---------------------------------------------------------------------------
# 3. common.py header guard test
# ---------------------------------------------------------------------------

_WEBARENA_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "webarena"
if str(_WEBARENA_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBARENA_DIR))

from common import ensure_csv_header


@pytest.mark.unit
def test_ensure_csv_header_rejects_mismatched_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    old_columns = ["a", "b", "c"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=old_columns).writeheader()

    new_columns = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="Header mismatch"):
        ensure_csv_header(csv_path, new_columns)


@pytest.mark.unit
def test_ensure_csv_header_accepts_matching_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    columns = ["x", "y"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=columns).writeheader()
    ensure_csv_header(csv_path, columns)


@pytest.mark.unit
def test_ensure_csv_header_creates_new_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "new.csv"
    columns = ["x", "y"]
    ensure_csv_header(csv_path, columns)
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == columns


# ---------------------------------------------------------------------------
# 4. JSONL writer test
# ---------------------------------------------------------------------------

from common import append_judge_calls_record, judge_calls_path, judge_scores_path


@pytest.mark.unit
def test_judge_calls_path_derivation() -> None:
    p = Path("/data/webarena/trajectories_1.csv")
    assert judge_calls_path(p) == Path("/data/webarena/trajectories_1_judge_calls.jsonl")


@pytest.mark.unit
def test_judge_scores_path_derivation() -> None:
    p = Path("/data/webarena/trajectories_1.csv")
    assert judge_scores_path(p) == Path("/data/webarena/trajectories_1_judge_scores.csv")


@pytest.mark.unit
def test_append_judge_calls_record(tmp_path: Path) -> None:
    jsonl = tmp_path / "calls.jsonl"
    append_judge_calls_record(
        jsonl,
        task_id=42,
        intent="Find the price",
        run_status="ok",
        deferred_reward=1.0,
        calls=[{"kind": "fuzzy", "pred": "a", "reference": "b", "intent": "c"}],
    )
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task_id"] == 42
    assert record["deferred_reward"] == 1.0
    assert len(record["calls"]) == 1
