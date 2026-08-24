"""The deferred WebArena LLM judge: call capture, scoring and its CSV artefacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_memories.agent.browsergym import deferred_judge
from agent_memories.webarena.csv_io import (
    append_judge_calls_record,
    ensure_csv_header,
    judge_calls_path,
    judge_scores_path,
)
from evaluation.webarena.judge.scorer import score_task

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_shim():
    deferred_judge.reset()
    yield
    deferred_judge.reset()


@pytest.mark.parametrize(
    ("stub", "kind"),
    [
        (deferred_judge._stub_fuzzy_match, "fuzzy"),
        (deferred_judge._stub_ua_match, "ua"),
    ],
)
def test_stub_defers_the_call_and_returns_1(stub: Callable[..., float], kind: str) -> None:
    assert stub("pred_val", "ref_val", "intent_val") == 1.0
    assert deferred_judge.calls() == [
        {
            "kind": kind,
            "pred": "pred_val",
            "reference": "ref_val",
            "intent": "intent_val",
        }
    ]


def test_reset_clears_buffer() -> None:
    deferred_judge._stub_fuzzy_match("a", "b", "c")
    assert len(deferred_judge.calls()) == 1
    deferred_judge.reset()
    assert len(deferred_judge.calls()) == 0


def test_calls_returns_copy() -> None:
    deferred_judge._stub_fuzzy_match("a", "b", "c")
    copy = deferred_judge.calls()
    copy.append({"extra": True})
    assert len(deferred_judge.calls()) == 1


def test_install_is_idempotent() -> None:
    import types

    fake_harness = types.ModuleType("webarena.evaluation_harness")
    fake_evaluators = types.ModuleType("webarena.evaluation_harness.evaluators")
    fake_evaluators.llm_fuzzy_match = lambda p, r, q: 0.0  # type: ignore[attr-defined]
    fake_evaluators.llm_ua_match = lambda p, r, q: 0.0  # type: ignore[attr-defined]
    fake_harness.evaluators = fake_evaluators  # type: ignore[attr-defined]

    saved = deferred_judge._installed
    try:
        deferred_judge._installed = False
        with patch.dict(
            "sys.modules",
            {
                "webarena.evaluation_harness": fake_harness,
                "webarena.evaluation_harness.evaluators": fake_evaluators,
            },
        ):
            deferred_judge.install()
            assert fake_evaluators.llm_fuzzy_match is deferred_judge._stub_fuzzy_match
            assert fake_evaluators.llm_ua_match is deferred_judge._stub_ua_match
            deferred_judge.install()
            assert fake_evaluators.llm_fuzzy_match is deferred_judge._stub_fuzzy_match
    finally:
        deferred_judge._installed = saved


def test_score_task_single_correct() -> None:
    calls = [{"kind": "fuzzy", "pred": "3h30", "reference": "3h 30min", "intent": "How long?"}]

    def fake_score(call):
        return 1.0, "scored"

    with patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(
            calls, 1.0, max_retries=1, sleep_between=0
        )
    assert verdicts == [1.0]
    assert product == 1.0
    assert final == 1.0
    assert success is True
    assert status == "scored"


def test_score_task_single_incorrect() -> None:
    calls = [{"kind": "fuzzy", "pred": "wrong", "reference": "right", "intent": "q"}]

    def fake_score(call):
        return 0.0, "scored"

    with patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(
            calls, 1.0, max_retries=1, sleep_between=0
        )
    assert verdicts == [0.0]
    assert product == 0.0
    assert final == 0.0
    assert success is False


def test_score_task_multi_reference_product() -> None:
    calls = [
        {"kind": "fuzzy", "pred": "a", "reference": "r1", "intent": "q"},
        {"kind": "fuzzy", "pred": "a", "reference": "r2", "intent": "q"},
    ]

    results = iter([(1.0, "scored"), (0.0, "scored")])

    def fake_score(call):
        return next(results)

    with patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(
            calls, 1.0, max_retries=1, sleep_between=0
        )
    assert verdicts == [1.0, 0.0]
    assert product == 0.0
    assert final == 0.0


def test_score_task_zero_deferred_reward_wins() -> None:
    calls = [{"kind": "fuzzy", "pred": "a", "reference": "r", "intent": "q"}]

    def fake_score(call):
        return 1.0, "scored"

    with patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(
            calls, 0.0, max_retries=1, sleep_between=0
        )
    assert product == 1.0
    assert final == 0.0
    assert success is False


def test_score_task_unparseable_status() -> None:
    calls = [{"kind": "fuzzy", "pred": "a", "reference": "r", "intent": "q"}]

    def fake_score(call):
        return 0.0, "unparseable"

    with patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score):
        verdicts, product, final, success, status = score_task(
            calls, 1.0, max_retries=1, sleep_between=0
        )
    assert status == "unparseable"


def test_score_task_paces_between_calls() -> None:
    calls = [
        {"kind": "fuzzy", "pred": "a", "reference": "r1", "intent": "q"},
        {"kind": "fuzzy", "pred": "a", "reference": "r2", "intent": "q"},
    ]
    results = iter([(1.0, "scored"), (1.0, "scored")])

    def fake_score(call):
        return next(results)

    with (
        patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score),
        patch("evaluation.webarena.judge.scorer.time.sleep") as sleep_mock,
    ):
        score_task(calls, 1.0, max_retries=1, sleep_between=30.0)

    assert sleep_mock.call_count == 2
    assert all(call.args[0] == 30.0 for call in sleep_mock.call_args_list)


def test_score_task_rate_limit_retry_waits_a_minute() -> None:
    calls = [{"kind": "fuzzy", "pred": "a", "reference": "r", "intent": "q"}]
    results = iter(
        [
            (0.0, "judge_error: SDKError: Status 429 rate_limited"),
            (1.0, "scored"),
        ]
    )

    def fake_score(call):
        return next(results)

    with (
        patch("evaluation.webarena.judge.scorer.score_single_call", side_effect=fake_score),
        patch("evaluation.webarena.judge.scorer.time.sleep") as sleep_mock,
        patch("evaluation.webarena.judge.scorer.random.uniform", return_value=0.0),
    ):
        score_task(calls, 1.0, max_retries=3, sleep_between=1.0)

    assert sleep_mock.call_count == 2
    assert sleep_mock.call_args_list[0].args[0] == 60.0
    assert sleep_mock.call_args_list[1].args[0] == 1.0


def test_ensure_csv_header_rejects_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    old_columns = ["a", "b", "c"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=old_columns).writeheader()

    new_columns = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="Header mismatch"):
        ensure_csv_header(csv_path, new_columns)


def test_ensure_csv_header_accepts_match(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    columns = ["x", "y"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=columns).writeheader()
    ensure_csv_header(csv_path, columns)


def test_ensure_csv_header_creates_new_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "new.csv"
    columns = ["x", "y"]
    ensure_csv_header(csv_path, columns)
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == columns


def test_judge_calls_path_derivation() -> None:
    p = Path("/data/webarena/trajectories_1.csv")
    assert judge_calls_path(p) == Path("/data/webarena/trajectories_1_judge_calls.jsonl")


def test_judge_scores_path_derivation() -> None:
    p = Path("/data/webarena/trajectories_1.csv")
    assert judge_scores_path(p) == Path("/data/webarena/trajectories_1_judge_scores.csv")


def test_append_judge_calls_record(tmp_path: Path) -> None:
    jsonl = tmp_path / "calls.jsonl"
    append_judge_calls_record(
        jsonl,
        task_id=137,
        intent="Find the price",
        run_status="ok",
        deferred_reward=1.0,
        calls=[{"kind": "fuzzy", "pred": "a", "reference": "b", "intent": "c"}],
    )
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task_id"] == 137
    assert record["deferred_reward"] == 1.0
    assert len(record["calls"]) == 1
