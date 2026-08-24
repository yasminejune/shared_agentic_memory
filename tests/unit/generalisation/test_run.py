"""Ordering and failure handling across the four shared-memory steps."""

from __future__ import annotations

import pytest

from agent_memories.generalisation import run

pytestmark = pytest.mark.unit


def _record_steps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    called: list[str] = []
    for name in ("step1_main", "step2_main", "step3_main", "step4_main"):
        monkeypatch.setattr(run, name, (lambda n: lambda: called.append(n))(name))
    return called


def test_main_runs_the_four_steps_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    called = _record_steps(monkeypatch)
    run.main()
    assert called == ["step1_main", "step2_main", "step3_main", "step4_main"]


def test_main_reports_the_paths_each_step_reads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _record_steps(monkeypatch)
    run.main()
    out = capsys.readouterr().out
    assert str(run.MEMORIES_CSV) in out
    assert str(run.WORK_DIR) in out
    assert str(run.DEFAULT_SHARED_STORE) in out
    assert "Steps 1-4 complete." in out


def test_main_stops_at_the_first_failing_step(monkeypatch: pytest.MonkeyPatch) -> None:
    called = _record_steps(monkeypatch)

    def boom() -> None:
        raise RuntimeError("step 2 failed")

    monkeypatch.setattr(run, "step2_main", boom)
    with pytest.raises(RuntimeError, match="step 2 failed"):
        run.main()
    assert called == ["step1_main"]
