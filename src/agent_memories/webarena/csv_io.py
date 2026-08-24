"""Append-only CSV helpers shared by the WebArena batch runners.

Every runner writes one row per task and resumes by reading back the
task_ids already recorded, so the header is checked before the first
append rather than trusted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agent_memories.webarena.constants import RUN_STATUS_OK


def load_ok_task_ids(csv_path: Path) -> set[int]:
    """Return task_ids that completed successfully and should be skipped on resume."""
    if not csv_path.exists():
        return set()
    ok_ids: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("run_status") != RUN_STATUS_OK:
                continue
            task_id = row.get("task_id", "").strip()
            if task_id.isdigit():
                ok_ids.add(int(task_id))
    return ok_ids


def load_memory_built_task_ids(csv_path: Path) -> set[int]:
    """Return task_ids already present in the memories CSV (resume skip)."""
    if not csv_path.exists():
        return set()
    built: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            if task_id.isdigit() and row.get("judge_outcome", "").strip():
                built.add(int(task_id))
    return built


def ensure_csv_header(csv_path: Path, columns: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            existing = next(reader, None)
        if existing is not None and existing != columns:
            raise ValueError(
                f"Header mismatch in {csv_path}:\n"
                f"  expected: {columns}\n"
                f"  found:    {existing}\n"
                "Pass a fresh --csv-path to start a new file."
            )
        return
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path: Path, columns: list[str], row: dict[str, str]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writerow(row)
        fh.flush()


def judge_calls_path(csv_path: Path) -> Path:
    """Sidecar JSONL path for deferred judge calls, derived from the CSV stem."""
    return csv_path.with_name(csv_path.stem + "_judge_calls.jsonl")


def judge_scores_path(csv_path: Path) -> Path:
    """Sidecar CSV path for offline judge scores, derived from the CSV stem."""
    return csv_path.with_name(csv_path.stem + "_judge_scores.csv")


def append_judge_calls_record(
    jsonl_path: Path,
    *,
    task_id: int,
    intent: str,
    run_status: str,
    deferred_reward: float,
    calls: list[dict],
) -> None:
    """Append one JSONL record for a task whose judge was deferred."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "intent": intent,
        "run_status": run_status,
        "deferred_reward": deferred_reward,
        "calls": calls,
    }
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
