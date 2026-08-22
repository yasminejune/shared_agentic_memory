"""Compare WebArena conditions A/B/C/D on the 812-task success rate.

Writes data/webarena/condition_comparison.csv (and B/C memory-pull CSVs).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR, REPO_ROOT

N_TASKS = 812
PRIVATE_MEMORIES_CSV = (
    REPO_ROOT / "data" / "webarena" / "trajectories_reasoningbank_private_memories.csv"
)
SHARED_STORE = DEFAULT_MEMORY_DIR / "shared.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "webarena"
TERMINAL_CONTENT_CHARS = 120

SUMMARY_COLUMNS = [
    "condition",
    "n_success",
    "success_rate",
    "n_ok",
    "n_agent_error",
    "n_missing",
    "judge_pending",
    "judge_scored",
    "judge_status",
    "n_unknown",
]
MEMORY_COLUMNS = ["title", "content", "pulls"]


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def _parse_json_list(raw: str | None) -> list:
    text = (raw or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        return []
    return parsed


def judge_scores_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_judge_scores.csv")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def last_row_per_task(rows: Sequence[dict[str, str]]) -> dict[int, dict[str, str]]:
    """Keep the last run_status=ok row per task; else the last row."""
    last: dict[int, dict[str, str]] = {}
    last_ok: dict[int, dict[str, str]] = {}
    for row in rows:
        raw = (row.get("task_id") or "").strip()
        if not raw.isdigit():
            continue
        task_id = int(raw)
        last[task_id] = row
        if row.get("run_status") == "ok":
            last_ok[task_id] = row
    return {task_id: last_ok.get(task_id, row) for task_id, row in last.items()}


def load_judge_scores(path: Path) -> dict[int, dict[str, str]]:
    """Last row per task; prefer the last status=scored row if any."""
    if not path.exists():
        return {}
    last: dict[int, dict[str, str]] = {}
    last_scored: dict[int, dict[str, str]] = {}
    for row in load_csv_rows(path):
        raw = (row.get("task_id") or "").strip()
        if not raw.isdigit():
            continue
        task_id = int(raw)
        last[task_id] = row
        if row.get("status") == "scored":
            last_scored[task_id] = row
    return {task_id: last_scored.get(task_id, row) for task_id, row in last.items()}


def combined_for_row(
    row: dict[str, str],
    scores: dict[int, dict[str, str]],
) -> tuple[bool, bool]:
    """Return (success, unknown) for one last-row-per-task.

    Judge-pending tasks need a scored sidecar row; unscored counts as fail.
    """
    harness = _is_true(row.get("harness_success"))
    if not _is_true(row.get("judge_pending")):
        return harness, False
    raw = (row.get("task_id") or "").strip()
    task_id = int(raw) if raw.isdigit() else -1
    score = scores.get(task_id)
    if score is not None and score.get("status") == "scored":
        return _is_true(score.get("final_success")), False
    return False, True


def summarise_condition(
    name: str,
    rows_by_task: dict[int, dict[str, str]],
    scores: dict[int, dict[str, str]],
) -> dict[str, str]:
    """WebArena success rate over all N_TASKS ids (missing/error/unscored = 0)."""
    n_agent_error = 0
    n_ok = 0
    n_missing = 0
    n_pending = 0
    n_scored = 0
    n_success = 0
    n_unknown = 0

    for task_id in range(N_TASKS):
        row = rows_by_task.get(task_id)
        if row is None:
            n_missing += 1
            continue

        status = row.get("run_status", "")
        if status == "agent_error":
            n_agent_error += 1
        if status == "ok":
            n_ok += 1
        if _is_true(row.get("judge_pending")):
            n_pending += 1
            if scores.get(task_id, {}).get("status") == "scored":
                n_scored += 1

        if status != "ok":
            continue
        success, unknown = combined_for_row(row, scores)
        if unknown:
            n_unknown += 1
        elif success:
            n_success += 1

    if n_pending == 0:
        judge_status = "none"
    elif n_scored == n_pending:
        judge_status = "complete"
    else:
        judge_status = f"partial ({n_scored}/{n_pending})"

    rate_pct = 100.0 * n_success / N_TASKS
    return {
        "condition": name,
        "n_success": str(n_success),
        "success_rate": f"{rate_pct:.1f}%",
        "n_ok": str(n_ok),
        "n_agent_error": str(n_agent_error),
        "n_missing": str(n_missing),
        "judge_pending": str(n_pending),
        "judge_scored": str(n_scored),
        "judge_status": judge_status,
        "n_unknown": str(n_unknown),
    }


def load_private_memory_items(path: Path) -> dict[int, list[tuple[str, str]]]:
    """Map source task id to (title, content) items from the RB memories CSV."""
    items_by_task: dict[int, list[tuple[str, str]]] = {}
    for row in load_csv_rows(path):
        raw = (row.get("task_id") or "").strip()
        if not raw.isdigit():
            continue
        if row.get("memory_extracted", "").strip() != "True":
            continue
        payload = json.loads(row.get("memory") or "{}")
        items: list[tuple[str, str]] = []
        for item in payload.get("items") or []:
            title = str(item.get("title") or "")
            content = str(item.get("content") or "")
            items.append((title, content))
        items_by_task[int(raw)] = items
    return items_by_task


def load_shared_items(path: Path) -> list[tuple[str, str, str]]:
    """Return (query, title, content) for every item in shared.jsonl."""
    records: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            query = str(entry.get("query") or "")
            for item in entry.get("items") or []:
                records.append(
                    (
                        query,
                        str(item.get("title") or ""),
                        str(item.get("content") or ""),
                    )
                )
    return records


def count_b_pulls(
    rows_by_task: dict[int, dict[str, str]],
    items_by_task: dict[int, list[tuple[str, str]]],
) -> list[dict[str, str]]:
    pulls: Counter[tuple[str, str]] = Counter()
    for row in rows_by_task.values():
        for source_id in _parse_json_list(row.get("retrieved_task_ids")):
            try:
                task_id = int(source_id)
            except (TypeError, ValueError):
                continue
            for title, content in items_by_task.get(task_id, ()):
                pulls[(title, content)] += 1
    return _pulls_to_rows(pulls)


def count_c_pulls(
    rows_by_task: dict[int, dict[str, str]],
    shared_items: Sequence[tuple[str, str, str]],
) -> list[dict[str, str]]:
    by_query_title: dict[tuple[str, str], str] = {}
    by_title: dict[str, str] = {}
    for query, title, content in shared_items:
        by_query_title.setdefault((query, title), content)
        by_title.setdefault(title, content)

    pulls: Counter[tuple[str, str]] = Counter()
    for row in rows_by_task.values():
        labels = [str(x) for x in _parse_json_list(row.get("retrieved_task_ids"))]
        titles = [str(x) for x in _parse_json_list(row.get("retrieved_memory_titles"))]
        if len(labels) == len(titles):
            pairs = list(zip(labels, titles, strict=True))
        else:
            pairs = [("", title) for title in titles]
        for label, title in pairs:
            if label:
                content = by_query_title.get((label, title), by_title.get(title, ""))
            else:
                content = by_title.get(title, "")
            pulls[(title, content)] += 1
    return _pulls_to_rows(pulls)


def _pulls_to_rows(pulls: Counter[tuple[str, str]]) -> list[dict[str, str]]:
    rows = [
        {"title": title, "content": content, "pulls": str(count)}
        for (title, content), count in pulls.most_common()
    ]
    return rows


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _col_widths(columns: Sequence[str], rows: Sequence[dict[str, str]]) -> list[int]:
    widths = [len(col) for col in columns]
    for row in rows:
        for i, col in enumerate(columns):
            widths[i] = max(widths[i], len(row.get(col, "")))
    return widths


def print_table(title: str, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    display_rows = []
    for row in rows:
        shown = dict(row)
        if "content" in shown and len(shown["content"]) > TERMINAL_CONTENT_CHARS:
            shown["content"] = shown["content"][: TERMINAL_CONTENT_CHARS - 3] + "..."
        display_rows.append(shown)
    widths = _col_widths(columns, display_rows)
    print(f"\n{title}")
    header = "  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(header)
    print("  ".join("-" * widths[i] for i in range(len(columns))))
    for row in display_rows:
        print(
            "  ".join(row.get(col, "").ljust(widths[i]) for i, col in enumerate(columns))
        )


def load_condition(path: Path) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    if not path.exists():
        print(f"Trajectory CSV not found: {path}", file=sys.stderr)
        sys.exit(1)
    rows_by_task = last_row_per_task(load_csv_rows(path))
    scores = load_judge_scores(judge_scores_path(path))
    return rows_by_task, scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--a", type=Path, default=None, help="Condition A trajectory CSV")
    parser.add_argument("--b", type=Path, default=None, help="Condition B trajectory CSV")
    parser.add_argument("--c", type=Path, default=None, help="Condition C trajectory CSV")
    parser.add_argument("--d", type=Path, default=None, help="Condition D trajectory CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for output CSVs (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)

    named: list[tuple[str, Path]] = []
    for letter, path in (("A", args.a), ("B", args.b), ("C", args.c), ("D", args.d)):
        if path is not None:
            named.append((letter, path))
    if not named:
        parser.error("at least one of --a/--b/--c/--d is required")

    summary_rows: list[dict[str, str]] = []
    loaded: dict[str, dict[int, dict[str, str]]] = {}
    for name, path in named:
        rows_by_task, scores = load_condition(path)
        loaded[name] = rows_by_task
        summary_rows.append(summarise_condition(name, rows_by_task, scores))

    out_dir: Path = args.out_dir
    summary_path = out_dir / "condition_comparison.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print_table(
        "Table 1 — success rate over 812 tasks (missing, agent_error, unscored judge = 0)",
        SUMMARY_COLUMNS,
        summary_rows,
    )
    print(f"\nWrote {summary_path}")

    if args.b is not None:
        if not PRIVATE_MEMORIES_CSV.exists():
            print(
                f"Private memories CSV not found: {PRIVATE_MEMORIES_CSV}",
                file=sys.stderr,
            )
            sys.exit(1)
        items_by_task = load_private_memory_items(PRIVATE_MEMORIES_CSV)
        memory_rows = count_b_pulls(loaded["B"], items_by_task)
        mem_path = out_dir / "condition_comparison_memories_B.csv"
        write_csv(mem_path, MEMORY_COLUMNS, memory_rows)
        print_table("Memories B — injected items (content truncated in terminal)", MEMORY_COLUMNS, memory_rows)
        print(f"\nWrote {mem_path} ({len(memory_rows)} distinct items)")

    if args.c is not None:
        if not SHARED_STORE.exists():
            print(f"Shared store not found: {SHARED_STORE}", file=sys.stderr)
            sys.exit(1)
        shared_items = load_shared_items(SHARED_STORE)
        memory_rows = count_c_pulls(loaded["C"], shared_items)
        mem_path = out_dir / "condition_comparison_memories_C.csv"
        write_csv(mem_path, MEMORY_COLUMNS, memory_rows)
        print_table("Memories C — injected items (content truncated in terminal)", MEMORY_COLUMNS, memory_rows)
        print(f"\nWrote {mem_path} ({len(memory_rows)} distinct items)")


if __name__ == "__main__":
    main()
