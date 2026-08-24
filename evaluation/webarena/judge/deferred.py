"""Score deferred WebArena LLM-judge calls with Mistral.

Output: <stem>_judge_scores.csv next to the calls JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_memories.config import load_random_seed, set_global_seed
from agent_memories.services.mistral_client import MistralClient
from agent_memories.webarena.preflight import require_wa_env_vars
from evaluation.webarena.judge.scorer import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_SCORES_COLUMNS,
    install_mistral_judge,
    score_task,
)

# Minimum seconds between API attempts (success or failure); 429 retries use >=60s.
SLEEP_BETWEEN = 1.0
MAX_RETRIES = 3


def _export_bare_wa_vars() -> None:
    """Copy WA_* vars to their bare names for the webarena import-time assertion."""
    for key in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        wa_key = f"WA_{key}"
        if wa_key in os.environ:
            os.environ[key] = os.environ[wa_key]


def _load_calls(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_scored_task_ids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    scored: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("status") == "scored":
                tid = row.get("task_id", "").strip()
                if tid.isdigit():
                    scored.add(int(tid))
    return scored


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--calls",
        type=Path,
        required=True,
        help="Judge-calls JSONL file produced by the runner",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    seed = load_random_seed()
    set_global_seed(seed)
    print(f"[Judge] random_seed={seed}", flush=True)
    require_wa_env_vars()
    _export_bare_wa_vars()

    if not args.calls.exists():
        print(f"Judge-calls file not found: {args.calls}", file=sys.stderr)
        sys.exit(1)

    out_path = args.calls.with_name(
        args.calls.stem.replace("_judge_calls", "") + "_judge_scores.csv"
    )

    client = MistralClient(model=JUDGE_MODEL, max_response_tokens=JUDGE_MAX_TOKENS)
    install_mistral_judge(client)

    all_records = _load_calls(args.calls)

    already_scored = _load_scored_task_ids(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    if write_header:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=JUDGE_SCORES_COLUMNS).writeheader()

    total = len(all_records)
    new_scored = 0
    results: dict[int, dict] = {}

    for i, record in enumerate(all_records, 1):
        task_id = record["task_id"]
        if task_id in already_scored:
            results[task_id] = {"status": "scored"}
            continue

        calls = record["calls"]
        deferred_reward = record["deferred_reward"]

        print(
            f"[Judge] Scoring task {task_id} ({i}/{total}, {len(calls)} call(s))...",
            flush=True,
        )
        verdicts, judge_product, final_reward, final_success, status = score_task(
            calls,
            deferred_reward,
            max_retries=MAX_RETRIES,
            sleep_between=SLEEP_BETWEEN,
        )

        row = {
            "task_id": str(task_id),
            "judge_model": JUDGE_MODEL,
            "n_calls": str(len(calls)),
            "verdicts": json.dumps(verdicts),
            "judge_product": str(judge_product),
            "deferred_reward": str(deferred_reward),
            "final_reward": str(final_reward),
            "final_success": str(final_success),
            "status": status,
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with out_path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=JUDGE_SCORES_COLUMNS).writerow(row)
            fh.flush()

        results[task_id] = {"status": status}
        new_scored += 1

        if status == "scored":
            label = "PASS" if final_success else "fail"
            print(f"[Judge] Task {task_id}: {label} (judge_product={judge_product})", flush=True)
        else:
            print(f"[Judge] Task {task_id}: {status}", flush=True)

    scored_ok = sum(1 for r in results.values() if r["status"] == "scored")
    already_count = sum(1 for tid in already_scored if tid in {r["task_id"] for r in all_records})
    unresolved = {
        tid: r["status"]
        for tid, r in results.items()
        if r["status"] != "scored" and tid not in already_scored
    }

    print(
        f"\n[Judge] Scored {scored_ok}/{total} tasks "
        f"({new_scored - len(unresolved)} new this run, {already_count} already scored).",
        flush=True,
    )
    if unresolved:
        parts = [f"task {tid} {status}" for tid, status in sorted(unresolved.items())]
        print(f"[Judge] {len(unresolved)} unresolved: {', '.join(parts)}.", flush=True)
        print("[Judge] Re-run the same command to retry the unresolved tasks.", flush=True)

    # Count final_success among all scored (read back from file for accuracy)
    if out_path.exists():
        success_count = 0
        with out_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("final_success") == "True":
                    success_count += 1
        print(
            f"[Judge] Of the {scored_ok} scored, {success_count} passed "
            f"the judge (final_success=True).",
            flush=True,
        )

    print(f"[Judge] Scores at {out_path}", flush=True)


if __name__ == "__main__":
    main()
