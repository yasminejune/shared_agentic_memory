"""Build ReasoningBank memories from a WebArena trajectory CSV.

Output: data/webarena/trajectories_memories.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent_memories.config import load_random_seed, set_global_seed
from agent_memories.memory import Embedder, MemoryStore

# Off .pipeline rather than the package: the lazy __getattr__ on
# agent_memories.memory hides the type from mypy.
from agent_memories.memory.pipeline import MemoryBuildResult, MemoryPipeline
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient
from agent_memories.webarena.constants import (
    DEFAULT_MEMORIES_CSV,
    DEFAULT_TRAJECTORIES_CSV,
    MEMORY_CSV_COLUMNS,
    QWEN_MODEL,
    WEBARENA_USER_ID,
)
from agent_memories.webarena.csv_io import (
    append_csv_row,
    ensure_csv_header,
    load_memory_built_task_ids,
)
from agent_memories.webarena.preflight import require_ollama_model
from agent_memories.webarena.trajectories import state_from_trajectory_row


def _build_client(seed: int) -> ChatClient:
    return OllamaClient(model=QWEN_MODEL, seed=seed)


def _load_trajectory_rows(
    csv_path: Path,
    *,
    start_id: int,
    end_id: int,
) -> list[dict[str, str]]:
    if not csv_path.exists():
        print(f"Trajectories file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task_id_raw = row.get("task_id", "").strip()
            if not task_id_raw.isdigit():
                continue
            task_id = int(task_id_raw)
            if task_id < start_id or task_id > end_id:
                continue
            rows.append(row)
    return rows


def _write_memory_row(
    *,
    csv_path: Path,
    trajectory_row: dict[str, str],
    build_result: MemoryBuildResult,
) -> None:
    memory_json = "{}"
    embedding_json = "[]"
    entry = build_result.entry
    if entry is not None:
        memory_json = json.dumps(entry.to_dict_without_embedding(), ensure_ascii=False)
        embedding_json = json.dumps(entry.embedding or [], ensure_ascii=False)

    append_csv_row(
        csv_path,
        MEMORY_CSV_COLUMNS,
        {
            "task_id": trajectory_row["task_id"],
            "intent": trajectory_row.get("intent", ""),
            "intent_template_id": trajectory_row.get("intent_template_id", ""),
            "raw_trajectory": trajectory_row.get("raw_trajectory", ""),
            "memory": memory_json,
            "embedding": embedding_json,
            "judge_outcome": build_result.judge_outcome,
            "memory_extracted": str(build_result.memory_extracted),
            "harness_reward": trajectory_row.get("harness_reward", ""),
            "harness_success": trajectory_row.get("harness_success", ""),
            "run_status": trajectory_row.get("run_status", ""),
        },
    )


def _build_memory_for_row(
    *,
    trajectory_row: dict[str, str],
    pipeline: MemoryPipeline,
    embedder: Embedder,
) -> MemoryBuildResult:
    state = state_from_trajectory_row(trajectory_row)
    return pipeline.build_from_run(
        state,
        final_state=trajectory_row.get("final_state_yaml", ""),
        user_id=WEBARENA_USER_ID,
        embedder=embedder,
        response=trajectory_row.get("bot_response", "") or "N/A",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=811)
    args = parser.parse_args(argv)

    load_dotenv()
    seed = load_random_seed()
    set_global_seed(seed)
    print(f"[Memory] random_seed={seed}", flush=True)
    require_ollama_model(QWEN_MODEL)

    built_task_ids = load_memory_built_task_ids(DEFAULT_MEMORIES_CSV)
    ensure_csv_header(DEFAULT_MEMORIES_CSV, MEMORY_CSV_COLUMNS)

    client = _build_client(seed)
    embedder = Embedder()
    throwaway_store = MemoryStore.load(
        DEFAULT_MEMORIES_CSV.parent / ".throwaway_memories.jsonl",
        user_id=WEBARENA_USER_ID,
        embedder=embedder,
    )
    pipeline = MemoryPipeline(client=client, store=throwaway_store)

    trajectory_rows = _load_trajectory_rows(
        DEFAULT_TRAJECTORIES_CSV,
        start_id=args.start_id,
        end_id=args.end_id,
    )
    if not trajectory_rows:
        print(
            f"[Memory] No trajectory rows in [{args.start_id}, {args.end_id}] "
            f"from {DEFAULT_TRAJECTORIES_CSV}",
            flush=True,
        )
        return

    for row in trajectory_rows:
        task_id = int(row["task_id"])
        if task_id in built_task_ids:
            print(f"[Task {task_id}] skipped (memory already in output CSV)", flush=True)
            continue

        print(f"[Task {task_id}] building memory...", flush=True)
        build_result = _build_memory_for_row(
            trajectory_row=row,
            pipeline=pipeline,
            embedder=embedder,
        )
        _write_memory_row(
            csv_path=DEFAULT_MEMORIES_CSV,
            trajectory_row=row,
            build_result=build_result,
        )
        print(
            f"[Task {task_id}] judge={build_result.judge_outcome} "
            f"memory_extracted={build_result.memory_extracted}",
            flush=True,
        )

    print(f"[Memory] Done. Output at {DEFAULT_MEMORIES_CSV}", flush=True)


if __name__ == "__main__":
    main()
