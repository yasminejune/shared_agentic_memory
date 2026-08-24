"""Run a WebArena batch under one of the four memory conditions.

  A  no memories            -> data/webarena/trajectories_A_no_memories.csv
  B  private memories       -> data/webarena/trajectories_B_private_run.csv
  C  shared memories        -> data/webarena/trajectories_C_shared_only.csv
  D  private + shared       -> data/webarena/trajectories_D_private_shared.csv

The conditions differ only in which memory records are indexed; retrieval,
the agent loop and the CSV writing are shared.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_WEBARENA_DIR = Path(__file__).resolve().parent
if str(_WEBARENA_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBARENA_DIR))

from common import (
    DEFAULT_INFRA_LOG,
    DEFAULT_MAX_STEPS,
    DEFAULT_MEMORIES_CSV,
    DEFAULT_MEMORY_RUN_CSV,
    DEFAULT_PRIVATE_SHARED_RUN_CSV,
    DEFAULT_SHARED_RUN_CSV,
    DEFAULT_SHARED_STORE,
    DEFAULT_TRAJECTORIES_CSV,
    MEMORY_RUN_CSV_COLUMNS,
    QWEN_MODEL,
    RUN_STATUS_AGENT_ERROR,
    RUN_STATUS_OK,
    THINK_REQUEST_TIMEOUT_SECONDS,
    TRAJECTORY_CSV_COLUMNS,
    MemoryIndex,
    MemoryRecord,
    append_csv_row,
    append_judge_calls_record,
    audit_ids,
    ensure_csv_header,
    flatten_records_for_think,
    is_infra_error,
    judge_calls_path,
    load_intent_template_ids,
    load_ok_task_ids,
    load_private_records,
    load_shared_records,
    log_infra_error,
    prepare_webarena,
    require_nltk_punkt_tab,
    require_ollama_model,
    require_wa_env_vars,
    truncate_observation,
)
from dotenv import load_dotenv

from agent_memories.agent.browsergym import deferred_judge
from agent_memories.agent.browsergym.env import WebArenaEnvWrapper, make_webarena_env
from agent_memories.agent.browsergym.graph import build_graph
from agent_memories.agent.browsergym.nodes import extract_bot_response, make_think
from agent_memories.agent.constants import OBSERVATION_CHAR_BUDGET
from agent_memories.agent.state import AgentState, new_state
from agent_memories.config import load_random_seed, set_global_seed
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

DEFAULT_K = 3


@dataclass(frozen=True)
class Condition:
    """One experimental condition: where it writes and what it retrieves."""

    csv_path: Path
    columns: list[str]
    # "private", "shared", both, or neither for the no-memory baseline.
    sources: tuple[str, ...]


CONDITIONS = {
    "A": Condition(DEFAULT_TRAJECTORIES_CSV, TRAJECTORY_CSV_COLUMNS, ()),
    "B": Condition(DEFAULT_MEMORY_RUN_CSV, MEMORY_RUN_CSV_COLUMNS, ("private",)),
    "C": Condition(DEFAULT_SHARED_RUN_CSV, MEMORY_RUN_CSV_COLUMNS, ("shared",)),
    "D": Condition(
        DEFAULT_PRIVATE_SHARED_RUN_CSV,
        MEMORY_RUN_CSV_COLUMNS,
        ("private", "shared"),
    ),
}


def _build_client(seed: int) -> ChatClient:
    return OllamaClient(
        model=QWEN_MODEL,
        seed=seed,
        think=True,
        request_timeout=THINK_REQUEST_TIMEOUT_SECONDS,
    )


def _build_index(sources: tuple[str, ...], k: int) -> MemoryIndex | None:
    """Load the records for these sources. None when the condition uses no memories.

    Returning None for condition A keeps sentence-transformers unloaded.
    """
    if not sources:
        return None

    from agent_memories.memory import Embedder

    embedder = Embedder()
    records: list[MemoryRecord] = []

    if "private" in sources:
        private = load_private_records(DEFAULT_MEMORIES_CSV)
        if not private:
            print(f"No retrievable private memories in {DEFAULT_MEMORIES_CSV}", file=sys.stderr)
            sys.exit(1)
        records.extend(private)

    if "shared" in sources:
        shared = load_shared_records(DEFAULT_SHARED_STORE, embedder)
        if not shared:
            print(f"No retrievable shared memories in {DEFAULT_SHARED_STORE}", file=sys.stderr)
            sys.exit(1)
        records.extend(shared)

    n_private = sum(1 for source, _, _ in records if source == "private")
    print(
        f"[Memory] Loaded {n_private} private + {len(records) - n_private} shared "
        f"entries (k={k})",
        flush=True,
    )
    return MemoryIndex(records, embedder)


def _write_trajectory_row(
    *,
    csv_path: Path,
    columns: list[str],
    task_id: int,
    intent: str,
    intent_template_id: int,
    state: AgentState,
    wrapper: WebArenaEnvWrapper,
    final_state_yaml: str,
    bot_response: str,
    run_status: str,
    retrieved: list[MemoryRecord],
) -> None:
    harness_reward = wrapper.last_reward
    harness_success = harness_reward > 0
    pending = bool(wrapper.last_judge_calls)

    row = {
        "task_id": str(task_id),
        "intent": intent,
        "intent_template_id": str(intent_template_id),
        "raw_trajectory": json.dumps(state.get("history", []), ensure_ascii=False),
        "final_state_yaml": final_state_yaml,
        "bot_response": bot_response,
        "harness_reward": str(harness_reward),
        "harness_success": str(harness_success),
        "run_status": run_status,
        "judge_pending": str(pending),
    }
    # Condition A keeps the original 10-column layout so its existing CSV resumes.
    if "retrieved_task_ids" in columns:
        row["retrieved_task_ids"] = json.dumps(audit_ids(retrieved))
        row["retrieved_memory_titles"] = json.dumps(
            [item.title for _, _, entry in retrieved for item in entry.items],
            ensure_ascii=False,
        )

    append_csv_row(csv_path, columns, row)
    if pending:
        append_judge_calls_record(
            judge_calls_path(csv_path),
            task_id=task_id,
            intent=intent,
            run_status=run_status,
            deferred_reward=harness_reward,
            calls=wrapper.last_judge_calls,
        )


def _run_task(
    *,
    task_id: int,
    intent_template_id: int,
    client: ChatClient,
    index: MemoryIndex | None,
    k: int,
    condition: Condition,
    infra_log: Path,
    max_steps: int,
    headless: bool,
) -> None:
    env = make_webarena_env(task_id, headless=headless)
    wrapper = WebArenaEnvWrapper(env)
    state: AgentState = new_state(aim="")
    intent = ""
    retrieved: list[MemoryRecord] = []

    try:
        obs, _info = wrapper.reset()
        intent = str(obs.get("goal", ""))
        state = new_state(aim=intent)

        retrieved = index.search(intent, k=k) if index is not None else []
        state["memories"] = flatten_records_for_think(retrieved)
        if index is not None:
            print(
                f"[Task {task_id}] retrieved {len(retrieved)} entries "
                f"({len(state['memories'])} item(s)) ids={audit_ids(retrieved)}",
                flush=True,
            )

        graph = build_graph(wrapper, make_think(client), max_steps=max_steps)
        state = graph.invoke(state)

        final_state_yaml = truncate_observation(
            state.get("observation", {}).get("tree_yaml", ""),
            OBSERVATION_CHAR_BUDGET,
        )
        bot_response = extract_bot_response(state)

        _write_trajectory_row(
            csv_path=condition.csv_path,
            columns=condition.columns,
            task_id=task_id,
            intent=intent,
            intent_template_id=intent_template_id,
            state=state,
            wrapper=wrapper,
            final_state_yaml=final_state_yaml,
            bot_response=bot_response,
            run_status=RUN_STATUS_OK,
            retrieved=retrieved,
        )
        print(
            f"[Task {task_id}] harness_success={wrapper.last_reward > 0} "
            f"steps={len(state.get('history', []))}",
            flush=True,
        )
    except Exception as exc:
        if is_infra_error(exc):
            log_infra_error(
                infra_log,
                task_id=task_id,
                message=f"{type(exc).__name__}: {exc}",
            )
            return
        print(f"[Task {task_id}] ERROR: {type(exc).__name__}: {exc}", flush=True)
        final_state_yaml = truncate_observation(
            state.get("observation", {}).get("tree_yaml", ""),
            OBSERVATION_CHAR_BUDGET,
        )
        bot_response = extract_bot_response(state)
        _write_trajectory_row(
            csv_path=condition.csv_path,
            columns=condition.columns,
            task_id=task_id,
            intent=intent or f"(task {task_id} failed before reset)",
            intent_template_id=intent_template_id,
            state=state,
            wrapper=wrapper,
            final_state_yaml=final_state_yaml,
            bot_response=bot_response,
            run_status=RUN_STATUS_AGENT_ERROR,
            retrieved=retrieved,
        )
    finally:
        wrapper.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--no-headless", action="store_false", dest="headless", default=True)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=811)
    args = parser.parse_args(argv)

    condition = CONDITIONS[args.condition]

    load_dotenv()
    seed = load_random_seed()
    set_global_seed(seed)
    print(f"[Runner] condition={args.condition} random_seed={seed}", flush=True)
    require_wa_env_vars()
    require_nltk_punkt_tab()
    require_ollama_model(QWEN_MODEL)

    index = _build_index(condition.sources, args.k)

    prepare_webarena(
        headless=args.headless,
        smoke=True,
        massage=True,
        infra_log=DEFAULT_INFRA_LOG,
    )
    deferred_judge.install()

    template_ids = load_intent_template_ids()
    ok_task_ids = load_ok_task_ids(condition.csv_path)
    ensure_csv_header(condition.csv_path, condition.columns)

    client = _build_client(seed)

    for task_id in range(args.start_id, args.end_id + 1):
        if task_id in ok_task_ids:
            print(f"[Task {task_id}] skipped (run_status=ok in CSV)", flush=True)
            continue

        intent_template_id = template_ids.get(task_id, -1)
        _run_task(
            task_id=task_id,
            intent_template_id=intent_template_id,
            client=client,
            index=index,
            k=args.k,
            condition=condition,
            infra_log=DEFAULT_INFRA_LOG,
            max_steps=DEFAULT_MAX_STEPS,
            headless=args.headless,
        )

    print(f"[Runner] Done. Trajectories at {condition.csv_path}", flush=True)
    calls_file = judge_calls_path(condition.csv_path)
    if calls_file.exists():
        n = sum(1 for _ in calls_file.open("r", encoding="utf-8"))
        print(
            f"[Runner] {n} tasks awaiting LLM judge. Next:\n"
            f"  python scripts/webarena/score_deferred_judge.py "
            f"--calls {calls_file}",
            flush=True,
        )


if __name__ == "__main__":
    main()
