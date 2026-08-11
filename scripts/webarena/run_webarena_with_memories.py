"""WP4 upper bound — WebArena rerun with top-k private (non-generalised) memories.

Mirrors ``scripts/webarena/run_webarena.py`` exactly except that before
each task the top-``k`` memory entries most similar to the task intent
are retrieved from ``data/webarena/trajectories_memories.csv`` (produced
by ``build_memories_from_trajectories.py``) and injected into
``state['memories']`` as flattened ``{title, content}`` dicts, the shape
the BrowserGym Think prompt renders under the ReasoningBank
memory-injection instruction.

These are the pure per-task private memories (no WP2 generalisation), so
the run is the upper bound on what memory injection can contribute. By
default the memory distilled from a task's own trajectory is retrievable
(its query is the identical intent string, so it ranks first); pass
``--exclude-own-memory`` for the leave-one-out variant that measures
transfer from other tasks only.

Output columns are the ``trajectories_0.csv`` columns plus two trailing
audit columns (``retrieved_task_ids``, ``retrieved_memory_titles``), so
row-by-row comparison against the no-memory run works unchanged on the
shared prefix. Resume skips only rows with ``run_status=ok``. Infra
failures are logged to ``infra_errors.log`` and are not written to the CSV.

Output: ``data/webarena/trajectories_private_memories.csv`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_WEBARENA_DIR = Path(__file__).resolve().parent
if str(_WEBARENA_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBARENA_DIR))

import numpy as np
from common import (
    DEFAULT_INFRA_LOG,
    DEFAULT_MAX_STEPS,
    DEFAULT_MEMORIES_CSV,
    DEFAULT_MEMORY_RUN_CSV,
    MEMORY_RUN_CSV_COLUMNS,
    QWEN_MODEL,
    RUN_STATUS_AGENT_ERROR,
    RUN_STATUS_OK,
    THINK_REQUEST_TIMEOUT_SECONDS,
    append_csv_row,
    append_judge_calls_record,
    ensure_csv_header,
    is_infra_error,
    judge_calls_path,
    load_intent_template_ids,
    load_memory_entries_from_csv,
    load_ok_task_ids,
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
from agent_memories.agent.nodes import OBSERVATION_CHAR_BUDGET
from agent_memories.agent.state import AgentState, new_state
from agent_memories.config import load_random_seed, set_global_seed
from agent_memories.memory import Embedder, MemoryEntry
from agent_memories.memory.store import _cosine_top_k
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

DEFAULT_K = 3


def _build_client(seed: int) -> ChatClient:
    return OllamaClient(
        model=QWEN_MODEL,
        seed=seed,
        think=True,
        request_timeout=THINK_REQUEST_TIMEOUT_SECONDS,
    )


class PrivateMemoryIndex:
    """Cosine top-k retrieval over the CSV-loaded ``(task_id, MemoryEntry)`` pairs.

    Same brute-force cosine search as :class:`MemoryStore` (reusing its
    ``_cosine_top_k``), but keeps the originating ``task_id`` alongside
    each entry so the current task's own memory can be excluded
    (leave-one-out) and the retrieval can be audited in the output CSV.
    """

    def __init__(self, pairs: list[tuple[int, MemoryEntry]], embedder: Embedder) -> None:
        self._pairs = pairs
        self._embedder = embedder
        self._matrix = np.asarray(
            [entry.embedding for _, entry in pairs],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def search(
        self,
        intent: str,
        *,
        k: int,
        exclude_task_id: int | None = None,
    ) -> list[tuple[int, MemoryEntry]]:
        """Return the top-``k`` pairs by cosine similarity to ``intent``.

        When ``exclude_task_id`` is set, the memory built from that
        task's own trajectory is filtered out; one extra candidate is
        fetched so the result still holds ``k`` entries.
        """
        if not self._pairs or k <= 0:
            return []
        query_vec = np.asarray(self._embedder.embed(intent), dtype=np.float32)
        fetch = k + 1 if exclude_task_id is not None else k
        indices = _cosine_top_k(query_vec, self._matrix, fetch)
        results = [self._pairs[i] for i in indices]
        if exclude_task_id is not None:
            results = [pair for pair in results if pair[0] != exclude_task_id]
        return results[:k]


def _flatten_entries_for_think(entries: list[MemoryEntry]) -> list[dict[str, str]]:
    """Render retrieved entries as the flat ``{title, content}`` list Think expects.

    The ``description`` field is dropped because the paper renders items
    in the agent prompt with title and content only (Appendix A.2).
    """
    flat: list[dict[str, str]] = []
    for entry in entries:
        for item in entry.items:
            flat.append({"title": item.title, "content": item.content})
    return flat


def _write_trajectory_row(
    *,
    csv_path: Path,
    task_id: int,
    intent: str,
    intent_template_id: int,
    state: AgentState,
    wrapper: WebArenaEnvWrapper,
    final_state_yaml: str,
    bot_response: str,
    run_status: str,
    retrieved: list[tuple[int, MemoryEntry]],
) -> None:
    harness_reward = wrapper.last_reward
    harness_success = harness_reward > 0
    pending = bool(wrapper.last_judge_calls)

    retrieved_task_ids = json.dumps([tid for tid, _ in retrieved])
    retrieved_titles = json.dumps(
        [item.title for _, entry in retrieved for item in entry.items],
        ensure_ascii=False,
    )

    append_csv_row(
        csv_path,
        MEMORY_RUN_CSV_COLUMNS,
        {
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
            "retrieved_task_ids": retrieved_task_ids,
            "retrieved_memory_titles": retrieved_titles,
        },
    )
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
    index: PrivateMemoryIndex,
    k: int,
    exclude_own_memory: bool,
    csv_path: Path,
    infra_log: Path,
    max_steps: int,
    headless: bool,
) -> None:
    env = make_webarena_env(task_id, headless=headless)
    wrapper = WebArenaEnvWrapper(env)
    state: AgentState = new_state(aim="")
    intent = ""
    retrieved: list[tuple[int, MemoryEntry]] = []

    try:
        obs, _info = wrapper.reset()
        intent = str(obs.get("goal", ""))
        state = new_state(aim=intent)

        retrieved = index.search(
            intent,
            k=k,
            exclude_task_id=task_id if exclude_own_memory else None,
        )
        state["memories"] = _flatten_entries_for_think([entry for _, entry in retrieved])
        print(
            f"[Task {task_id}] retrieved {len(retrieved)} entries "
            f"({len(state['memories'])} item(s)) from tasks "
            f"{[tid for tid, _ in retrieved]}",
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
            csv_path=csv_path,
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
            csv_path=csv_path,
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
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=811)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="How many memory entries to inject into the Think prompt (default: 1)",
    )
    parser.add_argument(
        "--exclude-own-memory",
        action="store_true",
        help="Leave-one-out: never retrieve the memory built from the task's own trajectory",
    )
    parser.add_argument(
        "--memories-csv",
        type=Path,
        default=DEFAULT_MEMORIES_CSV,
        help="Input memories CSV from build_memories_from_trajectories.py",
    )
    parser.add_argument(
        "--reset-every", type=int, default=0, help="Full reset every N tasks (0=off)"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_MEMORY_RUN_CSV,
        help="Append-only trajectories CSV output path",
    )
    parser.add_argument(
        "--infra-log",
        type=Path,
        default=DEFAULT_INFRA_LOG,
        help="Append-only log for connection / reachability failures",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip webarena.0 env open/reset smoke step",
    )
    parser.add_argument(
        "--skip-massage",
        action="store_true",
        help="Skip BrowserGym massage_tasks warm-up",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    seed = load_random_seed()
    set_global_seed(seed)
    print(f"[Runner] random_seed={seed}", flush=True)
    require_wa_env_vars()
    require_nltk_punkt_tab()
    require_ollama_model(QWEN_MODEL)

    memory_pairs = load_memory_entries_from_csv(args.memories_csv)
    if not memory_pairs:
        print(f"No retrievable memories in {args.memories_csv}", file=sys.stderr)
        sys.exit(1)
    embedder = Embedder()
    index = PrivateMemoryIndex(memory_pairs, embedder)
    print(
        f"[Memory] Loaded {len(index)} retrievable entries from {args.memories_csv} "
        f"(k={args.k}, exclude_own_memory={args.exclude_own_memory})",
        flush=True,
    )

    smoke = not args.skip_smoke
    prepare_webarena(
        headless=args.headless,
        smoke=smoke,
        massage=not args.skip_massage,
        infra_log=args.infra_log,
    )
    deferred_judge.install()

    template_ids = load_intent_template_ids()
    ok_task_ids = load_ok_task_ids(args.csv_path)
    ensure_csv_header(args.csv_path, MEMORY_RUN_CSV_COLUMNS)

    client = _build_client(seed)

    import browsergym.webarena  # noqa: F401
    from browsergym.webarena.instance import WebArenaInstance

    instance = WebArenaInstance()
    tasks_run = 0
    for task_id in range(args.start_id, args.end_id + 1):
        if task_id in ok_task_ids:
            print(f"[Task {task_id}] skipped (run_status=ok in CSV)", flush=True)
            continue

        if args.reset_every > 0 and tasks_run > 0 and tasks_run % args.reset_every == 0:
            print(f"[Runner] full_reset() after {tasks_run} tasks...", flush=True)
            instance.full_reset()

        intent_template_id = template_ids.get(task_id, -1)
        _run_task(
            task_id=task_id,
            intent_template_id=intent_template_id,
            client=client,
            index=index,
            k=args.k,
            exclude_own_memory=args.exclude_own_memory,
            csv_path=args.csv_path,
            infra_log=args.infra_log,
            max_steps=args.max_steps,
            headless=args.headless,
        )
        tasks_run += 1

    print(f"[Runner] Done. Trajectories at {args.csv_path}", flush=True)
    calls_file = judge_calls_path(args.csv_path)
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
