"""Shared helpers for the WebArena batch runners."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import requests

if TYPE_CHECKING:
    from agent_memories.memory import Embedder, MemoryEntry

from agent_memories.agent.browsergym.env import make_webarena_env
from agent_memories.agent.state import AgentState
from agent_memories.config import DEFAULT_MEMORY_DIR, REPO_ROOT
from agent_memories.services.ollama_client import DEFAULT_BASE_URL

QWEN_MODEL = "qwen3.5:4b-nvfp4"
WEBARENA_USER_ID = "webarena_batch"
RUN_STATUS_OK = "ok"
RUN_STATUS_AGENT_ERROR = "agent_error"

WA_ENV_VARS = (
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
)

MASSAGE_TASK_IDS = [
    "webarena.410",
    "webarena.533",
    "webarena.561",
    "webarena.562",
    "webarena.574",
    "webarena.640",
    "webarena.680",
    "webarena.740",
]

TRAJECTORY_CSV_COLUMNS = [
    "task_id",
    "intent",
    "intent_template_id",
    "raw_trajectory",
    "final_state_yaml",
    "bot_response",
    "harness_reward",
    "harness_success",
    "run_status",
    "judge_pending",
]

MEMORY_CSV_COLUMNS = [
    "task_id",
    "intent",
    "intent_template_id",
    "raw_trajectory",
    "memory",
    "embedding",
    "judge_outcome",
    "memory_extracted",
    "harness_reward",
    "harness_success",
    "run_status",
]

# Memory-run CSV: trajectory columns plus retrieved_task_ids and
# retrieved_memory_titles.
MEMORY_RUN_CSV_COLUMNS = [
    *TRAJECTORY_CSV_COLUMNS,
    "retrieved_task_ids",
    "retrieved_memory_titles",
]

# source ("private" or "shared"), audit key (task id or DP label), entry
MemoryRecord = tuple[str, "int | str", "MemoryEntry"]

DEFAULT_TRAJECTORIES_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_A_no_memories.csv"
DEFAULT_MEMORIES_CSV = (
    REPO_ROOT / "data" / "webarena" / "trajectories_reasoningbank_private_memories.csv"
)
DEFAULT_MEMORY_RUN_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_B_private_run.csv"
DEFAULT_SHARED_RUN_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_C_shared_only.csv"
DEFAULT_PRIVATE_SHARED_RUN_CSV = (
    REPO_ROOT / "data" / "webarena" / "trajectories_D_private_shared.csv"
)
DEFAULT_SHARED_STORE = DEFAULT_MEMORY_DIR / "shared.jsonl"
DEFAULT_INFRA_LOG = REPO_ROOT / "data" / "webarena" / "infra_errors.log"
DEFAULT_MAX_STEPS = 30
# Agent Think calls with think=True can run to tens of seconds on a shared
# A30; a spurious timeout is written as agent_error, not retried as infra.
THINK_REQUEST_TIMEOUT_SECONDS = 300.0


def require_wa_env_vars() -> None:
    missing = [key for key in WA_ENV_VARS if not os.environ.get(key)]
    if missing:
        print(
            "Missing WebArena environment variables:\n  "
            + "\n  ".join(missing)
            + "\nSee scripts/webarena/instructions_spinup.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def require_nltk_punkt_tab() -> None:
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print(
            "NLTK punkt_tab is required before importing browsergym.webarena.\n"
            "Run: python -c \"import nltk; nltk.download('punkt_tab')\"\n"
            "Or: make install",
            file=sys.stderr,
        )
        sys.exit(1)


def require_ollama_model(model: str, *, base_url: str = DEFAULT_BASE_URL) -> None:
    """Fail fast when Ollama is down or the configured model is not pulled."""
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(tags_url, timeout=10.0)
        response.raise_for_status()
    except httpx.ConnectError:
        print(
            "Ollama is not running. Start it with: ollama serve",
            file=sys.stderr,
        )
        sys.exit(1)
    except httpx.HTTPError as exc:
        print(f"Ollama tags check failed ({tags_url}): {exc}", file=sys.stderr)
        sys.exit(1)

    names = [str(entry.get("name", "")) for entry in response.json().get("models", [])]
    if model in names:
        return
    if any(name.split(":")[0] == model.split(":")[0] for name in names if name):
        return

    print(
        f"Ollama model {model!r} is not loaded.\n" f"Run: ollama pull {model}",
        file=sys.stderr,
    )
    sys.exit(1)


def is_infra_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, ConnectionRefusedError)):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {61, 111}:
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "not reacheable" in msg or "not reachable" in msg:
            return True
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        return is_infra_error(cause)
    return False


def log_infra_error(log_path: Path, *, task_id: int | None, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    label = f"task_id={task_id}" if task_id is not None else "preflight"
    line = f"{stamp} [{label}] {message}\n"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(f"[Infra] {message}", file=sys.stderr, flush=True)
    print(f"[Infra] Logged to {log_path}", file=sys.stderr, flush=True)


def load_intent_template_ids() -> dict[int, int]:
    path = REPO_ROOT / "evaluation" / "task_descriptions_all.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["task_id"]): int(row["intent_template_id"]) for row in raw}


def truncate_observation(tree_yaml: str, budget: int) -> str:
    if budget <= 0 or len(tree_yaml) <= budget:
        return tree_yaml
    marker = "\n# ... <observation truncated to fit context window> ..."
    head = tree_yaml[: max(0, budget - len(marker))]
    return head + marker


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


def prepare_webarena(
    *,
    headless: bool,
    smoke: bool,
    massage: bool,
    infra_log: Path,
) -> None:
    """Single WebArena startup path: status/reset, optional smoke task-0, warm-up."""
    import browsergym.webarena  # noqa: F401  # registers tasks
    from browsergym.experiments.benchmark.utils import massage_tasks
    from browsergym.webarena.instance import WebArenaInstance

    instance = WebArenaInstance()
    try:
        if os.environ.get("WA_FULL_RESET"):
            print("[Runner] full_reset() at batch start...", flush=True)
            instance.full_reset()
        else:
            print("[Runner] check_status()...", flush=True)
            instance.check_status()
    except Exception as exc:
        if is_infra_error(exc):
            log_infra_error(
                infra_log,
                task_id=None,
                message=(
                    f"{type(exc).__name__}: {exc}. "
                    "Start WebArena Docker / webarena-setup. "
                    "See scripts/webarena/instructions_spinup.txt"
                ),
            )
            sys.exit(1)
        raise

    if smoke:
        print("[Smoke] Opening webarena.0 (reset only, no agent)...", flush=True)
        env = make_webarena_env(0, headless=headless)
        try:
            env.reset()
        except Exception as exc:
            if is_infra_error(exc):
                log_infra_error(
                    infra_log,
                    task_id=0,
                    message=f"Smoke task 0 failed: {type(exc).__name__}: {exc}",
                )
                sys.exit(1)
            raise
        finally:
            env.close()
        print("[Smoke] Task 0 env OK.", flush=True)

    if massage:
        print("[Runner] Warming up WebArena (massage_tasks)...", flush=True)
        massage_tasks(MASSAGE_TASK_IDS)

    print("[Runner] WebArena ready.", flush=True)


def load_memory_entries_from_csv(csv_path: Path) -> list[tuple[int, MemoryEntry]]:
    """Load (task_id, MemoryEntry) pairs from the memories CSV.

    Skips rows with no extracted memory or no stored embedding.
    """
    from agent_memories.memory import MemoryEntry

    if not csv_path.exists():
        print(f"Memories file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[int, MemoryEntry]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task_id_raw = row.get("task_id", "").strip()
            if not task_id_raw.isdigit():
                continue
            if row.get("memory_extracted", "").strip() != "True":
                continue
            memory_data = json.loads(row.get("memory", "{}") or "{}")
            embedding = json.loads(row.get("embedding", "[]") or "[]")
            if not memory_data or not embedding:
                continue
            entry = MemoryEntry.from_jsonl_dict(memory_data)
            entry.embedding = [float(x) for x in embedding]
            pairs.append((int(task_id_raw), entry))
    return pairs


def load_private_records(csv_path: Path) -> list[MemoryRecord]:
    """Private memories from the ReasoningBank CSV, keyed by source task_id."""
    return [
        ("private", task_id, entry) for task_id, entry in load_memory_entries_from_csv(csv_path)
    ]


def load_shared_records(store_path: Path, embedder: Embedder) -> list[MemoryRecord]:
    """Shared memories from the JSONL store, keyed by DP label."""
    from agent_memories.memory import MemoryStore

    store = MemoryStore.load(store_path, user_id="shared", embedder=embedder)
    return [("shared", entry.query, entry) for entry in store.all() if entry.embedding is not None]


class MemoryIndex:
    """Cosine top-k over private and/or shared memory records.

    Every condition retrieves the same way; only the record list differs.
    MemoryStore.search runs the same _cosine_top_k call, so shared entries
    are indexed here rather than searched through the store.
    """

    def __init__(self, records: list[MemoryRecord], embedder: Embedder) -> None:
        self._records = records
        self._embedder = embedder
        self._matrix = np.asarray(
            [entry.embedding for _, _, entry in records],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self._records)

    def search(self, intent: str, *, k: int) -> list[MemoryRecord]:
        """Return the top-k records by cosine similarity to intent."""
        if not self._records or k <= 0:
            return []
        from agent_memories.memory.store import _cosine_top_k

        query_vec = np.asarray(self._embedder.embed(intent), dtype=np.float32)
        indices = _cosine_top_k(query_vec, self._matrix, k)
        return [self._records[i] for i in indices]


def audit_ids(records: list[MemoryRecord]) -> list[int | str]:
    """Audit keys in rank order: private task ids, shared DP labels."""
    return [key for _, key, _ in records]


def flatten_records_for_think(records: list[MemoryRecord]) -> list[dict[str, str]]:
    """Flatten records to the {title, content} list Think expects."""
    flat: list[dict[str, str]] = []
    for _, _, entry in records:
        for item in entry.items:
            flat.append({"title": item.title, "content": item.content})
    return flat


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


def state_from_trajectory_row(row: dict[str, str]) -> AgentState:
    """Rebuild AgentState from a trajectories CSV row."""
    from agent_memories.agent.state import new_state

    intent = row.get("intent", "")
    state = new_state(aim=intent)
    state["memories"] = []
    raw = row.get("raw_trajectory", "").strip()
    if raw:
        state["history"] = json.loads(raw)
    else:
        state["history"] = []
    return state
