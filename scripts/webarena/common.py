"""Shared helpers for WebArena batch scripts."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import requests

if TYPE_CHECKING:
    from agent_memories.memory import MemoryEntry

from agent_memories.agent.browsergym.env import make_webarena_env
from agent_memories.agent.state import AgentState
from agent_memories.config import REPO_ROOT
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

# Memory-augmented rerun output: the trajectory columns plus two trailing
# audit columns, so comparison against trajectories_0.csv works unchanged
# on the shared prefix.
MEMORY_RUN_CSV_COLUMNS = [
    *TRAJECTORY_CSV_COLUMNS,
    "retrieved_task_ids",
    "retrieved_memory_titles",
]

DEFAULT_TRAJECTORIES_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_0.csv"
DEFAULT_MEMORIES_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_memories.csv"
DEFAULT_MEMORY_RUN_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_private_memories.csv"
DEFAULT_INFRA_LOG = REPO_ROOT / "data" / "webarena" / "infra_errors.log"
DEFAULT_MAX_STEPS = 30


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
    import browsergym.webarena  # noqa: F401 — registers tasks
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
    """Load ``(task_id, MemoryEntry)`` pairs from the memories CSV.

    Rows without an extracted memory (``memory_extracted != True``) or
    without a stored embedding are skipped: they carry no retrievable
    content. The ``memory`` column is already in the
    :meth:`MemoryEntry.to_dict_without_embedding` schema, so hydration
    reuses :meth:`MemoryEntry.from_jsonl_dict` with the embedding
    re-attached from the ``embedding`` column — no embedder call needed.
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


def state_from_trajectory_row(row: dict[str, str]) -> AgentState:
    """Rebuild minimal :class:`AgentState` from a trajectories CSV row."""
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
