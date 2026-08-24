"""Model, batch user id, CSV schemas, and the default WebArena paths.

One place for every column list and every data/webarena/ path, so the
runners, the memory builder and the comparison all read the same layout.
"""

from __future__ import annotations

from pathlib import Path

from agent_memories.config import DEFAULT_MEMORY_DIR, REPO_ROOT

QWEN_MODEL = "qwen3.5:4b-nvfp4"
WEBARENA_USER_ID = "webarena_batch"

# The 812 WebArena tasks, ids 0-811.
N_TASKS = 812

RUN_STATUS_OK = "ok"
RUN_STATUS_AGENT_ERROR = "agent_error"

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

WEBARENA_DATA_DIR = REPO_ROOT / "data" / "webarena"

DEFAULT_TRAJECTORIES_CSV = WEBARENA_DATA_DIR / "trajectories_A_no_memories.csv"
DEFAULT_MEMORY_RUN_CSV = WEBARENA_DATA_DIR / "trajectories_B_private_run.csv"
DEFAULT_SHARED_RUN_CSV = WEBARENA_DATA_DIR / "trajectories_C_shared_only.csv"
DEFAULT_PRIVATE_SHARED_RUN_CSV = WEBARENA_DATA_DIR / "trajectories_D_private_shared.csv"
DEFAULT_MEMORIES_CSV = WEBARENA_DATA_DIR / "trajectories_reasoningbank_private_memories.csv"
DEFAULT_SHARED_STORE = DEFAULT_MEMORY_DIR / "shared.jsonl"
DEFAULT_INFRA_LOG = WEBARENA_DATA_DIR / "infra_errors.log"

# Condition label -> the trajectory CSV it writes, in report order.
CONDITION_CSVS: tuple[tuple[str, Path], ...] = (
    ("A", DEFAULT_TRAJECTORIES_CSV),
    ("B", DEFAULT_MEMORY_RUN_CSV),
    ("C", DEFAULT_SHARED_RUN_CSV),
    ("D", DEFAULT_PRIVATE_SHARED_RUN_CSV),
)

DEFAULT_MAX_STEPS = 30
# Agent Think calls with think=True can run to tens of seconds on a shared
# A30; a spurious timeout is written as agent_error, not retried as infra.
THINK_REQUEST_TIMEOUT_SECONDS = 300.0
