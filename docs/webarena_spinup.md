# Local WebArena via webarena-setup (BrowserGym recommended path)

## Prerequisites

- Docker with enough disk space (~200 GB+ if you include the OpenStreetMap tile stack)
- Python 3.10+ project venv: `make install` (installs browsergym-webarena, NLTK punkt_tab, Playwright Chromium)

## 1. webarena-setup

Follow https://github.com/ServiceNow/BrowserGym/tree/main/browsergym/webarena#option-2-webarena-setup
to pull the official WebArena Docker images and start all seven services.

## 2. WA_* environment variables (localhost template)

Set these before running any of the `webarena-*` commands, either in `.env` or exported in the shell (adjust host/ports if your setup differs):

  export WA_SHOPPING=http://localhost:7770
  export WA_SHOPPING_ADMIN=http://localhost:7780/admin
  export WA_REDDIT=http://localhost:9999
  export WA_GITLAB=http://localhost:8023
  export WA_MAP=http://localhost:3000
  export WA_WIKIPEDIA=http://localhost:8888
  export WA_HOMEPAGE=http://localhost:4399

The runners check all seven and exit before the first task if any is missing.

# After webarena-setup is running; verify with:
  curl http://localhost:7565/status
  export WA_FULL_RESET=http://localhost:7565

## 3. NLTK punkt_tab

Required before `import browsergym.webarena`. Installed automatically by `make install`, or run manually:

  python -c "import nltk; nltk.download('punkt_tab')"

## 4. Run trajectories, then build memories (two commands)

Step A — agent trajectories only (tasks 0–4 first):

  venv/bin/webarena-run --condition A --start-id 0 --end-id 4

Step B — judge + memory extraction + embeddings from trajectories:

  venv/bin/webarena-build-memories --start-id 0 --end-id 4

Full corpus (812 tasks, multi-day):

  venv/bin/webarena-run --condition A
  venv/bin/webarena-build-memories

--condition selects the memory condition and its output CSV:

  * A  no memories        data/webarena/trajectories_A_no_memories.csv
  * B  private            data/webarena/trajectories_B_private_run.csv
  * C  shared             data/webarena/trajectories_C_shared_only.csv
  * D  private + shared   data/webarena/trajectories_D_private_shared.csv

B and D read data/webarena/trajectories_reasoningbank_private_memories.csv, so
step B must have run first. C and D read data/memories/shared.jsonl, which the
InvisibleInk pipeline writes:

  venv/bin/python -m agent_memories.generalisation.run

The full order across all four conditions, including the deferred judge and the
comparison, is in the README.

Resume: each command skips task_ids already completed in its own output CSV.

## 5. Reset cadence

- The runner calls full_reset() once at batch start when WA_FULL_RESET is set, and check_status() when it is not.
- It then opens task 0 as a smoke test (reset only, no agent) and warms up eight known-slow tasks with massage_tasks().
- Each task afterwards gets a fresh BrowserGym env (new browser context).
- Docker DB state (shopping carts, gitlab posts, etc.) can drift across tasks. There is no mid-run reset flag, so a manual full reset between run-throughs is the way to clear it.

## 6. CSV output

Trajectories: one CSV per condition under data/webarena/, named in section 4 (gitignored).
Private memories: data/webarena/trajectories_reasoningbank_private_memories.csv (gitignored).
Shared memories: data/memories/shared.jsonl, with the Step 1–3 artefacts under data/webarena/shared_memory_run_1/ (gitignored).

Each trajectory CSV carries two sidecar files: <stem>_judge_calls.jsonl, the deferred WebArena judge calls captured during the run, and <stem>_judge_scores.csv, written by webarena-score-judge.

judge_outcome and harness_success are independent columns:
- judge_outcome: ReasoningBank LLM-as-Judge on the trajectory
- harness_success: WebArena eval harness (reward > 0 via BrowserGym)