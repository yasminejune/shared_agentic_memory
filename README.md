# shared-memories-for-agents

Implementation of shared memories for LLM-based web agents: a LangGraph Observe-Think-Act loop that writes per-user private memories, then an InvisibleInk pipeline that releases a shared store under differential privacy, evaluated on WebArena.

## Requirements

- Python 3.10 or later (`requires-python` in `pyproject.toml`).
- A local Ollama server on `http://localhost:11434` with `qwen3.5:4b-nvfp4` pulled. That is the default Think model in the agent and WebArena runners. `make install` does not install Ollama.
- A running WebArena instance for anything under `scripts/webarena/`. Those scripts exit if the `WA_*` site URLs are missing. Docker, webarena-setup, the localhost URL template, and the status check are in `scripts/webarena/instructions_spinup.txt`.
- HuggingFace access to `google/gemma-2-2b-it` for the InvisibleInk generalisation steps (labels and shared content).
- A Mistral API key if you pass `--model mistral` or run the deferred WebArena judge.

TODO: no deploy path and no CI workflow file. Local `make ci` is the quality check.

## Install

```bash
make install
```

Creates `venv/`, installs the package editable with the `dev` extras, downloads the Chromium binary Playwright uses, and fetches NLTK `punkt_tab`. Makefile targets call `venv/bin/python` directly, so the venv does not need to be activated for `make`.

Copy `.env.example` to `.env` and fill in the keys you need. Agent and evaluation scripts also require `RANDOM_SEED` (an integer, e.g. `RANDOM_SEED=3006`). That variable is not listed in `.env.example`.

## Run

```bash
make test
```

`make ci` runs Black in check mode, Ruff, mypy, and the same pytest coverage command.

Single-page Observe-Think-Act (Playwright, default Qwen via Ollama). Needs `RANDOM_SEED`:

```bash
venv/bin/python scripts/langgraph/WP1_3.py --model qwen --url https://www.amazon.co.uk --aim "Find amazon results for paper"
```

Same loop with private-memory retrieval and a post-run ReasoningBank write:

```bash
venv/bin/python scripts/memories/WP1_5.py --model qwen --k 1
```

WebArena with no memories (tasks 0-811 by default). Needs the `WA_*` URLs, Ollama, and `RANDOM_SEED`:

```bash
venv/bin/python scripts/webarena/run_webarena_no_memories.py --start-id 0 --end-id 0
```

The other batch conditions sit next to that script: `run_webarena_with_private_memories.py`, `run_webarena_with_shared_memories.py`, and `run_webarena_with_private_shared_memories.py`. After a run, score deferred judge calls with:

```bash
venv/bin/python scripts/webarena/score_deferred_judge.py --calls <calls.jsonl>
```

InvisibleInk shared-memory pipeline (steps 1-4, in order):

```bash
venv/bin/python -m agent_memories.generalisation.run
```

## Project layout

- `src/agent_memories/` - agent graph, memory store, InvisibleInk generalisation, LLM clients.
- `scripts/` - runners: `langgraph/`, `memories/`, `webarena/`, plus `amin_et_al/` for the DP experiments.
- `evaluation/` - deferred WebArena judge scorer and task-summary helpers.
- `tests/` - pytest suite (`unit/` and `integration/`).
- `data/` - runtime outputs (`memories/`, `webarena/`); both are gitignored.
- `docs/CODING_STANDARDS.md` - local setup notes, including Ollama.

## Configuration

Runners load `.env` through `python-dotenv`. Only variables that the code actually reads:

- `MISTRAL_API_KEY` - Mistral chat client and the deferred WebArena judge.
- `GEMMA_ACCESS_TOKEN` - HuggingFace login for `google/gemma-2-2b-it`.
- `RANDOM_SEED` - integer seed for agent and evaluation scripts. Missing or non-integer values fail the run.
- `AGENT_MEMORY_USER_ID` - default simulated user id (`user_a` if unset).
- `WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`, `WA_GITLAB`, `WA_WIKIPEDIA`, `WA_MAP`, `WA_HOMEPAGE` - local WebArena site origins. Required by the batch runners. Localhost ports are in `scripts/webarena/instructions_spinup.txt`.
- `WA_FULL_RESET` - if set, call `full_reset()` on the instance at batch start; otherwise `check_status()`. The spin-up notes set this to `http://localhost:7565`.

TODO: `GITLAB_API_TOKEN` is in `.env.example` but is not read anywhere in the code.
