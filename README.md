# Differentially private memories for agents

This repo is an implementation of shared memories for LLM-based web agents. The agent is built on a LangGraph Observe-Think-Act loop that writes per-user private memories with ReasoningBank, upon which an InvisibleInk pipeline is built that releases a shared store under differential privacy, evaluated on WebArena.

## Repository layout

The repository separates the system the thesis proposes from the work that arrived at it.

```
.
├── src/agent_memories/          # the package
│   ├── agent/
│   │   ├── invisible_ink/       # the InvisibleInk mechanism and its zCDP accounting
│   │   ├── browsergym/          # the ReAct graph and nodes used for the evaluation
│   │   ├── playwright/          # the same loop over a live page, no WebArena
│   │   ├── lm/                  # prompts and raw token-logit generation
│   │   └── amin_et_al/          # the superseded Amin et al. privatisation
│   ├── generalisation/          # shared memory creation, Steps 1-4
│   ├── memory/                  # memory store, embedder, ReasoningBank pipeline
│   ├── webarena/                # trajectory records, CSV io, preflight, memory builder
│   ├── services/                # Ollama and Mistral clients
│   └── config/                  # repo paths, seeding
├── evaluation/
│   └── webarena/                # scenario runner, deferred judge, scenario comparison
├── scripts/                     # the project experiments while building it
│   ├── langgraph/
│   ├── playwright/
│   ├── qwen/
│   ├── gemma/
│   ├── amin_et_al/
│   ├── labels/
│   ├── memories/
│   └── generalisation/
├── tests/
│   ├── unit/
│   └── integration/
├── data/                        # runtime outputs, gitignored
│   ├── memories/                # per-user private stores and shared.jsonl
│   └── webarena/                # trajectory CSVs and the Step 1-3 artefacts
├── docs/
│   ├── CODING_STANDARDS.md      # branch naming, commits, remotes, local model setup
│   └── webarena_spinup.md       # Docker, webarena-setup, the WA_* template
├── Makefile                     # install, format, lint, typecheck, test, ci
├── pyproject.toml               # dependencies, the four entry points, tool configuration
├── CITATION.cff
├── LICENSE                      # GPL-3.0-only
└── THIRD_PARTY_NOTICES.md
```

`scripts/` is the working record of the project rather than a part of the system. Each directory is one strand of the build-up, and the files in it are the experiments that produced the decision for the thesis.

## Requirements

- Python 3.10 or later (`requires-python` in `pyproject.toml`).
- A local Ollama server on `http://localhost:11434` with `qwen3.5:4b-nvfp4` pulled. That is the Think model for the agent and for every other non-private call in the system, and `make install` does not set it up.
- A running WebArena instance for the `webarena-*` commands. They exit if the `WA_*` site URLs are missing. Docker, webarena-setup, the localhost URL template and the status check are in `docs/webarena_spinup.md`.
- HuggingFace access to `google/gemma-2-2b-it`. The differentially private generation reads its next-token logits directly off the forward pass, which is why this path runs a locally loaded model rather than an endpoint.
- A Mistral API key for the deferred WebArena judge, and for `--model mistral` on the single-page scripts.

## Install

```bash
make install
```

Creates `venv/`, installs the package editable with the `dev` extras, downloads the Chromium binary Playwright drives, and fetches NLTK `punkt_tab`. Makefile targets call `venv/bin/python` directly, so the venv does not need to be activated for `make`.

Ollama is a brew install rather than a pip package, so `make install` does not touch it. Install it, start the server, and pull the Think model:

```bash
brew install ollama
ollama serve &
ollama pull qwen3.5:4b-nvfp4
```

The server has to be running for every command below except `webarena-score-judge`, which calls Mistral, and `webarena-compare`, which only reads CSVs. `webarena-run` and `webarena-build-memories` check the server and the model before the first task and exit with the command to run if either is missing. The single-page scripts and the generalisation pipeline have no such check and fail on their first call instead. `docs/CODING_STANDARDS.md` has the commands that verify the server is up and serving the model.

Copy `.env.example` to `.env` and fill in the three keys it lists. `RANDOM_SEED` is read by every agent and evaluation entry point, and a missing or non-integer value fails the run rather than falling back to a default.

## Reproducing the evaluation

The four scenarios are not independent. Scenario A produces the trajectories the private memories are distilled from, and those private memories are what the shared store is synthesised from, so the steps below run in order. Each run-through covers all 812 WebArena tasks and takes several days on one machine.

```bash
# 1. Condition A, the no-memory baseline
venv/bin/webarena-run --condition A

# 2. Private memories: LLM-as-judge, ReasoningBank extraction, embeddings
venv/bin/webarena-build-memories

# 3. Shared memories: InvisibleInk Steps 1-4, writing data/memories/shared.jsonl
venv/bin/python -m agent_memories.generalisation.run

# 4. The three memory conditions
venv/bin/webarena-run --condition B    # private only
venv/bin/webarena-run --condition C    # shared only
venv/bin/webarena-run --condition D    # private and shared

# 5. Score the deferred judge calls, once per condition
venv/bin/webarena-score-judge --calls data/webarena/trajectories_A_no_memories_judge_calls.jsonl

# 6. Compare the four conditions on the 812-task success rate
venv/bin/webarena-compare
```

Each condition writes its own CSV under `data/webarena/`, with the judge calls and judge scores as separate files next to it. `--start-id` and `--end-id` narrow a run (0 to 811 by default), which is how to test a handful of tasks before committing to a full pass. `--k` sets how many memories are retrieved (default 3, ignored by A). A run skips tasks already recorded as `ok` in its own CSV, so an interrupted run resumes rather than restarts.

The four commands are declared in `[project.scripts]` and installed by `make install`. They name the supported entry points among the forty-odd Python files here, and each one pins `venv/bin/python` in its shebang, so a run cannot pick up a system interpreter with a different dependency set. Each is also runnable as a module, which needs no reinstall after the entry point changes:

| Command | Module |
| --- | --- |
| `webarena-run` | `evaluation.webarena.run` |
| `webarena-build-memories` | `agent_memories.webarena.build_memories` |
| `webarena-score-judge` | `evaluation.webarena.judge.deferred` |
| `webarena-compare` | `evaluation.webarena.compare` |

## Running the agent on a single page

Neither script needs WebArena. Both drive a live Chromium page through Playwright, and both need `RANDOM_SEED`.

Observe-Think-Act only:

```bash
venv/bin/python scripts/langgraph/WP1_3.py --model qwen --url https://www.amazon.co.uk --aim "Find amazon results for paper"
```

The same loop with private-memory retrieval and a ReasoningBank write after the run:

```bash
venv/bin/python scripts/memories/WP1_5.py --model qwen --k 1
```

## Quality checks

```bash
make test    # pytest with a line and branch coverage report
make ci      # Black in check mode, Ruff, mypy, then the same pytest command
```

`make ci` is the only quality gate. There is no CI workflow file and no deploy path, so the merge request checklist names the local command instead. `make help` lists the remaining targets.

## Configuration

Runners load `.env` through `python-dotenv`. Only the variables the code actually reads:

- `RANDOM_SEED` - integer seed for the agent and evaluation scripts.
- `MISTRAL_API_KEY` - Mistral chat client and the deferred WebArena judge.
- `GEMMA_ACCESS_TOKEN` - HuggingFace login for `google/gemma-2-2b-it`.
- `AGENT_MEMORY_USER_ID` - simulated user id for the private memory store (`user_a` if unset).
- `WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`, `WA_GITLAB`, `WA_WIKIPEDIA`, `WA_MAP`, `WA_HOMEPAGE` - local WebArena site origins. Required by the batch runners. Localhost ports are in `docs/webarena_spinup.md`.
- `WA_FULL_RESET` - if set, call `full_reset()` on the instance at batch start; otherwise `check_status()`. The spin-up notes set this to `http://localhost:7565`.

## Licence and third-party software

GPL-3.0-only, see `LICENSE`. The licence is set by `invink`, which supplies the
differential privacy primitives and is itself GPL-3.0.

`THIRD_PARTY_NOTICES.md` records which parts come from `invink` and which are
original, the Gemma 2 use terms, and the licences of the other dependencies.

## Citing

See `CITATION.cff`. Work that uses the differentially private generation should
also cite the method it implements:

> Vishnu Vinod, Krishna Pillutla and Abhradeep Guha Thakurta. InvisibleInk:
> High-Utility and Low-Cost Text Generation with Differential Privacy. NeurIPS,
> 2025. arXiv:2507.02974.