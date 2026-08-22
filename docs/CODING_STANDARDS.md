# Coding Standards

## Setup

```bash
make install
```

Installs all production and development dependencies (pytest, black, ruff, mypy, ipython) from `pyproject.toml` and the Chromium binary Playwright drives.

### Ollama (required from WP1.3 onwards)

The WP1.3 Think node defaults to a local open-weight model served by [Ollama](https://ollama.com), so an Ollama install and a pulled model are runtime prerequisites for `scripts/langgraph/WP1_3.py` (and anything downstream that uses `agent_memories.services.ollama_client.OllamaClient`). `make install` does **not** handle this — Ollama is a brew install, not a pip package.

```bash
brew install ollama
ollama serve &                    # start the local server on http://localhost:11434
ollama pull qwen3.5:4b-nvfp4      # project default (MLX-tagged)
```


```bash
ps aux | grep 'ollama runner --mlx-engine' | grep -v grep   # should show the subprocess
curl -s http://localhost:11434/api/tags | grep qwen3.5      # should list qwen3.5:4b-nvfp4
```

### Sentence-transformers (used from WP1.5 onwards)

The WP1.5 RAG retrieval pipeline depends on a small sentence-transformers encoder (`all-MiniLM-L6-v2`, ~80 MB) bound via the `Embedder` class in `agent_memories.memory.embedder`. The model is fetched from the HuggingFace Hub the first time `scripts/memories/WP1_5.py` runs, then cached locally; no extra setup step is needed beyond `make install` (which installs `sentence-transformers` as a regular dependency).

---

## Remotes

This repo pushes to two places: **GitLab (cseegit)** — the marked repo — and **GitHub** — a public mirror. `origin` is configured to push to both at once.

### Expected `git remote -v`

After a working clone, the output should be:

```
github	https://github.com/yasminejune/shared_agentic_memory.git (fetch)
github	https://github.com/yasminejune/shared_agentic_memory.git (push)
origin	git@cseegit.essex.ac.uk:25-26-ce901-su-ce902-sp/25-26_CE901-SU_CE902-SP_frizlen_yasmine.git (fetch)
origin	git@cseegit.essex.ac.uk:25-26-ce901-su-ce902-sp/25-26_CE901-SU_CE902-SP_frizlen_yasmine.git (push)
origin	https://github.com/yasminejune/shared_agentic_memory.git (push)
```

`origin` has one fetch URL (cseegit) and two push URLs (cseegit + GitHub). Run `git remote -v` after every fresh clone to confirm the setup.

### Commands

| Command | Pushes to |
|---------|-----------|
| `git push origin <branch>` | **Both** cseegit and GitHub |
| `git push github <branch>` | GitHub only |
| `git fetch origin` | cseegit only |

The default — `git push -u origin <branch>` — updates both remotes in a single command. Use `git push github <branch>` only when you want to update the public mirror without touching cseegit.

### Why it's set up this way

- **cseegit is the marked repo.** All MRs, the squashed `master` history, and assessment happen on GitLab.
- **GitHub is a public mirror**, kept in sync so the work is visible outside Essex.
- **Never** repoint `origin` at GitHub alone — submission relies on cseegit having the latest commit on `master`.

### Re-creating the setup after a fresh clone

A clone from cseegit will only have `origin` pointing at cseegit. To restore the dual-push setup:

```bash
# Add cseegit as an explicit push URL (this overrides the implicit fetch-URL fallback)
git remote set-url --add --push origin git@cseegit.essex.ac.uk:25-26-ce901-su-ce902-sp/25-26_CE901-SU_CE902-SP_frizlen_yasmine.git

# Add GitHub as a second push URL on origin
git remote set-url --add --push origin https://github.com/yasminejune/shared_agentic_memory.git

# Add github as a separate remote for GitHub-only pushes
git remote add github https://github.com/yasminejune/shared_agentic_memory.git
```

Verify with `git remote -v` — the output should match the block above.

---

## Branch Naming

```
<ticket_number>_<brief_description>
```

Examples:
- `16175_repository_structure`
- `16180_langgraph_agent_memory`
- `16195_evaluation_pipeline`

Branch directly from `master`. One branch per ticket.

```bash
git checkout master
git pull origin master
git checkout -b 16180_your_feature_name
git push -u origin 16180_your_feature_name
```

---

## Commit Messages

Format: `[TICKET] Brief description`

For work-in-progress commits during development, a single line is fine:

```
[16175] Add shared memory store
[16175] Fix token count in summarise.py
[16175] Add unit tests for memory module
```

For squashed (final) commits, add bullet points summarising what was done:

```
[16175] Add repository structure

- Add src/ layout with __init__.py files
- Add pyproject.toml with dev dependencies
- Add coding standards and MR template
```

---

## Commit Squashing

Squash all commits on a branch into one before merging. This keeps `master` history clean and makes each ticket traceable as a single commit.

### Method 1: Interactive Rebase (Recommended)

```bash
# See how many commits to squash
git log --oneline master..HEAD

# Rebase the last N commits (replace N with the count)
git rebase -i HEAD~N
```

In the editor that opens:
- Keep the first line as `pick`
- Change all others to `squash` (or `s`)
- Save and close, then write the final commit message

```
pick a1b2c3 [16175] Initial structure
squash e4f5g6 [16175] Add tests
squash i7j8k9 [16175] Fix linting
```

### Method 2: Soft Reset (Alternative)

```bash
# Soft reset to master — keeps all changes staged
git reset --soft master

# Commit everything as one
git commit -m "[16175] Add repository structure

- Add src/ layout with __init__.py files
- Add pyproject.toml with dev dependencies
- Add coding standards and MR template"
```

### Force Push After Squashing

Squashing rewrites history, so a force push is required to update the remote branch:

```bash
git push --force-with-lease origin your-branch-name
```

⚠️ Only force push to your own feature branches. Never force push to `master`.

---

## Package Structure (`__init__.py`)

Add `__init__.py` to every folder your code imports from. Leave it empty unless you want a clean public API.

**Add it to:**
- `src/<package>/` and all sub-folders (`config/`, `services/`)
- `tests/` and all sub-folders (`unit/`, `unit/config/`, `unit/services/`, `unit/`)
- `evaluation/` and any importable sub-folders

**Do not add it to:**
- `docs/`, `scripts/`, `data/` — not imported, just files
- Project root

**Only expose a public API if needed:**
```python
# services/__init__.py — only if you want callers to write `from services import LLMService`
from .llm_service import LLMService
```

---

## Managing Dependencies

All dependencies live in `pyproject.toml`. Do not use `requirements.txt`.

**Adding a new dependency:**

1. Install it: `pip install <package>`
2. Add to `pyproject.toml`:
   - `dependencies` — needed to run the project
   - `[project.optional-dependencies] dev` — tooling only (linters, test runners)
3. Reinstall: `pip install -e ".[dev]"`

**Version pinning:**

| Syntax | When to use |
|--------|-------------|
| `>=` | Default — allows updates |
| `~=` | Minor version stability (e.g. `langgraph~=1.0.9`) |
| `==` | Only when exact version is critical |
| no pin | Avoid — unpredictable across environments |

---

## Daily Workflow

```bash
make format       # auto-fix style with black - checks "is the code formatted correctly?"
make lint         # auto-fix imports and style with ruff - checks "does the code look suspicious or sloppy (imports, names, common bugs)?"
make typecheck    # static type check with mypy - checks "do the types line up with what we claimed?"
make test         # run test suite with coverage report
make ci           # all checks, no auto-fix — use before opening an MR
```

---

## Pre-MR Checklist

```bash
make ci   # must exit with no errors before opening an MR
```

- [ ] All tests pass
- [ ] No linting errors
- [ ] Type checking passes
- [ ] New functionality has at least one test
- [ ] No secrets or credentials in the diff
- [ ] `pyproject.toml` updated if new dependencies were added
- [ ] Commits squashed into one (`git log --oneline master..HEAD` should show 1 commit)
- [ ] Branch is up to date with master (`git fetch origin && git rebase origin/master`)

---

## Merge Request Process

When work is complete, tested, and squashed into a single commit:

1. Go to GitLab → **Merge Requests** → **New merge request**
2. **Source branch:** your feature branch — **Target branch:** `master`
3. **Title:** your squashed commit title (e.g. `[16175] Add repository structure`)
4. **Description:** copy the bullet points from your squashed commit message
5. Tick **"Delete source branch when merge request is accepted"**
6. Tick **"Squash commits when merge request is accepted"** (safety net if not already squashed locally)
7. Click **Create merge request**, then merge when ready

---