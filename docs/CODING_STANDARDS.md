# Coding Standards

## Setup

```bash
pip install -e ".[dev]"
```

Installs all production and development dependencies (pytest, black, ruff, mypy, ipython) from `pyproject.toml`.

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
- `src/<package>/` and all sub-folders (`config/`, `services/`, `utils/`)
- `tests/` and all sub-folders (`unit/`, `unit/config/`, `unit/services/`, `unit/utils/`)
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