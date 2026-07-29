.PHONY: help venv install format lint typecheck test ci

# Use the project's venv for every target so that pre-commit hooks,
# CI runners and supervisor machines do not depend on the caller
# having `source venv/bin/activate`d. Override with `make PYTHON=...`
# to point at a different interpreter.
PYTHON ?= venv/bin/python

## help: print this help message
help:
	@grep -E '^## [a-z]' Makefile | sed 's/## /  make /'

## venv: create the project virtualenv if it does not yet exist
venv:
	@test -x $(PYTHON) || python3 -m venv venv

## install: install the project, dev dependencies and the Chromium binary Playwright drives
install: venv
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m playwright install chromium
	$(PYTHON) -c "import nltk; nltk.download('punkt_tab')"

## format: auto-format source code with Black
format:
	$(PYTHON) -m black src/ evaluation/ scripts/ tests/

## lint: lint and auto-fix with Ruff (style, imports, naming)
lint:
	$(PYTHON) -m ruff check src/ evaluation/ scripts/ tests/ --fix

## typecheck: run mypy static type checker on first-party code
typecheck:
	$(PYTHON) -m mypy src/agent_memories/ evaluation/

## test: run the full test suite (unit + integration) with coverage report
test:
	$(PYTHON) -m pytest --cov=src/agent_memories --cov=evaluation --cov-report=term-missing

## ci: run all quality checks without auto-fixing (use in CI pipelines)
ci:
	$(PYTHON) -m black --check src/ evaluation/ scripts/ tests/
	$(PYTHON) -m ruff check src/ evaluation/ scripts/ tests/
	$(PYTHON) -m mypy src/agent_memories/ evaluation/
	$(PYTHON) -m pytest --cov=src/agent_memories --cov=evaluation --cov-report=term-missing
