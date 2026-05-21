.PHONY: help install format lint typecheck test ci

## help: print this help message
help:
	@grep -E '^## [a-z]' Makefile | sed 's/## /  make /'

## install: install the project and all dev dependencies
install:
	pip install -e ".[dev]"

## format: auto-format source code with Black
format:
	python -m black evaluation/ scripts/ src/

## lint: lint and auto-fix with Ruff (style, imports, naming)
lint:
	python -m ruff check evaluation/ scripts/ src/ --fix

## typecheck: run mypy static type checker
typecheck:
	python -m mypy evaluation/

## test: run the test suite with coverage report
test:
	python -m pytest --cov=evaluation --cov-report=term-missing

## ci: run all quality checks without auto-fixing (use in CI pipelines)
ci:
	python -m black --check evaluation/ scripts/
	python -m ruff check evaluation/ scripts/
	python -m mypy evaluation/
	python -m pytest --cov=evaluation --cov-report=term-missing
