"""Fail-fast checks and startup for the WebArena batch runners.

Every check exits the process rather than raising: these run before any
task, and a half-configured batch is not worth starting.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import requests

from agent_memories.agent.browsergym.env import make_webarena_env
from agent_memories.services.ollama_client import DEFAULT_BASE_URL

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


def require_wa_env_vars() -> None:
    missing = [key for key in WA_ENV_VARS if not os.environ.get(key)]
    if missing:
        print(
            "Missing WebArena environment variables:\n  "
            + "\n  ".join(missing)
            + "\nSee docs/webarena_spinup.md",
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
                    "See docs/webarena_spinup.md"
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
