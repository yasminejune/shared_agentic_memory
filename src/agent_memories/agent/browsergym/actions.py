"""Action validation for BrowserGym WebArena high-level action strings."""

from __future__ import annotations

import ast

from .env import WEBARENA_ACTION_SET


class ActionParseError(ValueError):
    """Raised when an LLM response does not match the BrowserGym action grammar."""


_ALLOWED_FUNCTIONS = frozenset(WEBARENA_ACTION_SET.action_set.keys())


def parse_action(line: str) -> str:
    """Validate a single BrowserGym WebArena action string.

    Returns the stripped action unchanged for ``env.step()``.
    Termination uses ``send_msg_to_user('...')`` or ``report_infeasible('...')`` only.
    """
    if line is None:
        raise ActionParseError("Action line was None")
    stripped = line.strip()
    if not stripped:
        raise ActionParseError("Action line was empty")
    if "\n" in stripped:
        raise ActionParseError(f"Action must be a single line, got: {line!r}")

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise ActionParseError(f"Could not parse action: {stripped!r}") from exc

    body = tree.body
    if not isinstance(body, ast.Call):
        raise ActionParseError(f"Action must be a function call, got: {stripped!r}")
    if not isinstance(body.func, ast.Name):
        raise ActionParseError(f"Action must call a bare function name, got: {stripped!r}")
    if body.func.id not in _ALLOWED_FUNCTIONS:
        raise ActionParseError(f"Disallowed action function: {body.func.id!r}")

    return stripped
