"""Action validation for BrowserGym WebArena high-level action strings."""

from __future__ import annotations

import ast
import re

from .env import WEBARENA_ACTION_SET


class ActionParseError(ValueError):
    """Raised when an LLM response does not match the BrowserGym action grammar."""


_ALLOWED_FUNCTIONS = frozenset(WEBARENA_ACTION_SET.action_set.keys())

# Native thinking usually arrives in message.thinking, but some model /
# Ollama combinations leak a <think> block into content. Drop both the
# closed and the unterminated form (budget ran out mid-trace).
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def extract_action_line(reply: str) -> str:
    """Pull a single BrowserGym action line out of a talkative reply.

    Order: drop a leaked ``<think>`` block, strip Markdown fence marker
    lines, then return the last remaining line that validates as an
    allowed call. If none validate, return the stripped text so
    :func:`parse_action` raises with the same shape of error as today.
    """
    if reply is None:
        return ""
    text = _THINK_BLOCK_RE.sub("", reply).strip()
    text = _strip_markdown_fence_lines(text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return parse_action(line)
        except ActionParseError:
            continue
    return text


def _strip_markdown_fence_lines(text: str) -> str:
    """Drop lines that are only a Markdown fence opener or closer."""
    kept: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        kept.append(line)
    return "\n".join(kept)


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
