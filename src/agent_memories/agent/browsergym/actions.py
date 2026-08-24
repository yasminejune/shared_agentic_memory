"""This script parses one BrowserGym action line out of a Qwen reply."""

from __future__ import annotations

import ast
import re

# Env is the BrowserGym environment that contains the allowed actions.
from .env import WEBARENA_ACTION_SET


class ActionParseError(ValueError):
    """This specific error is raised when the reply is not a single line, in valid python, and allowed under BrowserGym"""


_ALLOWED_FUNCTIONS = frozenset(WEBARENA_ACTION_SET.action_set.keys())

# Some Ollama/Qwen unfortunately leak <think> into the content.
# Anything between, and including <think> tags should be stripped away.
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def extract_action_line(reply: str) -> str:
    """Take one allowed action line from a Qwen reply.

    Drops any <think> block and markdown signs, then takes the last
    line that parses. If none do, return the leftover text for parse_action.
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
    kept: list[str] = []
    for line in text.splitlines():
        # Remove any markdown signs
        if line.strip().startswith("```"):
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_action(line: str) -> str:
    """Accept one allowed function call, else raise ActionParseError.

    To end the loop, the agent needs to have returned send_msg_to_user or report_infeasible.
    The string is passed to env.step unchanged.
    """
    if line is None:
        raise ActionParseError("Action line was None")
    stripped = line.strip()
    if not stripped:
        raise ActionParseError("Action line was empty")
    if "\n" in stripped:
        raise ActionParseError(f"Action must be a single line, got: {line!r}")

    try:
        # Returns an abstract syntax tree, which is a tree representation of the code.
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise ActionParseError(f"Could not parse action: {stripped!r}") from exc

    body = tree.body
    if not isinstance(body, ast.Call):
        raise ActionParseError(f"Action must be a function call, got: {stripped!r}")
    if not isinstance(body.func, ast.Name):
        raise ActionParseError(f"Action must call a bare function name, got: {stripped!r}")
    if body.func.id not in _ALLOWED_FUNCTIONS:
        raise ActionParseError(f"Not an allowed action function: {body.func.id!r}")

    return stripped
