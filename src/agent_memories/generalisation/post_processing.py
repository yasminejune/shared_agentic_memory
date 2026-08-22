"""Title and description for a shared memory, from its DP-released content.

A normal Qwen call. The content is already private, so this step does not
spend more budget. Title is 1-10 words with no trailing punctuation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memories.memory.store import MemoryItem
from agent_memories.types import ChatClient

POST_PROC_MAX_TOKENS = 64
POST_PROC_TEMPERATURE = 0.0
POST_PROC_RETRY_TEMPERATURE = 0.0
TITLE_MAX_WORDS = 10
TITLE_MIN_WORDS = 1
FALLBACK_TITLE_WORDS = 8
_TRAILING_PUNCT = ".,;:!?"

SYSTEM_PROMPT_TEMPLATE = (
    "You will be given the content field of a shared memory item synthesised\n"
    "from one web-navigation topic (label: {label}). Write a title\n"
    "(1 to 10 words, capitalised, no trailing punctuation) and a one-sentence\n"
    "description that together identify what the content advises. Do not\n"
    "introduce any facts not present in the content. Output exactly two lines\n"
    "in this format and nothing else:\n"
    "\n"
    "Title: <title>\n"
    "Description: <description>"
)

USER_PROMPT_TEMPLATE = "Content:\n{content}"

_TITLE_LINE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)
_DESCRIPTION_LINE = re.compile(r"^Description:\s*(.+)$", re.MULTILINE)


def title_and_description(
    content: str,
    label: str,
    *,
    client: ChatClient,
) -> tuple[str, str]:
    """Produce (title, description) for a shared memory item's content.

    Title is 1-10 words with no trailing punctuation. Description is one
    sentence grounded in the content. First attempt, then retry once; if
    both fail, title is the first 8 words of content and description is
    its first sentence.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(label=label)
    user_prompt = USER_PROMPT_TEMPLATE.format(content=content)

    reply = client.chat(
        system_prompt,
        user_prompt,
        temperature=POST_PROC_TEMPERATURE,
        max_tokens=POST_PROC_MAX_TOKENS,
    )
    parsed = _parse_reply(reply)
    if parsed is not None:
        return parsed

    retry_reply = client.chat(
        system_prompt,
        user_prompt,
        temperature=POST_PROC_RETRY_TEMPERATURE,
        max_tokens=POST_PROC_MAX_TOKENS,
    )
    parsed = _parse_reply(retry_reply)
    if parsed is not None:
        return parsed

    return _fallback(content)


def save_intermediate(item: MemoryItem, *, label: str, path: Path) -> None:
    """Append one shared-memory record to an intermediate JSONL file.

    Writes a single JSON object per line. Schema is
    {label, title, description, content, created_at}.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "label": label,
        "title": item.title,
        "description": item.description,
        "content": item.content,
        "created_at": _utcnow_iso(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_reply(reply: str) -> tuple[str, str] | None:
    """Return (title, description) if reply matches the title constraints.

    Returns None when either line is missing, the title word count is
    outside [1, 10], the title ends in trailing punctuation, or the
    description is empty after stripping.
    """
    title_match = _TITLE_LINE.search(reply)
    description_match = _DESCRIPTION_LINE.search(reply)
    if title_match is None or description_match is None:
        return None
    title = title_match.group(1).strip()
    description = description_match.group(1).strip()
    if not _is_valid_title(title) or not description:
        return None
    return title, description


def _is_valid_title(title: str) -> bool:
    """True if title is 1-10 words with no trailing punctuation."""
    if not title:
        return False
    if title[-1] in _TRAILING_PUNCT:
        return False
    word_count = len(title.split())
    return TITLE_MIN_WORDS <= word_count <= TITLE_MAX_WORDS


def _fallback(content: str) -> tuple[str, str]:
    """Deterministic fallback when both LLM attempts fail to parse.

    Title is the first FALLBACK_TITLE_WORDS of content. Description is
    the first sentence of content (split on '. ', '! ', or '? ').
    """
    return _fallback_title(content), _fallback_description(content)


def _fallback_title(content: str) -> str:
    words = content.split()[:FALLBACK_TITLE_WORDS]
    if not words:
        return ""
    title = " ".join(words).rstrip(_TRAILING_PUNCT)
    if not title:
        return ""
    return title[:1].upper() + title[1:]


def _fallback_description(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    parts = re.split(r"[.!?] ", text, maxsplit=1)
    return parts[0].strip()


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
