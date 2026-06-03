"""WP2-plan §3.5: title and description from DP-released content.

The WP2 round-2 Amin mechanism (§3.3) emits only the ``content`` field
of a shared ReasoningBank :class:`MemoryItem`. This module produces
the matching ``title`` and ``description`` via a standard Qwen chat
turn over the existing :class:`ChatClient` Protocol. Both inputs
(``content`` from round 2, ``label`` from round 1) are DP-released
artifacts, so by the post-processing property of differential privacy
(Dwork and Roth 2014, Proposition 2.1) this call contributes zero to
``rho_total`` -- it is off the DP path (WP2-plan §4.7).

The §3.5 prompt is reproduced verbatim and held fixed as an
experimental control variable, matching the WP1.6 convention for the
ReasoningBank Appendix A.1 prompts (`.claude/memory/decisions.md`,
Conventions). It is split into a system message (instruction block
with ``LABEL_PLACEHOLDER`` substituted) and a user message (the
``Content:`` data block), mirroring the WP1.6 judge and extractor
split in :mod:`agent_memories.memory.pipeline`.

Robustness, per §3.5: the parser accepts the two-line ``Title:`` /
``Description:`` reply, retries once at ``temperature=0`` on parse
failure or constraint violation, and falls back to ``title = first 8
words of content`` and ``description = first sentence of content`` if
the retry also fails. Both fallbacks are deterministic functions of
the DP-released ``content`` so the privacy guarantee is unaffected
(§3.5 closing paragraph).
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
    """Produce ``(title, description)`` for a shared memory item's ``content``.

    Per WP2-plan §3.5 the title is 1-10 words, capitalised, no trailing
    punctuation; the description is one sentence summarising the
    content's recommendation. Both must be grounded in ``content``
    (the prompt forbids new facts).

    Behaviour:

    1. First attempt at ``temperature=0.0``. The §3.5 output is two
       short constrained lines so the §3.5 prompt does not need
       sampling diversity; ``temperature=0.0`` matches the WP1.6
       judge convention (`.claude/memory/decisions.md`, Conventions).
    2. On parse failure or constraint violation (title word count
       outside 1-10, or trailing punctuation, or empty description),
       retry once at ``temperature=0.0`` per §3.5.
    3. If the retry also fails, return the deterministic fallback
       ``(first 8 words of content, first sentence of content)``.
       The fallback is a function of the DP-released ``content`` only
       and incurs no additional privacy cost (§3.5, §4.7).

    ``client`` is any :class:`ChatClient`. The Qwen Ollama wrapper at
    :class:`agent_memories.services.ollama_client.OllamaClient` is the
    project default per `.claude/memory/decisions.md`.
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
    """Append one shared-memory record to the §3.5 intermediate JSONL file.

    Writes a single JSON object per line to ``path`` (creating its
    parent directory if missing), matching the
    :class:`agent_memories.memory.MemoryStore` JSONL convention
    (one record per line, UTF-8, newline-terminated). The schema is
    deliberately a thin intermediate carrier --
    ``{label, title, description, content, created_at}`` -- separate
    from the eventual ``data/memories/shared.jsonl`` written by the
    full WP2 pipeline; integration with :class:`MemoryStore` is left
    to WP2.4.
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
    """Return ``(title, description)`` if ``reply`` matches §3.5 constraints.

    Returns ``None`` when either line is missing, the title's word
    count is outside ``[1, 10]``, the title ends in trailing
    punctuation, or the description is empty after stripping. The
    caller treats ``None`` as a retry trigger.
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
    """Apply the §3.5 title constraints: 1-10 words, no trailing punctuation."""
    if not title:
        return False
    if title[-1] in _TRAILING_PUNCT:
        return False
    word_count = len(title.split())
    return TITLE_MIN_WORDS <= word_count <= TITLE_MAX_WORDS


def _fallback(content: str) -> tuple[str, str]:
    """Deterministic §3.5 fallback when both LLM attempts fail to parse.

    Title is the first :data:`FALLBACK_TITLE_WORDS` whitespace-split
    tokens of ``content`` joined by single spaces, stripped of
    trailing punctuation, and with the first character upper-cased.
    Description is the first sentence of ``content``, defined as the
    text before the earliest of the boundaries ``". "``, ``"! "``,
    ``"? "`` (the §3.5 split rule). Both outputs depend only on the
    DP-released ``content`` so the privacy guarantee is unaffected.
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
