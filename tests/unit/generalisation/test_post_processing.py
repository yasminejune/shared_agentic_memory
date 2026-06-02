"""Unit tests for the WP2-plan §3.5 post-processing arm.

A ``_FakeClient`` records every ``chat`` call and returns the next
queued reply so the tests cover prompt assembly, the retry behaviour
and the deterministic fallback without hitting Ollama. The structure
mirrors ``tests/unit/memory/test_pipeline.py``.

Covers:

* the happy path: a well-formed first reply is parsed and no retry
  fires;
* a malformed first reply triggers exactly one retry and the
  well-formed second reply is returned;
* both replies malformed triggers the deterministic ``(first 8
  words, first sentence)`` fallback;
* a too-long title on the first reply triggers a retry (constraint
  violation, not a parse failure);
* a title with trailing punctuation on the first reply triggers a
  retry;
* :func:`save_intermediate` writes one JSONL line carrying the §3.5
  schema fields and creates the parent directory if missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memories.generalisation.post_processing import (
    POST_PROC_MAX_TOKENS,
    POST_PROC_RETRY_TEMPERATURE,
    POST_PROC_TEMPERATURE,
    save_intermediate,
    title_and_description,
)
from agent_memories.memory.store import MemoryItem


class _FakeClient:
    """Chat double that returns a queued reply per call.

    Records the system + user payload and chat-time kwargs of every
    call so individual tests can assert call count and prompt
    assembly.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self._replies.pop(0)


GOOD_REPLY = "Title: Verify Venue On Issuer Page\nDescription: Confirm the venue and date on the issuer's page before paying.\n"
MALFORMED_REPLY = "Sorry, I cannot produce that output."

LABEL = "attending a recent event"
CONTENT = (
    "When booking event tickets through a third-party platform, confirm "
    "the venue and date on the issuer's own page before paying, because "
    "the aggregator may show stale availability."
)


@pytest.mark.unit
def test_success_path_parses_first_reply() -> None:
    """A well-formed first reply must be parsed without triggering a retry."""
    client = _FakeClient([GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == POST_PROC_TEMPERATURE
    assert call["max_tokens"] == POST_PROC_MAX_TOKENS
    assert LABEL in str(call["system"])
    assert CONTENT in str(call["user"])


@pytest.mark.unit
def test_retry_success_after_malformed_first_reply() -> None:
    """One malformed reply must trigger exactly one retry at ``temperature=0``."""
    client = _FakeClient([MALFORMED_REPLY, GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 2
    assert client.calls[1]["temperature"] == POST_PROC_RETRY_TEMPERATURE


@pytest.mark.unit
def test_double_failure_falls_back_deterministically() -> None:
    """If both replies are malformed, return the §3.5 deterministic fallback."""
    client = _FakeClient([MALFORMED_REPLY, MALFORMED_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "When booking event tickets through a third-party platform"
    assert description == CONTENT
    assert len(client.calls) == 2


@pytest.mark.unit
def test_fallback_description_takes_first_sentence_on_multi_sentence_content() -> None:
    """When ``content`` has multiple sentences, only the first is returned."""
    multi = "Confirm the venue first. Then pay only once. Never refresh."
    client = _FakeClient([MALFORMED_REPLY, MALFORMED_REPLY])

    _, description = title_and_description(multi, LABEL, client=client)

    assert description == "Confirm the venue first"


@pytest.mark.unit
def test_title_over_ten_words_triggers_retry() -> None:
    """A title with 11+ words violates the §3.5 constraint and must retry."""
    too_long_title = " ".join(["Word"] * 11)
    bad_reply = f"Title: {too_long_title}\nDescription: A description.\n"
    client = _FakeClient([bad_reply, GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 2


@pytest.mark.unit
def test_title_with_trailing_period_triggers_retry() -> None:
    """A title ending in punctuation violates §3.5 and must retry."""
    bad_reply = "Title: Verify Venue On Issuer Page.\nDescription: A description.\n"
    client = _FakeClient([bad_reply, GOOD_REPLY])

    title, _ = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert len(client.calls) == 2


@pytest.mark.unit
def test_empty_description_triggers_retry() -> None:
    """A blank description fails the non-empty check and must retry."""
    bad_reply = "Title: Verify Venue On Issuer Page\nDescription:    \n"
    client = _FakeClient([bad_reply, GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 2


@pytest.mark.unit
def test_fallback_title_capitalises_and_strips_trailing_punctuation() -> None:
    """The fallback title takes the first 8 words and capitalises the first letter."""
    short_content = "shop the site, then leave; never linger past checkout."
    client = _FakeClient([MALFORMED_REPLY, MALFORMED_REPLY])

    title, description = title_and_description(short_content, LABEL, client=client)

    assert title == "Shop the site, then leave; never linger past"
    assert description == short_content


@pytest.mark.unit
def test_save_intermediate_writes_schema_to_jsonl(tmp_path: Path) -> None:
    """``save_intermediate`` must append one record carrying the §3.5 schema."""
    out_path = tmp_path / "outputs" / "intermediate_memories.jsonl"
    item = MemoryItem(
        title="Verify Venue On Issuer Page",
        description="Confirm the venue and date on the issuer's page before paying.",
        content=CONTENT,
    )

    save_intermediate(item, label=LABEL, path=out_path)

    assert out_path.exists()
    line = out_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["label"] == LABEL
    assert record["title"] == "Verify Venue On Issuer Page"
    assert record["description"].startswith("Confirm the venue and date")
    assert record["content"] == CONTENT
    assert isinstance(record["created_at"], str) and record["created_at"]


@pytest.mark.unit
def test_save_intermediate_appends_subsequent_records(tmp_path: Path) -> None:
    """Two calls must produce two JSONL lines (append semantics)."""
    out_path = tmp_path / "intermediate_memories.jsonl"
    item_a = MemoryItem(title="First Title", description="First.", content="alpha")
    item_b = MemoryItem(title="Second Title", description="Second.", content="beta")

    save_intermediate(item_a, label="label-a", path=out_path)
    save_intermediate(item_b, label="label-b", path=out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["label"] == "label-a"
    assert json.loads(lines[1])["label"] == "label-b"
