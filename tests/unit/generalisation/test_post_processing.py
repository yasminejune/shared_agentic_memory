"""Titles and descriptions derived from shared memory content."""

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
from tests.conftest import FakeChatClient

pytestmark = pytest.mark.unit


GOOD_REPLY = "Title: Verify Venue On Issuer Page\nDescription: Confirm the venue and date on the issuer's page before paying.\n"
MALFORMED_REPLY = "Sorry, I cannot produce that output."

LABEL = "attending a recent event"
CONTENT = (
    "When booking event tickets through a third-party platform, confirm "
    "the venue and date on the issuer's own page before paying, because "
    "the aggregator may show stale availability."
)


def test_parses_first_reply() -> None:
    client = FakeChatClient([GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == POST_PROC_TEMPERATURE
    assert call["max_tokens"] == POST_PROC_MAX_TOKENS
    assert LABEL in str(call["system"])
    assert CONTENT in str(call["user"])


def test_retries_after_malformed_reply() -> None:
    client = FakeChatClient([MALFORMED_REPLY, GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 2
    assert client.calls[1]["temperature"] == POST_PROC_RETRY_TEMPERATURE


def test_falls_back_after_two_failures() -> None:
    client = FakeChatClient([MALFORMED_REPLY, MALFORMED_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "When booking event tickets through a third-party platform"
    assert description == CONTENT
    assert len(client.calls) == 2


def test_fallback_description_is_first_sentence() -> None:
    multi = "Confirm the venue first. Then pay only once. Never refresh."
    client = FakeChatClient([MALFORMED_REPLY, MALFORMED_REPLY])

    _, description = title_and_description(multi, LABEL, client=client)

    assert description == "Confirm the venue first"


def test_title_over_ten_words_triggers_retry() -> None:
    too_long_title = " ".join(["Word"] * 11)
    bad_reply = f"Title: {too_long_title}\nDescription: A description.\n"
    client = FakeChatClient([bad_reply, GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 2


def test_title_with_trailing_period_triggers_retry() -> None:
    bad_reply = "Title: Verify Venue On Issuer Page.\nDescription: A description.\n"
    client = FakeChatClient([bad_reply, GOOD_REPLY])

    title, _ = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert len(client.calls) == 2


def test_empty_description_triggers_retry() -> None:
    bad_reply = "Title: Verify Venue On Issuer Page\nDescription:    \n"
    client = FakeChatClient([bad_reply, GOOD_REPLY])

    title, description = title_and_description(CONTENT, LABEL, client=client)

    assert title == "Verify Venue On Issuer Page"
    assert description.startswith("Confirm the venue and date")
    assert len(client.calls) == 2


def test_fallback_title_capitalises_and_strips_punctuation() -> None:
    short_content = "shop the site, then leave; never linger past checkout."
    client = FakeChatClient([MALFORMED_REPLY, MALFORMED_REPLY])

    title, description = title_and_description(short_content, LABEL, client=client)

    assert title == "Shop the site, then leave; never linger past"
    assert description == short_content


def test_save_intermediate_writes_jsonl(tmp_path: Path) -> None:
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


def test_save_intermediate_appends(tmp_path: Path) -> None:
    out_path = tmp_path / "intermediate_memories.jsonl"
    item_a = MemoryItem(title="First Title", description="First.", content="alpha")
    item_b = MemoryItem(title="Second Title", description="Second.", content="beta")

    save_intermediate(item_a, label="label-a", path=out_path)
    save_intermediate(item_b, label="label-b", path=out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["label"] == "label-a"
    assert json.loads(lines[1])["label"] == "label-b"
