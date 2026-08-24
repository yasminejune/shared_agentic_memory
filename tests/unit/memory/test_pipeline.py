"""Judge and extractor behaviour in the ReasoningBank memory pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memories.agent.state import new_state
from agent_memories.memory.pipeline import (
    EXTRACTOR_MAX_TOKENS,
    FAILURE_SYSTEM_PROMPT,
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM_PROMPT,
    SUCCESS_SYSTEM_PROMPT,
    MemoryPipeline,
)
from agent_memories.memory.store import MemoryStore
from tests.conftest import FakeChatClient, FakeEmbedder

pytestmark = pytest.mark.unit


def _make_store(tmp_path: Path, mapping: dict[str, list[float]] | None = None) -> MemoryStore:
    return MemoryStore(
        path=tmp_path / "user_a.jsonl",
        user_id="user_a",
        embedder=FakeEmbedder(mapping),
    )


SUCCESS_TWO_ITEMS = (
    "# Memory Item 1\n"
    "## Title Use the labelled search input\n"
    "## Description Find the primary search box by accessible label first.\n"
    "## Content On a shopping site, the main search input is the one whose label\n"
    "names the site; submitting via Enter is the most reliable path.\n"
    "\n"
    "# Memory Item 2\n"
    "## Title Dismiss cookie banners before clicking results\n"
    "## Description Consent overlays intercept the first click on a tile.\n"
    "## Content Dismiss or accept the EU cookie banner before clicking the first\n"
    "result tile so the click is not absorbed by the overlay.\n"
)

FAILURE_ONE_ITEM = (
    "# Memory Item 1\n"
    "## Title Avoid retrying identical clicks when the page state has not changed\n"
    "## Description Repeating an action without an observable change wastes the loop.\n"
    "## Content When the same click produces no new observation, switch strategy\n"
    "rather than retry; the stuck-detector will otherwise abort the run.\n"
)


def test_judge_success_invokes_success_extractor(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"shop the site": [1.0, 0.0, 0.0]})
    client = FakeChatClient(
        [
            "Thoughts: looks fine to me.\nStatus: success",
            SUCCESS_TWO_ITEMS,
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    state = new_state(aim="shop the site")
    state["done"] = True

    entry = pipeline.create_from_run(state, final_state="(page)")

    assert entry is not None
    assert entry.outcome == "successful"
    assert entry.query == "shop the site"
    assert entry.user_id == "user_a"
    assert len(entry.items) == 2
    assert entry.items[0].title.startswith("Use the labelled")
    assert "Enter is the most reliable" in entry.items[0].content
    assert client.calls[1]["system"] == SUCCESS_SYSTEM_PROMPT


def test_judge_failure_invokes_failure_extractor(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"shop the site": [1.0, 0.0, 0.0]})
    client = FakeChatClient(
        [
            "Thoughts: the agent loop never advanced.\nStatus: failure",
            FAILURE_ONE_ITEM,
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    state = new_state(aim="shop the site")

    entry = pipeline.create_from_run(state, final_state="")

    assert entry is not None
    assert entry.outcome == "failed"
    assert len(entry.items) == 1
    assert client.calls[1]["system"] == FAILURE_SYSTEM_PROMPT


def test_malformed_judge_defaults_to_failure(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"do thing": [1.0, 0.0, 0.0]})
    client = FakeChatClient(
        [
            "I cannot tell from this trajectory.",
            FAILURE_ONE_ITEM,
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    entry = pipeline.create_from_run(new_state(aim="do thing"), final_state="")

    assert entry is not None
    assert entry.outcome == "failed"
    assert client.calls[1]["system"] == FAILURE_SYSTEM_PROMPT


def test_extractor_parses_three_items(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"do thing": [1.0, 0.0, 0.0]})
    three_items = (
        "# Memory Item 1\n"
        "## Title A\n## Description a desc.\n## Content a content.\n\n"
        "# Memory Item 2\n"
        "## Title B\n## Description b desc.\n## Content b content.\n\n"
        "# Memory Item 3\n"
        "## Title C\n## Description c desc.\n## Content c content.\n"
    )
    client = FakeChatClient(
        [
            "Thoughts: ok\nStatus: success",
            three_items,
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    entry = pipeline.create_from_run(new_state(aim="do thing"), final_state="")

    assert entry is not None
    assert [it.title for it in entry.items] == ["A", "B", "C"]
    assert [it.content for it in entry.items] == ["a content.", "b content.", "c content."]


def test_extractor_drops_malformed_block(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"do thing": [1.0, 0.0, 0.0]})
    broken_middle = (
        "# Memory Item 1\n"
        "## Title A\n## Description a desc.\n## Content a content.\n\n"
        "# Memory Item 2\n"
        "## Title B\n## Content b content.\n\n"  # no Description -> drop
        "# Memory Item 3\n"
        "## Title C\n## Description c desc.\n## Content c content.\n"
    )
    client = FakeChatClient(
        [
            "Thoughts: ok\nStatus: success",
            broken_middle,
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    entry = pipeline.create_from_run(new_state(aim="do thing"), final_state="")

    assert entry is not None
    assert [it.title for it in entry.items] == ["A", "C"]


def test_extractor_caps_at_three_items(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"do thing": [1.0, 0.0, 0.0]})
    four_items = "\n".join(
        f"# Memory Item {i + 1}\n" f"## Title T{i + 1}\n## Description d.\n## Content c.\n"
        for i in range(4)
    )
    client = FakeChatClient(["Thoughts: ok\nStatus: success", four_items])
    pipeline = MemoryPipeline(client=client, store=store)

    entry = pipeline.create_from_run(new_state(aim="do thing"), final_state="")

    assert entry is not None
    assert len(entry.items) == 3
    assert [it.title for it in entry.items] == ["T1", "T2", "T3"]


def test_zero_parsed_items_returns_none(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"do thing": [1.0, 0.0, 0.0]})
    client = FakeChatClient(
        [
            "Thoughts: ok\nStatus: success",
            "Sorry, I cannot extract any items from this trajectory.",
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    result = pipeline.create_from_run(new_state(aim="do thing"), final_state="")

    assert result is None
    assert len(store) == 0
    assert not store.path.exists()


def test_disk_record_carries_query_outcome_items_and_user_id(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"do thing": [1.0, 0.0, 0.0]})
    client = FakeChatClient(["Thoughts: ok\nStatus: success", SUCCESS_TWO_ITEMS])
    pipeline = MemoryPipeline(client=client, store=store)

    pipeline.create_from_run(new_state(aim="do thing"), final_state="")

    raw = store.path.read_text(encoding="utf-8").strip()
    record = json.loads(raw)
    assert record["user_id"] == "user_a"
    assert record["query"] == "do thing"
    assert record["outcome"] == "successful"
    assert isinstance(record["items"], list)
    assert len(record["items"]) == 2
    assert record["items"][0]["title"].startswith("Use the labelled")
    assert record["embedding"] == [1.0, 0.0, 0.0]


def test_judge_prompt_assembly(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"buy shoes": [1.0, 0.0, 0.0]})
    client = FakeChatClient(["Thoughts: ok\nStatus: success", SUCCESS_TWO_ITEMS])
    pipeline = MemoryPipeline(client=client, store=store)

    state = new_state(aim="buy shoes")
    state["history"] = [
        {"step": 0, "thought": "click [e1]", "action": {"type": "click"}, "outcome": "ok"},
        {"step": 1, "thought": "stop", "action": {"type": "stop"}, "outcome": "stop"},
    ]

    pipeline.create_from_run(state, final_state="- main\n  - heading 'Done'")

    judge_call = client.calls[0]
    assert judge_call["system"] == JUDGE_SYSTEM_PROMPT
    assert judge_call["temperature"] == 0.0
    assert judge_call["max_tokens"] == JUDGE_MAX_TOKENS
    user = judge_call["user"]
    assert isinstance(user, str)
    assert "User Intent: buy shoes" in user
    assert "'click [e1]' -> ok" in user
    assert "'stop' -> stop" in user
    assert "- main\n  - heading 'Done'" in user
    assert "Bot response to the user: N/A" in user


def test_extractor_call_uses_temperature_one_and_extractor_budget(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"buy shoes": [1.0, 0.0, 0.0]})
    client = FakeChatClient(["Thoughts: ok\nStatus: success", SUCCESS_TWO_ITEMS])
    pipeline = MemoryPipeline(client=client, store=store)

    pipeline.create_from_run(new_state(aim="buy shoes"), final_state="")

    extractor_call = client.calls[1]
    assert extractor_call["temperature"] == 1.0
    assert extractor_call["max_tokens"] == EXTRACTOR_MAX_TOKENS
    user = extractor_call["user"]
    assert isinstance(user, str)
    assert user.startswith("Query: buy shoes")
    assert "Trajectory:" in user


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Thoughts: x\nStatus: success", "successful"),
        ("Thoughts: x\nStatus: SUCCESS", "successful"),
        ('Thoughts: x\nStatus: "success"', "successful"),
        ("Thoughts: x\nStatus: failure", "failed"),
        ("nonsense", "failed"),
    ],
)
def test_judge_status_parser_variants(tmp_path: Path, reply: str, expected: str) -> None:
    store = _make_store(tmp_path, {"x": [1.0, 0.0, 0.0]})
    client = FakeChatClient([reply, FAILURE_ONE_ITEM])
    pipeline = MemoryPipeline(client=client, store=store)

    entry = pipeline.create_from_run(new_state(aim="x"), final_state="")

    assert entry is not None
    assert entry.outcome == expected


def test_build_from_run_does_not_write_jsonl(tmp_path: Path) -> None:
    store = _make_store(tmp_path, {"buy shoes": [1.0, 0.0, 0.0]})
    store_path = tmp_path / "user_a.jsonl"
    client = FakeChatClient(
        [
            "Thoughts: ok.\nStatus: success",
            FAILURE_ONE_ITEM,
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    result = pipeline.build_from_run(
        new_state(aim="buy shoes"),
        final_state="",
        user_id="webarena_batch",
        embedder=store.embedder,
    )

    assert result.memory_extracted is True
    assert result.entry is not None
    assert result.judge_outcome == "successful"
    assert result.entry.embedding == [1.0, 0.0, 0.0]
    assert "embedding" not in result.entry.to_dict_without_embedding()
    assert not store_path.exists()


def test_build_from_run_splits_judge_from_extraction(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    client = FakeChatClient(
        [
            "Thoughts: ok.\nStatus: success",
            "Sorry, I cannot extract any items from this trajectory.",
        ]
    )
    pipeline = MemoryPipeline(client=client, store=store)

    result = pipeline.build_from_run(
        new_state(aim="buy shoes"),
        final_state="",
        user_id="webarena_batch",
        embedder=store.embedder,
    )

    assert result.judge_outcome == "successful"
    assert result.memory_extracted is False
    assert result.entry is None
