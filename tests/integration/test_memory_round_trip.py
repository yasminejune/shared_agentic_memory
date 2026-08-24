"""A memory written by one user reaches another user's Think prompt.

Wires the pieces the WebArena conditions wire - pipeline, store, MemoryIndex,
Think - with fake chat and embedding so no model or browser is needed. The unit
tests cover each piece on its own; what is checked here is the hop between them,
and in particular that user_b retrieves what user_a wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memories.agent.browsergym.nodes import make_think
from agent_memories.agent.state import AgentState, new_state
from agent_memories.memory.pipeline import MemoryPipeline
from agent_memories.memory.store import MemoryStore
from agent_memories.webarena.records import (
    MemoryIndex,
    audit_ids,
    flatten_records_for_think,
    load_shared_records,
)
from tests.conftest import FakeChatClient, FakeEmbedder

pytestmark = pytest.mark.integration

USER_A_QUERY = "find the cheapest hairbrush"
USER_B_INTENT = "find a cheap toothbrush"
SHARED_LABEL = "reporting gitlab issues"

# user_b's intent sits on user_a's axis, so cosine puts the private entry first.
VECTORS = {
    USER_A_QUERY: [1.0, 0.0, 0.0],
    USER_B_INTENT: [1.0, 0.0, 0.0],
    SHARED_LABEL: [0.0, 1.0, 0.0],
}

EXTRACTED_ITEM = (
    "# Memory Item 1\n"
    "## Title Sort by price before opening a product\n"
    "## Description The listing page sorts more reliably than the filter sidebar.\n"
    "## Content Set the sort order to price ascending on the listing page, then\n"
    "open the first result, rather than filtering by price band.\n"
)


def _observation_state(aim: str) -> AgentState:
    state = new_state(aim=aim)
    state["observation"] = {
        "url": "http://localhost:7770/",
        "title": "One Stop Market",
        "tree_yaml": "[53] button 'Search'",
    }
    return state


def test_user_a_memory_reaches_user_b_think_prompt(tmp_path: Path) -> None:
    embedder = FakeEmbedder(VECTORS)

    # user_a finishes a task and the pipeline banks what it learned.
    writer = MemoryPipeline(
        client=FakeChatClient(["Thoughts: it worked.\nStatus: success", EXTRACTED_ITEM]),
        store=MemoryStore(path=tmp_path / "user_a.jsonl", user_id="user_a", embedder=embedder),
    )
    entry = writer.create_from_run(_observation_state(USER_A_QUERY), final_state="(cart)")

    assert entry is not None, "the pipeline must bank a memory for the retrieval hop to exist"
    assert entry.user_id == "user_a"

    # A shared DP memory on an unrelated topic sits alongside it in the index.
    shared_store = MemoryStore(path=tmp_path / "shared.jsonl", user_id="shared", embedder=embedder)
    shared_store.add_entry(query=SHARED_LABEL, outcome="shared", items=entry.items)

    records = [("private", 137, entry)] + load_shared_records(shared_store.path, embedder)
    index = MemoryIndex(records, embedder)

    # user_b starts a different task and retrieves across the user boundary.
    retrieved = index.search(USER_B_INTENT, k=1)

    assert audit_ids(retrieved) == [137], "the nearer private entry must outrank the shared one"
    assert retrieved[0][2].user_id == "user_a"

    client = FakeChatClient(["click('53')"])
    state = _observation_state(USER_B_INTENT)
    state["memories"] = flatten_records_for_think(retrieved)

    make_think(client)(state)

    prompt = str(client.calls[0]["user"])
    assert "Sort by price before opening a product" in prompt
    assert "sort order to price ascending" in prompt


def test_retrieval_is_empty_for_the_no_memory_condition(tmp_path: Path) -> None:
    """Condition A builds no index, so Think sees no memory block."""
    client = FakeChatClient(["click('53')"])
    state = _observation_state(USER_B_INTENT)
    state["memories"] = flatten_records_for_think([])

    make_think(client)(state)

    assert "Sort by price" not in str(client.calls[0]["user"])
