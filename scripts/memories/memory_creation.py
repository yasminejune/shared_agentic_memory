"""Smoke harness for ``MemoryPipeline.create_from_run``.

Runs the WP1.5 memory curator against a handful of hand-crafted fake
``AgentState`` scenarios and prints each verbatim reply. Useful for
verifying that the curator can in fact return non-None text on traces
that clearly contain something worth memorising -- and that ``None``
shows up on the dedup case.

The script writes every store into a fresh :class:`tempfile.TemporaryDirectory`
so the user's real ``data/memories/`` is untouched. Exit code is ``0``
when at least one scenario produced a memory, ``1`` when every scenario
returned ``None`` (likely indicates a broken prompt).

Usage:
    python scripts/memories/memory_creation.py
    python scripts/memories/memory_creation.py --model mistral
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_memories.agent.state import AgentState, new_state
from agent_memories.memory import Embedder, MemoryPipeline, MemoryStore
from agent_memories.services.mistral_client import MistralClient
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"
MISTRAL_MODEL = "mistral-small-latest"


def _build_client(name: str) -> ChatClient:
    if name == "qwen":
        return OllamaClient(model=QWEN_MODEL)
    return MistralClient(model=MISTRAL_MODEL)


def _make_state(
    *,
    aim: str,
    url: str,
    history: list[dict[str, Any]],
    steps: int,
) -> AgentState:
    """Wrap a fake history into a terminal :class:`AgentState`."""
    state = new_state(aim=aim)
    state["url"] = url
    state["history"] = history
    state["step"] = steps
    state["done"] = True
    return state


def _scenario_stuck_parse_failure() -> tuple[str, AgentState, list[str]]:
    """The model emits a two-line ``type ... \\n enter`` reply five times."""
    bad_reply = 'type [e34] "paper"\nenter'
    history: list[dict[str, Any]] = [
        {
            "step": i,
            "thought": bad_reply,
            "action": {},
            "outcome": "parse_failure: Action must be a single line",
        }
        for i in range(5)
    ]
    history.append(
        {
            "step": 5,
            "thought": "(stuck-detector aborted run)",
            "action": {},
            "outcome": "stuck: same thought repeated 5 times",
        }
    )
    state = _make_state(
        aim="Search Amazon for paper",
        url="https://www.amazon.co.uk/",
        history=history,
        steps=5,
    )
    return "stuck_parse_failure", state, []


def _scenario_stuck_repeated_click() -> tuple[str, AgentState, list[str]]:
    """The model fires the same valid click five times on an unchanging page."""
    history: list[dict[str, Any]] = [
        {
            "step": i,
            "thought": "click [e118]",
            "action": {"type": "click", "ref": "e118"},
            "outcome": "ok",
        }
        for i in range(5)
    ]
    history.append(
        {
            "step": 5,
            "thought": "(stuck-detector aborted run)",
            "action": {},
            "outcome": "stuck: same thought repeated 5 times",
        }
    )
    state = _make_state(
        aim="Submit the search on Amazon",
        url="https://www.amazon.co.uk/",
        history=history,
        steps=5,
    )
    return "stuck_repeated_click", state, []


def _scenario_successful_search() -> tuple[str, AgentState, list[str]]:
    """Clean three-step Amazon search that ends with ``stop``."""
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "thought": "click [e114]",
            "action": {"type": "click", "ref": "e114"},
            "outcome": "ok",
        },
        {
            "step": 1,
            "thought": 'type [e114] "paper"',
            "action": {"type": "fill", "ref": "e114", "value": "paper"},
            "outcome": "ok",
        },
        {
            "step": 2,
            "thought": "enter",
            "action": {"type": "press", "key": "Enter"},
            "outcome": "ok",
        },
        {
            "step": 3,
            "thought": "stop",
            "action": {"type": "stop"},
            "outcome": "stop",
        },
    ]
    state = _make_state(
        aim="Search Amazon for paper",
        url="https://www.amazon.co.uk/s?k=paper",
        history=history,
        steps=3,
    )
    return "successful_search", state, []


def _scenario_dedup_with_existing_memory() -> tuple[str, AgentState, list[str]]:
    """Same successful search, but a near-identical memory already exists.

    A well-behaved curator should look at the top-5 most similar
    existing memories and return ``None`` because nothing new is worth
    storing.
    """
    _, state, _ = _scenario_successful_search()
    seeds = [
        (
            "On Amazon, after typing a query into the search box and pressing "
            "Enter the page navigates to /s?k=<query> with the results."
        ),
    ]
    return "dedup_with_existing_memory", state, seeds


SCENARIOS = [
    _scenario_stuck_parse_failure,
    _scenario_stuck_repeated_click,
    _scenario_successful_search,
    _scenario_dedup_with_existing_memory,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    args = parser.parse_args(argv)

    load_dotenv()
    embedder = Embedder()
    client = _build_client(args.model)

    results: list[tuple[str, str | None]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for scenario_fn in SCENARIOS:
            name, state, seeds = scenario_fn()
            store = MemoryStore(
                path=Path(tmpdir) / f"{name}.jsonl",
                user_id="smoke_test",
                embedder=embedder,
            )
            if seeds:
                store.add_many(seeds)

            print(f"\n=== {name} ===", flush=True)
            print(f"  aim: {state['aim']}", flush=True)
            print(f"  steps: {state['step']}, done: {state['done']}", flush=True)
            print(f"  seeded memories: {len(seeds)}", flush=True)

            pipeline = MemoryPipeline(client=client, store=store)
            memory = pipeline.create_from_run(state)
            if memory is None:
                print("  curator -> None (no memory created)", flush=True)
                results.append((name, None))
            else:
                print(f"  curator -> {memory.text!r}", flush=True)
                results.append((name, memory.text))

    none_count = sum(1 for _, text in results if text is None)
    text_count = len(results) - none_count

    print("\n--- Summary ---", flush=True)
    for name, text in results:
        marker = "None" if text is None else "OK  "
        rendered = "None" if text is None else repr(text)
        print(f"  [{marker}] {name}: {rendered}", flush=True)
    print(
        f"\n{text_count}/{len(results)} scenarios yielded a memory; "
        f"{none_count} returned None.",
        flush=True,
    )

    if text_count == 0:
        print(
            "\n[WARNING] Curator returned None for every scenario. "
            "The system or user prompt is likely biasing the LLM toward 'None'.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
