"""Smoke harness for ``MemoryPipeline.create_from_run``.

Runs the WP1.6 ReasoningBank pipeline against a handful of hand-crafted
fake ``AgentState`` scenarios and prints the resulting :class:`MemoryEntry`
(or ``None`` when the extractor parsed zero items). Useful for
verifying that the judge correctly labels successes and failures and
that the extractor produces well-formed markdown blocks against
local Qwen and remote Mistral.

The script writes every store into a fresh :class:`tempfile.TemporaryDirectory`
so the user's real ``data/memories/`` is untouched. Exit code is ``0``
when at least one scenario produced a memory entry, ``1`` when every
scenario returned ``None`` (likely indicates a broken prompt or
parser).

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
from agent_memories.config import load_random_seed
from agent_memories.memory import Embedder, MemoryPipeline, MemoryStore
from agent_memories.services.mistral_client import MistralClient
from agent_memories.services.ollama_client import OllamaClient
from agent_memories.types import ChatClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"
MISTRAL_MODEL = "mistral-small-latest"


def _build_client(name: str, seed: int) -> ChatClient:
    if name == "qwen":
        return OllamaClient(model=QWEN_MODEL, seed=seed)
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


def _scenario_stuck_parse_failure() -> tuple[str, AgentState]:
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
    return "stuck_parse_failure", state


def _scenario_stuck_repeated_click() -> tuple[str, AgentState]:
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
    return "stuck_repeated_click", state


def _scenario_successful_search() -> tuple[str, AgentState]:
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
    return "successful_search", state


SCENARIOS = [
    _scenario_stuck_parse_failure,
    _scenario_stuck_repeated_click,
    _scenario_successful_search,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("qwen", "mistral"), default="qwen")
    args = parser.parse_args(argv)

    load_dotenv()
    embedder = Embedder()
    client = _build_client(args.model, load_random_seed())

    results: list[tuple[str, str | None]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for scenario_fn in SCENARIOS:
            name, state = scenario_fn()
            store = MemoryStore(
                path=Path(tmpdir) / f"{name}.jsonl",
                user_id="smoke_test",
                embedder=embedder,
            )

            print(f"\n=== {name} ===", flush=True)
            print(f"  aim: {state['aim']}", flush=True)
            print(f"  steps: {state['step']}, done: {state['done']}", flush=True)

            pipeline = MemoryPipeline(client=client, store=store)
            entry = pipeline.create_from_run(state, final_state="")
            if entry is None:
                print("  pipeline -> None (0 items parsed)", flush=True)
                results.append((name, None))
            else:
                print(
                    f"  pipeline -> outcome={entry.outcome} items={len(entry.items)}",
                    flush=True,
                )
                for i, item in enumerate(entry.items, start=1):
                    print(f"    item {i}: {item.title!r} -> {item.content!r}", flush=True)
                results.append((name, entry.outcome))

    none_count = sum(1 for _, outcome in results if outcome is None)
    ok_count = len(results) - none_count

    print("\n--- Summary ---", flush=True)
    for name, outcome in results:
        marker = "None" if outcome is None else "OK  "
        rendered = "None" if outcome is None else outcome
        print(f"  [{marker}] {name}: {rendered}", flush=True)
    print(
        f"\n{ok_count}/{len(results)} scenarios yielded a memory entry; "
        f"{none_count} returned None.",
        flush=True,
    )

    if ok_count == 0:
        print(
            "\n[WARNING] Pipeline returned None for every scenario. "
            "The judge or extractor prompt is likely producing unparseable output.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
