"""Smoke-runs the WP2-plan §3.5 post-processing arm against local Ollama.

Given a (``--content``, ``--label``) pair on the command line --
representing the DP-released ``content`` of a shared :class:`MemoryItem`
and its round-1 label -- the script calls
:func:`agent_memories.generalisation.title_and_description` through the
project-default Qwen Ollama wrapper, prints the resulting ``(title,
description)``, and appends one JSONL record to
``scripts/amin_et_al/outputs/intermediate_memories.jsonl`` via
:func:`agent_memories.generalisation.save_intermediate`.

Prerequisites: ``ollama serve`` running on http://localhost:11434 and
``ollama pull qwen3.5:4b-nvfp4``. The full WP2 pipeline lives in
``src/agent_memories/generalisation/`` and a future
``scripts/memories/WP2_pipeline.py`` driver; this demo is the
single-call usage example for the §3.5 arm in isolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent_memories.config import load_random_seed
from agent_memories.generalisation import save_intermediate, title_and_description
from agent_memories.memory.store import MemoryItem
from agent_memories.services.ollama_client import OllamaClient

QWEN_MODEL = "qwen3.5:4b-nvfp4"

DEFAULT_OUTPUT_PATH = Path("scripts/amin_et_al/outputs/intermediate_memories.jsonl")

DEFAULT_LABEL = "attending a recent event"
DEFAULT_CONTENT = (
    "When booking event tickets through a third-party platform, confirm "
    "the venue and date on the issuer's own page before paying, because "
    "the aggregator may show stale availability."
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--content", default=DEFAULT_CONTENT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "JSONL file to append the {label,title,description,content,created_at} "
            "record to (default: scripts/amin_et_al/outputs/intermediate_memories.jsonl)."
        ),
    )
    args = parser.parse_args(argv)

    load_dotenv()
    client = OllamaClient(model=QWEN_MODEL, seed=load_random_seed())
    title, description = title_and_description(
        content=args.content,
        label=args.label,
        client=client,
    )

    print(f"Title:       {title}")
    print(f"Description: {description}")

    item = MemoryItem(title=title, description=description, content=args.content)
    save_intermediate(item, label=args.label, path=args.output)
    print(f"Appended one record to {args.output}.")


if __name__ == "__main__":
    main()
