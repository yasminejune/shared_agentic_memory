"""Step 3: InvisibleInk content generation for qualifying label buckets.

Generates the content field only. Markdown title headers would spend
privacy budget, so they are not produced here. Accounting b is the gate
threshold (default 7). Large buckets are split into several InvisibleInk
runs: while at least 2b members remain, a prefix of b is peeled off;
the last run takes the rest (size b ... 2b-1). Writes one contents.jsonl
row per chunk.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from agent_memories.agent.invisible_ink import generate_with_oom_fallback
from agent_memories.agent.invisible_ink.accounting import InvisibleInkAccount
from agent_memories.agent.privacy.prompts import wrap
from agent_memories.generalisation.buffer import entry_id
from agent_memories.generalisation.io import (
    BUCKET_SIZE,
    DELTA,
    EPSILON,
    GEMMA_CHUNK_SIZE,
    STEP3_MAX_TOKENS,
    TAU,
    TOP_K,
    account_to_dict,
    add_io_arguments,
    assignments_path,
    contents_path,
    entry_from_payload,
    read_json,
    render_item_block,
    resolve_io_paths,
    write_jsonl,
)

GenerateFn = Callable[..., tuple[str, InvisibleInkAccount, str]]
T = TypeVar("T")


def _chunks(entries: Sequence[T], b: int) -> list[list[T]]:
    """Split entries into InvisibleInk batches of size about b.

    Prefix chunks of b while at least 2b remain; the last chunk always
    takes the rest (size in [b, 2b-1]). Remainders are not carried over.
    """
    chunks: list[list[T]] = []
    rest = list(entries)
    while len(rest) >= 2 * b:
        chunks.append(rest[:b])
        rest = rest[b:]
    if rest:
        chunks.append(rest)
    return chunks


def run_content_generation(
    qualifying: Sequence[dict[str, Any]],
    *,
    accounting_b: int = BUCKET_SIZE,
    epsilon: float = EPSILON,
    delta: float = DELTA,
    tau: float = TAU,
    top_k: int = TOP_K,
    max_total_tokens: int = STEP3_MAX_TOKENS,
    chunk_size: int = GEMMA_CHUNK_SIZE,
    generate_fn: GenerateFn = generate_with_oom_fallback,
) -> list[dict[str, Any]]:
    """Generate one content string per InvisibleInk chunk."""
    records: list[dict[str, Any]] = []
    for bucket in qualifying:
        label = str(bucket["label"])
        entries = [entry_from_payload(payload) for payload in bucket["entries"]]
        if len(entries) < accounting_b:
            raise ValueError(
                f"Qualifying bucket for {label!r} has {len(entries)} entries; "
                f"need >= accounting_b={accounting_b}."
            )
        for chunk_index, chunk in enumerate(_chunks(entries, accounting_b)):
            texts = [render_item_block(entry) for entry in chunk]
            raw, account, engine = generate_fn(
                texts,
                b=accounting_b,
                tau=tau,
                top_k=top_k,
                max_total_tokens=max_total_tokens,
                target_epsilon=epsilon,
                delta=delta,
                chunk_size=chunk_size,
                wrap_fn=wrap,
                label=label,
            )
            records.append(
                {
                    "label_index": bucket["label_index"],
                    "label": label,
                    "content": raw,
                    "engine": engine,
                    "account": account_to_dict(account),
                    "entry_ids": [entry_id(entry) for entry in chunk],
                    "actual_batch_size": len(chunk),
                    "accounting_b": accounting_b,
                    "chunk_index": chunk_index,
                }
            )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_io_arguments(parser)
    parser.add_argument("--accounting-b", type=int, default=BUCKET_SIZE)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-tokens", type=int, default=STEP3_MAX_TOKENS)
    parser.add_argument("--chunk-size", type=int, default=GEMMA_CHUNK_SIZE)
    args = parser.parse_args(argv)

    _, work_dir = resolve_io_paths(args)
    artefact = read_json(assignments_path(work_dir))
    qualifying = list(artefact.get("qualifying", []))
    print(f"[Step 3] {len(qualifying)} qualifying label(s); accounting b={args.accounting_b}")
    records = run_content_generation(
        qualifying,
        accounting_b=args.accounting_b,
        epsilon=args.epsilon,
        delta=args.delta,
        tau=args.tau,
        top_k=args.top_k,
        max_total_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
    )
    out = contents_path(work_dir)
    write_jsonl(out, records)
    for record in records:
        print(
            f"[Step 3] label={record['label']!r} chunk={record['chunk_index']} "
            f"engine={record['engine']} n={record['actual_batch_size']} "
            f"content={record['content']!r}"
        )
    print(f"[Step 3] Wrote {len(records)} record(s) to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
