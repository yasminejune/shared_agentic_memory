"""Step 1: differentially private labels from task descriptions.

InvisibleInk generates labels from task queries, not from memory items.
Writes labels.json under --work-dir.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from agent_memories.agent.invisible_ink import generate_with_oom_fallback
from agent_memories.agent.invisible_ink.accounting import InvisibleInkAccount
from agent_memories.agent.privacy.prompts import wrap_label
from agent_memories.generalisation.buffer import entry_id
from agent_memories.generalisation.io import (
    DELTA,
    EPSILON,
    GEMMA_CHUNK_SIZE,
    K_LABELS,
    STEP1_MAX_TOKENS,
    TAU,
    TOP_K,
    account_to_dict,
    add_io_arguments,
    labels_path,
    load_memory_entries_from_csv,
    resolve_io_paths,
    write_json,
)
from agent_memories.generalisation.labelling import parse_json_labels
from agent_memories.memory import MemoryEntry

GenerateFn = Callable[..., tuple[str, InvisibleInkAccount, str]]


def run_label_generation(
    entries: Sequence[MemoryEntry],
    *,
    k: int = K_LABELS,
    epsilon: float = EPSILON,
    delta: float = DELTA,
    tau: float = TAU,
    top_k: int = TOP_K,
    max_total_tokens: int = STEP1_MAX_TOKENS,
    chunk_size: int = GEMMA_CHUNK_SIZE,
    generate_fn: GenerateFn = generate_with_oom_fallback,
) -> dict[str, Any]:
    """Synthesise k labels from task queries; return the labels artefact."""
    if not entries:
        raise ValueError("Step 1 requires at least one extracted task.")
    texts = [entry.query for entry in entries]
    raw, account, engine = generate_fn(
        texts,
        b=len(texts),
        tau=tau,
        top_k=top_k,
        max_total_tokens=max_total_tokens,
        target_epsilon=epsilon,
        delta=delta,
        chunk_size=chunk_size,
        wrap_fn=wrap_label,
        k=k,
    )
    parsed = parse_json_labels(raw, k)
    return {
        "k": k,
        "labels": [text for text, _ in parsed],
        "label_parsed_flags": [ok for _, ok in parsed],
        "raw_output": raw,
        "engine": engine,
        "account": account_to_dict(account),
        "b": len(texts),
        "entry_ids": [entry_id(entry) for entry in entries],
        "queries": texts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_io_arguments(parser)
    parser.add_argument("--k-labels", type=int, default=K_LABELS)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-tokens", type=int, default=STEP1_MAX_TOKENS)
    parser.add_argument("--chunk-size", type=int, default=GEMMA_CHUNK_SIZE)
    args = parser.parse_args(argv)

    csv_path, work_dir = resolve_io_paths(args)
    out = labels_path(work_dir)
    print("[Step 1] InvisibleInk label generation", flush=True)
    print(f"[Step 1] memories CSV: {csv_path}", flush=True)
    print(f"[Step 1] work dir:     {work_dir}", flush=True)
    print(f"[Step 1] output:       {out}", flush=True)

    pairs = load_memory_entries_from_csv(csv_path)
    entries = [entry for _, entry in pairs]
    print(f"[Step 1] Loaded {len(entries)} extracted tasks", flush=True)
    print(
        f"[Step 1] k={args.k_labels}, b={len(entries)}, "
        f"epsilon={args.epsilon}, delta={args.delta}, tau={args.tau}, "
        f"top_k={args.top_k}, T={args.max_tokens}, chunk_size={args.chunk_size}",
        flush=True,
    )
    print(
        "[Step 1] Starting generation (loads Gemma 2 2B IT; this can take a long time)...",
        flush=True,
    )
    artefact = run_label_generation(
        entries,
        k=args.k_labels,
        epsilon=args.epsilon,
        delta=args.delta,
        tau=args.tau,
        top_k=args.top_k,
        max_total_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
    )
    write_json(out, artefact)
    print(
        f"[Step 1] engine={artefact['engine']} b={artefact['b']} k={artefact['k']}",
        flush=True,
    )
    print(f"[Step 1] Wrote {len(artefact['labels'])} labels to {out}", flush=True)
    for idx, (label, ok) in enumerate(
        zip(artefact["labels"], artefact["label_parsed_flags"], strict=True),
        start=1,
    ):
        tag = "" if ok else "  [fallback]"
        print(f"    {idx}. {label}{tag}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
