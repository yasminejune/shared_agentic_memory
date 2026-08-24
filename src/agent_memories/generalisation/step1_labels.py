"""Step 1: differentially private labels from task descriptions.

InvisibleInk generates labels from task queries, not from memory items.
Writes labels.json under the run work dir.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agent_memories.agent.invisible_ink import generate_with_oom_fallback
from agent_memories.agent.invisible_ink.accounting import InvisibleInkAccount
from agent_memories.agent.lm.prompts import wrap_label
from agent_memories.generalisation.buffer import entry_id
from agent_memories.generalisation.io import (
    DELTA,
    EPSILON,
    GEMMA_CHUNK_SIZE,
    K_LABELS,
    MEMORIES_CSV,
    STEP1_MAX_TOKENS,
    TAU,
    TOP_K,
    WORK_DIR,
    account_to_dict,
    labels_path,
    load_memory_entries_from_csv,
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
    """Get k labels from ReasoningBank memories, which include the task text"""
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


def main() -> None:
    csv_path, work_dir = MEMORIES_CSV, WORK_DIR
    out = labels_path(work_dir)
    print("[Step 1] InvisibleInk label generation", flush=True)
    print(f"[Step 1] memories CSV: {csv_path}", flush=True)
    print(f"[Step 1] work dir:     {work_dir}", flush=True)
    print(f"[Step 1] output:       {out}", flush=True)

    pairs = load_memory_entries_from_csv(csv_path)
    entries = [entry for _, entry in pairs]
    print(f"[Step 1] Loaded {len(entries)} extracted tasks", flush=True)
    print(
        f"[Step 1] k={K_LABELS}, b={len(entries)}, "
        f"epsilon={EPSILON}, delta={DELTA}, tau={TAU}, "
        f"top_k={TOP_K}, T={STEP1_MAX_TOKENS}, chunk_size={GEMMA_CHUNK_SIZE}",
        flush=True,
    )
    print(
        "[Step 1] Starting generation (loads Gemma 2 2B IT; this can take a long time)...",
        flush=True,
    )
    artefact = run_label_generation(
        entries,
        k=K_LABELS,
        epsilon=EPSILON,
        delta=DELTA,
        tau=TAU,
        top_k=TOP_K,
        max_total_tokens=STEP1_MAX_TOKENS,
        chunk_size=GEMMA_CHUNK_SIZE,
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
    main()
