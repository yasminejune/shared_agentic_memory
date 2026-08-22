"""Overlap of high-logit head tokens across the toy memories, before averaging.

For each of the 10 rows in examples.csv, take the private next-token logits
at TOKEN_POSITION, clip at each c, take the head (ranks above the biggest
early drop), and report pairwise/global overlap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agent_memories.agent.privacy.privatisation import clip_recenter
from agent_memories.agent.privacy.prompts import wrap_label
from agent_memories.agent.privacy.token_generation import (
    continue_batched,
    decode,
    encode_chat,
    prefill_padded,
)

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
OUTPUT_DIR = Path("scripts/amin_et_al/outputs")
K = 5
CLIP_VALUES = (50.0, 20.0, 10.0)
ELBOW_SCAN_RANKS = 200
# Generation step to analyse: 2 = second token, 3 = third token, etc.
# Steps 1..N-1 use a shared greedy prefix from the first example so format
# tokens like `["` do not dominate the overlap comparison.
TOKEN_POSITION = 2


def _token_label(token_id: int) -> str:
    text = decode([int(token_id)])
    text = text.replace("\n", "\\n")
    if text.strip():
        return text.strip()
    return f"<id:{int(token_id)}>"


def _shared_prefix_ids(reference_text: str) -> list[int]:
    """Greedy prefix from the reference memory for tokens 1..TOKEN_POSITION-1."""
    if TOKEN_POSITION <= 1:
        return []
    ids = encode_chat(wrap_label(items=reference_text, k=K))
    logits, state = prefill_padded([ids])
    prefix: list[int] = []
    for _ in range(TOKEN_POSITION - 1):
        tok = int(logits[0].argmax().item())
        prefix.append(tok)
        logits, state = continue_batched(state, [tok])
    return prefix


def _private_logits_at_step(example_text: str, prefix_ids: list[int]) -> np.ndarray:
    ids = encode_chat(wrap_label(items=example_text, k=K))
    logits, state = prefill_padded([ids])
    for tok in prefix_ids:
        logits, state = continue_batched(state, [tok])
    return logits[0].detach().cpu().numpy()


def _clip_logits(logits: np.ndarray, c: float) -> np.ndarray:
    z = torch.tensor(logits).unsqueeze(0)
    return clip_recenter(z, c).squeeze(0).detach().cpu().numpy()


def _elbow_head_token_ids(clipped: np.ndarray) -> tuple[set[int], int]:
    """Return head token ids and elbow rank via largest drop in early ranks."""
    order = np.argsort(-clipped)
    ranked = clipped[order]
    scan = min(ELBOW_SCAN_RANKS, len(ranked) - 1)
    drops = ranked[:scan] - ranked[1 : scan + 1]
    elbow_rank = int(np.argmax(drops))
    head_ids = {int(tid) for tid in order[: elbow_rank + 1]}
    return head_ids, elbow_rank


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _plot_jaccard(matrix: np.ndarray, labels: list[str], out_path: Path, *, c: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"Pairwise Jaccard overlap of head sets (c={int(c)}, step={TOKEN_POSITION})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    df = pd.read_csv(EXAMPLES_PATH)
    memory_labels = [f"m{row['id']}" for _, row in df.iterrows()]
    reference_text = str(df.loc[0, "text"])
    prefix_ids = _shared_prefix_ids(reference_text)
    prefix_text = decode(prefix_ids) if prefix_ids else "(none)"
    print(f"TOKEN_POSITION = {TOKEN_POSITION}")
    print(f"Shared prefix token ids: {prefix_ids}")
    print(f"Shared prefix text: {prefix_text!r}")
    print()

    base_logits = [
        _private_logits_at_step(str(row["text"]), prefix_ids) for _, row in df.iterrows()
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    step_tag = f"_step{TOKEN_POSITION}"

    for c in CLIP_VALUES:
        heads: list[set[int]] = []
        elbow_ranks: list[int] = []

        for idx, logits in enumerate(base_logits):
            clipped = _clip_logits(logits, c)
            head, elbow_rank = _elbow_head_token_ids(clipped)
            heads.append(head)
            elbow_ranks.append(elbow_rank)

            order = np.argsort(-clipped)
            top_id = int(order[0])
            summary_rows.append(
                {
                    "generation_step": TOKEN_POSITION,
                    "shared_prefix": prefix_text,
                    "c": int(c),
                    "memory": memory_labels[idx],
                    "head_size": len(head),
                    "elbow_rank": elbow_rank,
                    "top_token_id": top_id,
                    "top_token_text": _token_label(top_id),
                }
            )

        n = len(heads)
        jaccard = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                jaccard[i, j] = _jaccard(heads[i], heads[j])

        intersection = set.intersection(*heads) if heads else set()
        union = set.union(*heads) if heads else set()

        coverage_rows: list[dict[str, object]] = []
        for token_id in sorted(union):
            count = sum(1 for h in heads if token_id in h)
            coverage_rows.append(
                {
                    "generation_step": TOKEN_POSITION,
                    "c": int(c),
                    "token_id": token_id,
                    "token_text": _token_label(token_id),
                    "memory_count": count,
                    "in_all_memories": count == n,
                }
            )

        heatmap_path = OUTPUT_DIR / f"head_overlap_jaccard_c{int(c)}{step_tag}.png"
        coverage_path = OUTPUT_DIR / f"head_token_coverage_c{int(c)}{step_tag}.csv"
        _plot_jaccard(jaccard, memory_labels, heatmap_path, c=c)
        pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)

        print(f"=== c={int(c)}, step={TOKEN_POSITION} ===")
        print(f"  head sizes: {[len(h) for h in heads]}")
        print(f"  elbow ranks: {elbow_ranks}")
        print(f"  global intersection: {len(intersection)} tokens")
        print(f"  global union: {len(union)} tokens")
        if intersection:
            print(
                "  intersection tokens:",
                ", ".join(_token_label(tid) for tid in sorted(intersection)),
            )
        print(f"  mean pairwise Jaccard: {jaccard[np.triu_indices(n, k=1)].mean():.3f}")
        print(f"  heatmap: {heatmap_path}")
        print(f"  coverage csv: {coverage_path}")
        print()

    summary_path = OUTPUT_DIR / f"head_overlap_summary{step_tag}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
