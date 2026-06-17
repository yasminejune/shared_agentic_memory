"""Illustrate how Amin et al. clipping reshapes a logit vector.

Uses one example from ``scripts/amin_et_al/examples.csv``, builds private and
public prompts via the existing round-1 wrapper, captures the private branch's
next-token logits before sampling, and compares the baseline distribution to
clipped versions at c=50, c=20, and c=10.

Outputs:
- ``logit_rank_comparison.png``
- ``logit_distribution_comparison.png``
- ``softmax_mass_comparison.png``
- ``top10_tokens_before_after_clipping.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agent_memories.agent.privacy.privatisation import clip_recenter
from agent_memories.agent.privacy.prompts import wrap_label
from agent_memories.agent.privacy.token_generation import decode, encode_chat, prefill_padded

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
DEFAULT_OUTPUT_DIR = Path("scripts/amin_et_al/outputs")
CLIP_VALUES = (50.0, 20.0, 10.0)
TOP_K = 10


def _token_label(token_id: int) -> str:
    """Readable label for one token id."""
    text = decode([int(token_id)])
    text = text.replace("\n", "\\n")
    if text.strip():
        return text.strip()
    return f"<id:{int(token_id)}>"


def _extract_private_logits(example_text: str, *, k: int) -> torch.Tensor:
    """Get one pre-sampling private-branch next-token logit vector."""
    private_prompt = wrap_label(items=example_text, k=k)
    public_prompt = wrap_label(k=k)
    private_ids = encode_chat(private_prompt)
    public_ids = encode_chat(public_prompt)
    logits, _ = prefill_padded([private_ids, public_ids])
    return logits[0].detach().cpu()


def _build_vectors(base_logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Baseline and clipped vectors keyed by display label."""
    vectors: dict[str, np.ndarray] = {"baseline": base_logits.numpy()}
    z_batch = base_logits.unsqueeze(0)
    for c in CLIP_VALUES:
        clipped = clip_recenter(z_batch, c).squeeze(0).detach().cpu().numpy()
        vectors[f"c={int(c)}"] = clipped
    return vectors


def _plot_ranked_curves(
    vectors: dict[str, np.ndarray],
    out_path: Path,
    top_token_ids: np.ndarray,
) -> None:
    """Figure 1: ranked logits before/after clipping."""
    plt.figure(figsize=(12, 7))
    for name, vec in vectors.items():
        ranked = np.sort(vec)[::-1]
        plt.plot(ranked, linewidth=1.5, label=name)

    baseline = vectors["baseline"]
    baseline_order = np.argsort(-baseline)
    for rank, token_id in enumerate(top_token_ids, start=1):
        y = baseline[int(token_id)]
        label = _token_label(int(token_id))
        plt.scatter(rank - 1, y, s=22, color="black", zorder=5)
        plt.annotate(
            f"{rank}: {label}",
            xy=(rank - 1, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    plt.title("Ranked logit vectors: baseline vs Amin clipping")
    plt.xlabel("Token rank (descending baseline/value order per curve)")
    plt.ylabel("Logit value")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_distributions(vectors: dict[str, np.ndarray], out_path: Path) -> None:
    """Figure 2: value-distribution shift before/after clipping."""
    plt.figure(figsize=(12, 7))
    bins = 120
    for name, vec in vectors.items():
        plt.hist(
            vec,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=name,
        )
    plt.title("Logit value distribution: baseline vs Amin clipping")
    plt.xlabel("Logit value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _softmax(vec: np.ndarray) -> np.ndarray:
    shifted = vec - np.max(vec)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _write_top10_csv(
    vectors: dict[str, np.ndarray],
    top_token_ids: np.ndarray,
    out_path: Path,
) -> None:
    """Save baseline top-10 tokens and their before/after logits."""
    rows: list[dict[str, object]] = []
    baseline = vectors["baseline"]
    for rank, token_id in enumerate(top_token_ids, start=1):
        row: dict[str, object] = {
            "rank_baseline": rank,
            "token_id": int(token_id),
            "token_text": _token_label(int(token_id)),
            "baseline_logit": float(baseline[int(token_id)]),
        }
        for key, vec in vectors.items():
            if key == "baseline":
                continue
            row[f"{key}_logit"] = float(vec[int(token_id)])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Row index in scripts/amin_et_al/examples.csv (default: 0).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k passed to wrap_label for prompt consistency (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    df = pd.read_csv(EXAMPLES_PATH)
    if not 0 <= args.example_index < len(df):
        raise ValueError(
            f"--example-index must be between 0 and {len(df) - 1}, got {args.example_index}."
        )
    example_text = str(df.loc[args.example_index, "text"])

    base_logits = _extract_private_logits(example_text, k=args.k)
    vectors = _build_vectors(base_logits)
    baseline_order = np.argsort(-vectors["baseline"])
    top_token_ids = baseline_order[:TOP_K]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_plot_path = args.output_dir / "logit_rank_comparison.png"
    dist_plot_path = args.output_dir / "logit_distribution_comparison.png"
    top10_csv_path = args.output_dir / "top10_tokens_before_after_clipping.csv"

    _plot_ranked_curves(vectors, rank_plot_path, top_token_ids)
    _plot_distributions(vectors, dist_plot_path)
    _write_top10_csv(vectors, top_token_ids, top10_csv_path)

    print("Created clipping-illustration outputs:")
    print(f"  example_index = {args.example_index}")
    print(f"  k = {args.k}")
    print(f"  vocab_size = {vectors['baseline'].shape[0]}")
    print(f"  rank_plot = {rank_plot_path}")
    print(f"  distribution_plot = {dist_plot_path}")
    print(f"  top10_csv = {top10_csv_path}")


if __name__ == "__main__":
    main()
