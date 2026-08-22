"""End-to-end check of clip, average, and decode on the toy memories.

Repeats the first examples.csv row ten times so every batch row is
identical. Per-row samples should match the averaged sample under
argmax; this does not test averaging when rows differ.

--sample argmax is the deterministic plumbing check. --sample multinomial
matches simplified_amin.py and can disagree because each draw is independent.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from agent_memories.agent.privacy import token_generation as tg
from agent_memories.agent.privacy.privatisation import clip_recenter
from agent_memories.agent.privacy.prompts import wrap

C = 10.0  # logit clip bound, matches simplified_amin.py / WP2_3.py
MAX_TOTAL_TOKENS = 80  # matches simplified_amin.py / WP2_3.py R_MAX
LABEL = "attending a recent event"  # same stand-in as simplified_amin.py
TEXT = "I hated the event, terrible."  # row 1 of examples.csv
N = 10  # batch size, matches simplified_amin.py / WP2_3.py


def _argmax_sample(probs: torch.Tensor) -> int:
    """Greedy decoder: pick the most likely token."""
    return int(torch.argmax(probs, dim=-1).item())


def _multinomial_sample(probs: torch.Tensor) -> int:
    """Stochastic decoder: draw one token from ``probs``."""
    return int(torch.multinomial(probs, num_samples=1).item())


SAMPLERS: dict[str, Callable[[torch.Tensor], int]] = {
    "argmax": _argmax_sample,
    "multinomial": _multinomial_sample,
}


def step(
    Z: torch.Tensor,
    c: float,
    sample: Callable[[torch.Tensor], int],
) -> tuple[list[int], int, float]:
    """One clip+average step.

    Returns per-prompt samples, the averaged sample, and max |Z[i]-Z[0]|
    (MPS determinism check). Same sampler on both sides.
    """
    max_row_diff = float((Z - Z[0]).abs().max().item())
    per_prompt_tokens = [sample(torch.softmax(Z[i], dim=-1)) for i in range(Z.shape[0])]
    z_bar = clip_recenter(Z, c).mean(dim=0)
    averaged_token = sample(torch.softmax(z_bar, dim=-1))
    return per_prompt_tokens, averaged_token, max_row_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="argmax", choices=sorted(SAMPLERS))
    args = parser.parse_args()
    sample = SAMPLERS[args.sample]

    items_block = TEXT
    prompts = [wrap(items=items_block, label=LABEL)] * N
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    stop = tg.stop_ids()

    x_ids: list[int] = []
    mismatched_steps: list[int] = []

    print(f"Generation loop, max_total_tokens={MAX_TOTAL_TOKENS}, batch={N}")
    print(f"{'step':>4}  {'avg_tok':>7}  {'agree':>5}  " f"{'max|Z[i]-Z[0]|':>14}  decoded")

    while len(x_ids) < MAX_TOTAL_TOKENS:
        Z = tg.get_next_token_logits_from_ids([ids + x_ids for ids in prompt_ids])

        per_prompt_tokens, tok_bar, max_row_diff = step(Z, C, sample)
        n_agree = sum(1 for t in per_prompt_tokens if t == tok_bar)
        if n_agree != N:
            mismatched_steps.append(len(x_ids))

        x_ids.append(tok_bar)
        step_idx = len(x_ids) - 1
        decoded = tg.decode([tok_bar])
        print(
            f"{step_idx:>4}  {tok_bar:>7}  {n_agree:>2}/{N}  " f"{max_row_diff:>14.3e}  {decoded!r}"
        )
        if n_agree != N:
            disagreeing = [(i, t) for i, t in enumerate(per_prompt_tokens) if t != tok_bar]
            for i, t in disagreeing:
                print(f"        prompt {i}: token {t} -> {tg.decode([t])!r}")

        if tok_bar in stop:
            break

    print()
    print(f"Synthetic text (clipped+averaged, {args.sample}):")
    print(repr(tg.decode(x_ids)))
    print()
    if mismatched_steps:
        print(
            f"Per-prompt {args.sample} disagreed with averaged {args.sample} "
            f"at {len(mismatched_steps)} step(s): {mismatched_steps}"
        )
    else:
        print(
            f"Per-prompt {args.sample} matched averaged {args.sample} at "
            f"every step (0 / {len(x_ids)} disagreements)."
        )


if __name__ == "__main__":
    main()
