"""Privacy-free Amin clip+average on the 10 toy memories in examples.csv.

Same loop as WP2_3.py minus tau scaling, public prompt, theta, Laplace,
and privacy budget r. Clipping stays so one extreme prompt cannot dominate
the batch mean. Output is one sampled sentence.

Gemma 2 2B IT with the chat template. IT vs base is
scripts/gemma/compare_gemma.py, not this file.
"""

from __future__ import annotations

import pandas as pd
import torch

from agent_memories.agent.privacy import token_generation as tg
from agent_memories.agent.privacy.privatisation import clip_recenter
from agent_memories.agent.privacy.prompts import wrap

C = 10.0  # logit clip bound, matches WP2_3.py for apples-to-apples comparison
MAX_TOTAL_TOKENS = 80
LABEL = "attending a recent event"
# Same label as WP2_3.py so the privacy-free and DP runs share prompts.

MODEL_NAME = "google/gemma-2-2b-it"


def sample_next_token(Z: torch.Tensor, c: float) -> int:
    """Clip per-row logits, average the batch, sample one token.

    Z is (batch, vocab). Clip to [-c, c] before the mean so no single
    prompt dominates. Temperature is 1; no Laplace.
    """
    Z_clipped = clip_recenter(Z, c)
    z_bar = Z_clipped.mean(dim=0)
    probs = torch.softmax(z_bar, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(
    prompt_ids: list[list[int]],
    *,
    c: float = C,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> str:
    """Privacy-free averaging loop; return the decoded text.

    prompt_ids are the wrapped, chat-templated prefixes. Each step
    appends the running suffix ids; the prefix is not re-tokenised.
    """
    stop = tg.stop_ids()
    x_ids: list[int] = []
    i = 0
    while len(x_ids) < max_total_tokens:
        Z = tg.get_next_token_logits_from_ids([ids + x_ids for ids in prompt_ids])
        tok = sample_next_token(Z, c)
        x_ids.append(tok)
        print(f"Token {i} added")
        if tok in stop:
            break
        i += 1
    return tg.decode(x_ids)


def main() -> None:
    tg.set_model(MODEL_NAME)

    df = pd.read_csv("scripts/amin_et_al/examples.csv")
    items_blocks = df["text"].tolist()
    prompts = [wrap(items=block, label=LABEL) for block in items_blocks]
    prompt_ids = [tg.encode_chat(p) for p in prompts]

    synthetic = generate(prompt_ids)

    print()
    print(f"Synthetic text (clipped+averaged, no privacy, model={tg.MODEL_NAME}):")
    print(repr(synthetic))


if __name__ == "__main__":
    main()
