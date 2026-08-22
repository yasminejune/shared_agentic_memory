"""Gemma 2 instruct vs base on the same Amin averaging loop.

Runs the privacy-free clip-mean-softmax-multinomial loop twice on
the 10 toy rows in scripts/amin_et_al/examples.csv: once with
google/gemma-2-2b-it and the chat template (encode_chat), once with
google/gemma-2-2b and raw completion (encode). Prints the decoded
text next to the source reviews. No parser, no schema check; I read
both outputs.

The DP path in the thesis emits a single Lesson: paragraph with no
markdown. Title and description are added later by Qwen. That
post-processing is out of scope here on purpose, so the comparison
is only "does instruct vs base change the spending-step text?"

Held fixed: wrap() prompts, clip bound C, token cap MAX_TOTAL_TOKENS,
multinomial sampler. The only thing that moves is (model x prompt
regime).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch

from agent_memories.agent.amin_et_al.privatisation import clip_recenter
from agent_memories.agent.lm import token_generation as tg
from agent_memories.agent.lm.prompts import wrap

C = 20.0  # logit clip bound, matches simplified_amin.py / WP2_3.py
MAX_TOTAL_TOKENS = 80  # matches simplified_amin.py / WP2_3.py R_MAX
LABEL = "attending a recent event"  # same manual-in as simplified_amin.py

MODEL_INSTRUCT = "google/gemma-2-2b-it"
MODEL_BASE = "google/gemma-2-2b"


def generate(prompt_ids: list[list[int]]) -> str:
    """Privacy-free clip-mean-softmax-multinomial loop.

    Same loop as scripts/amin_et_al/simplified_amin.py, inlined so
    this comparison does not depend on that script's CLI. Each step
    concatenates x_ids onto every prompt's pre-tokenised ids and
    calls tg.get_next_token_logits_from_ids (no re-tokenisation).
    """
    stop = tg.stop_ids()
    x_ids: list[int] = []
    i = 0
    while len(x_ids) < MAX_TOTAL_TOKENS:
        Z = tg.get_next_token_logits_from_ids([ids + x_ids for ids in prompt_ids])
        z_bar = clip_recenter(Z, C).mean(dim=0)
        probs = torch.softmax(z_bar, dim=-1)
        tok = int(torch.multinomial(probs, num_samples=1).item())
        x_ids.append(tok)
        print(f"  token {i} added")
        if tok in stop:
            break
        i += 1
    return tg.decode(x_ids)


def run(
    model_name: str,
    encode_fn: Callable[[str], list[int]],
    texts: list[str],
) -> str:
    """Load ``model_name``, encode ``texts``, run the averaging loop."""
    print(f"\nLoading {model_name}...")
    tg.set_model(model_name)

    prompts = [wrap(items=text, label=LABEL) for text in texts]
    prompt_ids = [encode_fn(p) for p in prompts]
    return generate(prompt_ids)


def report(model_name: str, regime: str, output: str) -> None:
    """Print the regime header and the raw output for a hand look."""
    print()
    print("=" * 72)
    print(f"Model:  {model_name}")
    print(f"Regime: {regime}")
    print("-" * 72)
    print("DP-generated lesson (raw):")
    print(repr(output))


def main() -> None:
    print(f"Max_total_tokens: {MAX_TOTAL_TOKENS}, label: {LABEL!r}")

    df = pd.read_csv("scripts/amin_et_al/examples.csv")
    texts: list[str] = df["text"].tolist()

    print()
    print("Input memory items:")
    for text in texts:
        print(f"  - {text}")

    it_output = run(MODEL_INSTRUCT, tg.encode_chat, texts)
    base_output = run(MODEL_BASE, tg.encode, texts)
    report(MODEL_INSTRUCT, "chat template", it_output)
    report(MODEL_BASE, "raw completion", base_output)


if __name__ == "__main__":
    main()
