"""Side-by-side comparison of Gemma 2 IT vs Gemma 2 base on the WP2.3 prompt.

Runs the same privacy-free Amin Algorithm 1 averaging loop twice on
the 10 toy rows of ``scripts/amin_et_al/examples.csv``: once under
``google/gemma-2-2b-it`` with the chat-template prompt
(``tg.encode_chat``), once under ``google/gemma-2-2b`` with the
raw-completion prompt (``tg.encode``). For each regime the script
prints the raw decoded text alongside the input memory items so the
fittingness of the synthesised lesson against the 10 source reviews
can be assessed by hand. The judgement is intentionally subjective:
no parser verdict, no schema check.

This is the §3.5-aware harness: under the 2026-06-02 plan revision
(WP2-plan §3.3 + §11 deviation 9), the privacy-spending DP path
emits a single content paragraph anchored on ``Lesson:`` with no
markdown scaffolding, and the matching ``title`` and ``description``
are added downstream by the §3.5 Qwen post-processing call on the
DP-released content. That post-processing step is intentionally out
of scope here so the comparison isolates Gemma-regime effects on the
DP-spending step itself.

Fair-comparison invariants:

* same wrapped prompts (built via :func:`wrap` with the same
  :data:`LABEL` and the raw input texts as the items block),
* same clip bound :data:`C` and same token cap :data:`MAX_TOTAL_TOKENS`,
* same multinomial sampler.

The only confound that varies is (model x prompt regime), which is
exactly what the comparison is meant to isolate.

"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch

from agent_memories.agent.privacy import token_generation as tg
from agent_memories.agent.privacy.privatisation import clip_recenter
from agent_memories.agent.privacy.prompts import wrap

C = 20.0  # logit clip bound, matches simplified_amin.py / WP2_3.py
MAX_TOTAL_TOKENS = 80  # matches simplified_amin.py / WP2_3.py R_MAX
LABEL = "attending a recent event"  # same manual-in as simplified_amin.py

MODEL_INSTRUCT = "google/gemma-2-2b-it"
MODEL_BASE = "google/gemma-2-2b"


def generate(prompt_ids: list[list[int]]) -> str:
    """Privacy-free clip-mean-softmax-multinomial loop.

    Mirrors :func:`scripts.amin_et_al.simplified_amin.generate` but
    inlined so the comparison script does not depend on a sibling
    script's exact CLI shape. Each iteration concatenates ``x_ids``
    onto every prompt's pre-tokenised ids and feeds the result to
    :func:`tg.get_next_token_logits_from_ids`, paying no per-step
    re-tokenisation cost.
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
    """Swap to ``model_name``, encode prompts from ``texts``, run the loop."""
    print(f"\nLoading {model_name}...")
    tg.set_model(model_name)

    prompts = [wrap(items=text, label=LABEL) for text in texts]
    prompt_ids = [encode_fn(p) for p in prompts]
    return generate(prompt_ids)


def report(model_name: str, regime: str, output: str) -> None:
    """Print the regime header and the raw DP output for subjective review."""
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
