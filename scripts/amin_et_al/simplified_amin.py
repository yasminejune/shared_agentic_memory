"""Privacy-free baseline of the WP2.3 Amin et al. loop.

Mirrors :mod:`scripts.amin_et_al.WP2_3` end-to-end on the same 10
toy examples in ``scripts/amin_et_al/examples.csv`` but strips the
privacy machinery of Amin et al. Algorithm 1:

* no tau / temperature scaling on the sampled distribution
  (``tau = 1`` so the division is a no-op),
* no public-prompt branch (every token is the "private" path),
* no theta threshold, no Laplace noise, no privacy budget ``r``.

Per-prompt logit clipping (Amin et al. Eq. 1) is retained because it
does non-privacy work too: without it, a single prompt with extreme
raw-logit magnitudes can dominate the batch average, since softmax
is shift-invariant but the raw logits per prompt are not on a common
scale. Clipping bounds each row to ``[-c, c]`` componentwise before
the mean is taken, so each prompt's influence on the aggregate is
comparable.

The output is a single synthetic sentence sampled from the clipped,
averaged next-token distribution of the 10 prompts. Used as a
baseline against the differentially-private output of ``WP2_3.py``.

Both this baseline and ``WP2_3.py`` load Gemma 2 2B IT and tokenise
each wrapped prompt once via
:func:`~agent_memories.agent.privacy.token_generation.encode_chat`
so the chat template is applied; the per-step model call uses
:func:`~agent_memories.agent.privacy.token_generation.get_next_token_logits_from_ids`
with ``prompt_ids[i] + x_ids`` so the unchanging prefix is never
re-tokenised and the running suffix is never decoded then
re-encoded. The IT-vs-base ablation that previously lived here as
a ``USE_INSTRUCT`` toggle has been moved to
:mod:`scripts.amin_et_al.compare_gemma`, which is the only entry
point that loads ``google/gemma-2-2b`` (the raw-completion base);
all other scripts and the production WP2.3 path stay on the IT
default with the chat template.
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
# Stand-in for what WP2 round 1 would emit. The simplified harness has
# no round 1, so the round-2 public-input label is hardcoded to a
# string that fits the 10 toy event-review rows in ``examples.csv``;
# the same constant is used by ``WP2_3.py`` so the privacy-free and
# DP runs are compared on identical prompts.

MODEL_NAME = "google/gemma-2-2b-it"
# Project standard per WP2-plan §5: the instruction-tuned 2B Gemma
# loaded with chat-template prompting via :func:`tg.encode_chat`.
# The IT-vs-base ablation lives in :mod:`compare_gemma` only.


def sample_next_token(Z: torch.Tensor, c: float) -> int:
    """Clip per-row logits via Amin Eq. 1, average across the batch, sample.

    ``Z`` has shape ``(batch, vocab)``. Each row is clipped into
    ``[-c, c]`` componentwise via :func:`clip_recenter` **before**
    averaging, so no single prompt with extreme raw-logit magnitudes
    dominates the mean. Temperature is fixed at 1 (no division by
    ``tau``) and no Laplace noise is added — this remains the
    privacy-free baseline; clipping is restored only for its
    non-privacy role of bounding per-prompt influence on the aggregate.
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
    """Run the privacy-free averaging loop and return the decoded text.

    ``prompt_ids`` are the per-batch-member pre-tokenised wrapped
    prompts (built in :func:`main` from :func:`wrap` plus
    :func:`tg.encode_chat`, so the Gemma 2 IT chat template is
    applied). Each iteration concatenates the running ``x_ids`` to
    each prompt's ids and feeds the result to
    :func:`tg.get_next_token_logits_from_ids` directly, so the
    unchanging prefix is never re-tokenised and the running suffix is
    never decoded then re-encoded.
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
