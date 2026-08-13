"""InvisibleInk Algorithm 1 — DP text generation over token logits.

Personal implementation of the generation loop from
`Vinod et al. 2025 <https://arxiv.org/abs/2507.02974>`_ (NeurIPS 2025),
wired onto this project's Gemma 2 IT token plumbing and the same
``wrap_fn`` / ``**wrap_kwargs`` calling convention as
:mod:`agent_memories.agent.privacy.privatisation`.

Privacy-critical arithmetic is delegated to the authors' pip package:

* :func:`~agent_memories.agent.invisible_ink.mechanism.dclip_mean` wraps
  :func:`invink.utils.difference_clip`
* :func:`~agent_memories.agent.invisible_ink.mechanism.top_k_plus_mask`
  wraps :func:`invink.utils.get_topk` (with paper ``B``, not
  ``invink.generate``'s ``B + 1``)
* clip / epsilon / rho calibration wraps
  :func:`invink.utils.get_clip` / :func:`invink.utils.get_epsilon` /
  :func:`invink.utils.compute_rho` via
  :mod:`agent_memories.agent.invisible_ink.accounting`

What is ours: the per-token loop on
:func:`~agent_memories.agent.privacy.token_generation.prefill_padded` +
:func:`~agent_memories.agent.privacy.token_generation.continue_batched`,
the torch multinomial sample, and the prompt templates in
:mod:`agent_memories.agent.privacy.prompts`. We deliberately do **not**
call :func:`invink.generate`: it loads its own model, owns fixed
positional prompt templates, auto-derives ``max_toks`` from reference
length, and emits ``num`` sequences from disjoint partitions — none of
which fits "one JSON array of ``k`` labels from one batch of ``B``
references".

Per-token steps (paper Algorithm 1):

1. Prefill ``B`` sensitive prompts plus one public prompt.
2. Build Top-k+ vocabulary from public logits only:
   ``{y : φ_pub(y) ≥ ℓ − 2C/B}``.
3. DClip-aggregate private logits:
   ``φ_bar = φ_pub + (1/B) Σ clip_C(φ_i − φ_pub)``.
4. Sample from ``softmax(φ_bar[V_k+] / τ)``; every token spends budget
   (no Amin SVT public path).

Clip ``C`` is calibrated from target ``(ε, δ)`` via Theorem 2:
``C = B τ √(2 ρ / T)`` with ``ρ`` inverted from ``ε`` at fixed ``δ``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..privacy import token_generation as tg
from ..privacy.prompts import wrap
from .accounting import (
    InvisibleInkAccount,
    clip_for_budget,
    epsilon_for_tokens,
    rho_for_tokens,
)
from .mechanism import dclip_mean, sample_topk_plus, top_k_plus_mask


def generate(
    texts: list[str],
    *,
    b: int,
    tau: float,
    top_k: int,
    max_total_tokens: int,
    target_epsilon: float,
    delta: float,
    wrap_fn: Callable[..., str] = wrap,
    **wrap_kwargs: Any,
) -> tuple[str, InvisibleInkAccount]:
    """Run InvisibleInk Algorithm 1 on one batch; return text + account.

    ``texts`` are pre-rendered memory-items blocks, one per batch
    member; each drops into the ``{items}`` slot of whichever template
    ``wrap_fn`` formats. ``wrap_fn`` defaults to the round-2
    :func:`~agent_memories.agent.privacy.prompts.wrap`; the round-1
    caller passes ``wrap_fn=wrap_label`` together with ``k=...``. The
    public prompt is built by calling ``wrap_fn(**wrap_kwargs)`` with
    no ``items`` argument so the items block defaults to
    ``"(no examples)"``, aligning format tokens between the public and
    private branches. Under InvisibleInk ``φ_pub`` is subtracted from
    every private logit and defines the sampling support, so that
    alignment matters more than it did under Amin's SVT branch.

    ``b`` is the paper's private-reference count (typically
    ``len(texts)``). ``top_k`` is the truncation parameter; pass a
    value ``>=`` vocabulary size for full-vocabulary sampling.
    """
    if b < 1:
        raise ValueError(f"b (private references) must be >= 1, got {b}.")
    if len(texts) != b:
        raise ValueError(
            f"len(texts)={len(texts)} must equal b={b} (paper private-reference count)."
        )
    if max_total_tokens < 1:
        raise ValueError(f"max_total_tokens must be >= 1, got {max_total_tokens}.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    c = clip_for_budget(target_epsilon, delta, max_total_tokens, b, tau)

    stop = tg.stop_ids()
    prompts = [wrap_fn(items=text, **wrap_kwargs) for text in texts]
    public_prompt = wrap_fn(**wrap_kwargs)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)

    logits, state = tg.prefill_padded(prompt_ids + [public_ids])
    x_ids: list[int] = []
    topk_sizes: list[int] = []
    expansion_hits = 0

    while len(x_ids) < max_total_tokens:
        Z = logits[: len(prompt_ids)]  # noqa: N806
        z_pub = logits[len(prompt_ids)]

        mask, expansion_idxs = top_k_plus_mask(z_pub, k=top_k, c=c, b=b)
        topk_sizes.append(int(mask.sum().item()))
        phi_bar = dclip_mean(Z, z_pub, c)
        tok = sample_topk_plus(phi_bar, mask, tau)
        x_ids.append(tok)
        if expansion_idxs.size > 0 and tok in set(expansion_idxs.tolist()):
            expansion_hits += 1
        if tok in stop:
            break

        logits, state = tg.continue_batched(state, [tok] * (len(prompt_ids) + 1))

    n_toks = len(x_ids)
    topk_arr = np.asarray(topk_sizes, dtype=np.float64)
    topk_mean = float(topk_arr.mean()) if topk_sizes else 0.0
    topk_std = float(topk_arr.std()) if topk_sizes else 0.0
    rho = rho_for_tokens(n_toks, c, b, tau)
    eps = epsilon_for_tokens(n_toks, c, b, tau, delta)

    account = InvisibleInkAccount(
        epsilon=eps,
        delta=delta,
        rho=rho,
        t=max_total_tokens,
        b=b,
        c=c,
        tau=tau,
        top_k=top_k,
        tokens_used=n_toks,
        topk_plus_mean=topk_mean,
        topk_plus_std=topk_std,
        expansion_set_count=expansion_hits,
    )
    return tg.decode(x_ids), account
