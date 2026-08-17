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
:func:`~agent_memories.agent.privacy.token_generation.continue_batched`
(and the algorithmically identical microbatched fallback
:func:`generate_microbatched`), the torch multinomial sample, and the
prompt templates in :mod:`agent_memories.agent.privacy.prompts`. We
deliberately do **not** call :func:`invink.generate`: it loads its own
model, owns fixed positional prompt templates, auto-derives
``max_toks`` from reference length, and emits ``num`` sequences from
disjoint partitions — none of which fits "one JSON array of ``k``
labels from one batch of ``B`` references".

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
import torch

from ..privacy import token_generation as tg
from ..privacy.prompts import wrap
from .accounting import (
    InvisibleInkAccount,
    clip_for_budget,
    epsilon_for_tokens,
    rho_for_tokens,
)
from .mechanism import dclip_mean, sample_topk_plus, top_k_plus_mask


def _validate_generate_args(
    texts: list[str],
    *,
    b: int,
    top_k: int,
    max_total_tokens: int,
) -> None:
    if b < 1:
        raise ValueError(f"b (private references) must be >= 1, got {b}.")
    if len(texts) < b:
        raise ValueError(
            f"len(texts)={len(texts)} must be >= b={b} "
            "(accounting private-reference count; extra rows are allowed)."
        )
    if max_total_tokens < 1:
        raise ValueError(f"max_total_tokens must be >= 1, got {max_total_tokens}.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")


def _prepare(
    texts: list[str],
    *,
    b: int,
    tau: float,
    top_k: int,
    max_total_tokens: int,
    target_epsilon: float,
    delta: float,
    wrap_fn: Callable[..., str],
    wrap_kwargs: dict[str, Any],
) -> tuple[float, set[int], list[list[int]], list[int]]:
    """Validate args, calibrate C, wrap and tokenise private + public prompts."""
    _validate_generate_args(texts, b=b, top_k=top_k, max_total_tokens=max_total_tokens)
    c = clip_for_budget(target_epsilon, delta, max_total_tokens, b, tau)
    stop = tg.stop_ids()
    prompts = [wrap_fn(items=text, **wrap_kwargs) for text in texts]
    public_prompt = wrap_fn(**wrap_kwargs)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)
    return c, stop, prompt_ids, public_ids


def _sample_one_token(
    z_private: torch.Tensor,
    z_pub: torch.Tensor,
    *,
    c: float,
    b: int,
    tau: float,
    top_k: int,
) -> tuple[int, int, bool]:
    """DClip + Top-k+ sample; return ``(token_id, |V_k+|, expansion_hit)``."""
    mask, expansion_idxs = top_k_plus_mask(z_pub, k=top_k, c=c, b=b)
    phi_bar = dclip_mean(z_private, z_pub, c)
    tok = sample_topk_plus(phi_bar, mask, tau)
    hit = bool(expansion_idxs.size > 0 and tok in set(expansion_idxs.tolist()))
    return tok, int(mask.sum().item()), hit


def _account_from_run(
    *,
    x_ids: list[int],
    topk_sizes: list[int],
    expansion_hits: int,
    max_total_tokens: int,
    b: int,
    c: float,
    tau: float,
    top_k: int,
    delta: float,
) -> InvisibleInkAccount:
    n_toks = len(x_ids)
    topk_arr = np.asarray(topk_sizes, dtype=np.float64)
    topk_mean = float(topk_arr.mean()) if topk_sizes else 0.0
    topk_std = float(topk_arr.std()) if topk_sizes else 0.0
    rho = rho_for_tokens(n_toks, c, b, tau)
    eps = epsilon_for_tokens(n_toks, c, b, tau, delta)
    return InvisibleInkAccount(
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


def _next_logits_microbatched(
    prompt_ids: list[list[int]],
    public_ids: list[int],
    x_ids: list[int],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Private-batch logits in chunks of ``chunk_size``, plus one public row."""
    rows: list[torch.Tensor] = []
    for start in range(0, len(prompt_ids), chunk_size):
        chunk = [ids + x_ids for ids in prompt_ids[start : start + chunk_size]]
        rows.append(tg.get_next_token_logits_from_ids(chunk))
    z_private = torch.cat(rows, dim=0)
    z_pub = tg.get_next_token_logits_from_ids([public_ids + x_ids])[0]
    return z_private, z_pub


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

    ``b`` is the accounting private-reference count used to calibrate
    clip ``C`` and the Theorem 2 spend. It must satisfy
    ``len(texts) >= b``. When they are equal, this is the paper's
    ``B = |R|``. When ``len(texts) > b``, DClip still averages over
    the actual rows (sensitivity ``1/n``) while clip / epsilon use
    the smaller ``b`` — an upper-bound account for a qualifying
    bucket that is larger than the gate threshold.

    Uses a single padded KV-cache prefill over ``B + 1`` rows. On
    long inputs that path can OOM on MPS; callers that need a
    memory-capped twin should use :func:`generate_microbatched`.
    """
    c, stop, prompt_ids, public_ids = _prepare(
        texts,
        b=b,
        tau=tau,
        top_k=top_k,
        max_total_tokens=max_total_tokens,
        target_epsilon=target_epsilon,
        delta=delta,
        wrap_fn=wrap_fn,
        wrap_kwargs=wrap_kwargs,
    )

    logits, state = tg.prefill_padded(prompt_ids + [public_ids])
    x_ids: list[int] = []
    topk_sizes: list[int] = []
    expansion_hits = 0

    while len(x_ids) < max_total_tokens:
        tok, topk_size, hit = _sample_one_token(
            logits[: len(prompt_ids)],
            logits[len(prompt_ids)],
            c=c,
            b=b,
            tau=tau,
            top_k=top_k,
        )
        x_ids.append(tok)
        topk_sizes.append(topk_size)
        if hit:
            expansion_hits += 1
        if tok in stop:
            break
        logits, state = tg.continue_batched(state, [tok] * (len(prompt_ids) + 1))

    account = _account_from_run(
        x_ids=x_ids,
        topk_sizes=topk_sizes,
        expansion_hits=expansion_hits,
        max_total_tokens=max_total_tokens,
        b=b,
        c=c,
        tau=tau,
        top_k=top_k,
        delta=delta,
    )
    return tg.decode(x_ids), account


def generate_microbatched(
    texts: list[str],
    *,
    b: int,
    tau: float,
    top_k: int,
    max_total_tokens: int,
    target_epsilon: float,
    delta: float,
    chunk_size: int,
    wrap_fn: Callable[..., str] = wrap,
    **wrap_kwargs: Any,
) -> tuple[str, InvisibleInkAccount]:
    """Algorithmically identical to :func:`generate`, without a full-batch prefill.

    Each token step gathers private logits in chunks of ``chunk_size``
    via :func:`~agent_memories.agent.privacy.token_generation.get_next_token_logits_from_ids`
    and runs the public branch as a single-row forward. Re-tokenises
    the running suffix each step (no KV-cache reuse) so peak memory
    stays ``O(chunk_size * seq_len)`` rather than ``O(B * seq_len)``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    c, stop, prompt_ids, public_ids = _prepare(
        texts,
        b=b,
        tau=tau,
        top_k=top_k,
        max_total_tokens=max_total_tokens,
        target_epsilon=target_epsilon,
        delta=delta,
        wrap_fn=wrap_fn,
        wrap_kwargs=wrap_kwargs,
    )

    x_ids: list[int] = []
    topk_sizes: list[int] = []
    expansion_hits = 0

    while len(x_ids) < max_total_tokens:
        z_private, z_pub = _next_logits_microbatched(prompt_ids, public_ids, x_ids, chunk_size)
        tok, topk_size, hit = _sample_one_token(z_private, z_pub, c=c, b=b, tau=tau, top_k=top_k)
        x_ids.append(tok)
        topk_sizes.append(topk_size)
        if hit:
            expansion_hits += 1
        if tok in stop:
            break

    account = _account_from_run(
        x_ids=x_ids,
        topk_sizes=topk_sizes,
        expansion_hits=expansion_hits,
        max_total_tokens=max_total_tokens,
        b=b,
        c=c,
        tau=tau,
        top_k=top_k,
        delta=delta,
    )
    return tg.decode(x_ids), account


def _is_oom_error(exc: BaseException) -> bool:
    """True when ``exc`` looks like GPU / MPS memory exhaustion."""
    msg = str(exc).lower()
    return any(
        needle in msg for needle in ("out of memory", "buffer size", "oom", "mps backend")
    )


def generate_with_oom_fallback(
    texts: list[str],
    *,
    chunk_size: int = 8,
    **kwargs: Any,
) -> tuple[str, InvisibleInkAccount, str]:
    """Run :func:`generate`, falling back to :func:`generate_microbatched` on OOM.

    Returns ``(decoded_text, account, engine)`` where ``engine`` is
    ``"production"`` or ``"microbatched"``. ``kwargs`` are forwarded to
    both call sites (``b``, ``tau``, ``top_k``, ``max_total_tokens``,
    ``target_epsilon``, ``delta``, ``wrap_fn``, wrap kwargs).
    """
    try:
        raw, account = generate(texts, **kwargs)
        return raw, account, "production"
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        raw, account = generate_microbatched(texts, chunk_size=chunk_size, **kwargs)
        return raw, account, "microbatched"
