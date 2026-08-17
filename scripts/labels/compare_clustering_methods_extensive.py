"""Side-by-side cluster comparison: privacy-free Amin, DP Amin, k-means, BERTopic.

Sense-check harness that reads N memory texts from a CSV, produces k clusters
via each of four methods on the same inputs and the same k, assigns every
memory to its nearest cluster, and prints per-cluster membership plus the
cluster's label. The four methods isolate different axes of the WP2 round-1
labelling design:

* ``amin_nodp`` — privacy-free Amin Algorithm 1 with the production
  :func:`~agent_memories.agent.privacy.prompts.wrap_label` template and the
  production :func:`~agent_memories.generalisation.parse_json_labels` parser.
  The DP machinery is stripped (no Laplace noise, no SVT, no ``r`` budget),
  so this isolates "labels from an LLM averaging the inputs" from "DP noise
  on top of that". Mirrors :mod:`scripts.amin_et_al.simplified_amin`'s
  clip+mean+softmax+multinomial loop with ``wrap_label`` instead of
  :func:`~agent_memories.agent.privacy.prompts.wrap`.
* ``amin_dp`` — the production round-1 path, same call shape as
  :mod:`scripts.amin_et_al.WP2_8` (Amin Algorithm 1 with the SVT public-token
  trick, ``r`` and ``epsilon`` derived via :func:`solve_r`). Together with
  ``amin_nodp`` this lets the reviewer see how much of the cluster
  structure shifts under DP noise alone, with everything else held fixed.
* ``kmeans`` — spherical k-means: L2-normalise the
  ``sentence-transformers/all-MiniLM-L6-v2`` embeddings used elsewhere in the
  project so Euclidean k-means approximates cosine similarity (matching the
  geometry of the round-2 batch-assignment step in
  :mod:`~agent_memories.generalisation.assignment`). Post-hoc label per
  cluster is the input text nearest the cluster's centroid, truncated to
  four words for visual parity with the Amin labels.
* ``bertopic`` — BERTopic with the HDBSCAN clusterer swapped for
  ``sklearn.cluster.KMeans(n_clusters=k)`` and UMAP disabled via
  :class:`bertopic.dimensionality.BaseDimensionalityReduction`. With both
  stages neutered, the clustering itself becomes the same KMeans as method
  three; the only thing BERTopic adds on top is the c-TF-IDF label
  extraction (top-four discriminative terms per cluster). That isolates the
  labelling layer cleanly: any difference between ``kmeans`` and
  ``bertopic`` is attributable to c-TF-IDF labelling alone, since both run
  KMeans on identical pre-computed embeddings.
* ``bertopic_pure`` — BERTopic with its default UMAP and HDBSCAN backends,
  parameterised adaptively to the input size so the script still runs at
  the 10-row toy fixture (``UMAP(n_neighbors=min(15, N - 1))``,
  ``HDBSCAN(min_cluster_size=max(2, min(10, N // 3)))``) and collapses to
  the published BERTopic defaults at ``N >= 30``. Unlike the other four
  methods, the cluster count here is *not* pinned to ``--k`` — HDBSCAN
  decides how many topics exist and may also emit a ``-1`` "outlier"
  bucket for documents that fit no cluster, which is recorded as a
  synthetic last cluster labelled ``"outliers (HDBSCAN)"``. This is the
  "what does BERTopic do if you let it run with its own defaults?"
  reference point, in contrast to the apples-to-apples ``bertopic`` mode
  above. Expect uninformative single-cluster output on the toy fixture;
  this method only becomes meaningful at the WP3.4 pilot corpus size.
* ``invisible_ink`` — InvisibleInk (Vinod et al., arXiv:2507.02974)
  Algorithm 1 via
  :func:`~agent_memories.agent.invisible_ink.generate`, with the same
  production :func:`~agent_memories.agent.privacy.prompts.wrap_label`
  template and :func:`~agent_memories.generalisation.parse_json_labels`
  parser as the two Amin variants. Isolates the mechanism switch from
  Amin: same prompt, same Gemma 2 IT model, same ``k``, same cosine
  assignment; different clipping (DClip vs Amin clip-recenter),
  different support (Top-k+ vs full vocabulary), different accounting
  (every token paid a priori via Theorem 2, no SVT). Privacy-critical
  arithmetic is the authors' ``invink.utils``; the generation loop is
  ours on the project's KV-cache plumbing.

TopicDP (Wang et al. KDD 2022, a model-agnostic DP wrapper that injects
Gaussian noise calibrated via smooth-sensitivity sampling into a non-private
topic miner's output matrix) is intentionally out of scope here. This
harness compares cluster *structure* on a small fixture under matched k.
The matched-privacy comparison between Amin and TopicDP at varying epsilon
belongs to the WP4.5 privacy-utility sweep, on a larger corpus where
smooth-sensitivity sampling is meaningful and the comparison metric is
downstream retrieval / task success rather than cluster structure.

The fixture is :data:`EXAMPLES_PATH` (10 short event-review strings,
shared with the rest of ``scripts/amin_et_al/``) repeated 10 times so
``N = S = 100``. The only constant that *should* be retuned manually
when N grows is the Amin DP batch-size :data:`S`. The ``R_MAX`` cap on
private-token budget is independent of N. The privacy-free Amin loop's
:data:`MAX_TOTAL_TOKENS` is a hard cap on the total decoded length and
is also independent of N. See the inline comments on each constant for
the retune rationale.

``amin_dp`` calls production :func:`~agent_memories.agent.privacy.generate`
(the KV-cache ``prefill_padded`` + ``continue_batched`` path used by
:mod:`scripts.amin_et_al.WP2_8`). If that forward pass OOMs on long
inputs, it falls back to :func:`_generate_dp_microbatched`, which runs
forwards in chunks of ``--gemma-chunk-size`` (default
:data:`GEMMA_CHUNK_SIZE_DEFAULT`) without KV-cache reuse so peak memory
drops from ``O(N * seq_len)`` to ``O(gemma_chunk_size * seq_len)``.
``amin_nodp`` always uses the microbatched helper because it has no
production twin in ``privatisation``. ``invisible_ink`` calls production
:func:`~agent_memories.agent.invisible_ink.generate` the same way, and
on the same OOM falls back to
:func:`~agent_memories.agent.invisible_ink.generate_microbatched`
(chunked via the same ``--gemma-chunk-size``).

The script writes one JSONL record per method to
:data:`OUTPUTS_PATH` in overwrite mode (so the file always reflects the
most recent run end-to-end), and prints a per-method block to stdout with
the labels and the input texts grouped under their assigned cluster.

Stand-alone sense-check harness only: imports functions from ``src/`` but
does not modify them, and is not part of the production WP2 pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN
from invink.utils import cdp_eps
from sklearn.cluster import KMeans as SklearnKMeans
from umap import UMAP

from agent_memories.agent.invisible_ink import (
    InvisibleInkAccount,
)
from agent_memories.agent.invisible_ink import (
    generate as generate_invisible_ink,
)
from agent_memories.agent.invisible_ink import (
    generate_microbatched as generate_invisible_ink_microbatched,
)
from agent_memories.agent.privacy import (
    check_delta,
    epsilon_from_rho,
    generate,
    rho_for,
    solve_r,
)
from agent_memories.agent.privacy import token_generation as tg
from agent_memories.agent.privacy.privacy_accounting import PrivacyAccount
from agent_memories.agent.privacy.privatisation import (
    clip_recenter,
    sample_private,
    sample_public,
    softmax_l1_distance,
)
from agent_memories.agent.privacy.prompts import wrap_label
from agent_memories.generalisation import parse_json_labels
from agent_memories.memory import Embedder

# Amin DP / privacy-free hyperparameters. Values mirror
# scripts/amin_et_al/WP2_8.py so the DP path runs at the same privacy
# operating point as the existing round-1 harness. S matches the 10-row
# toy fixture repeated 10 times (N = 100); the Amin privacy proof bounds
# depend on the expected batch size. R_MAX is independent of N and rarely
# needs retuning.
S = 100
C = 20.0
TAU = 1.0
TAU_PUBLIC = 1.5
SIGMA = 0.1
THETA = 0.0
R_MAX = 80
EPSILON_DEFAULT = 200.0

# Hard cap on the privacy-free Amin loop's decoded length, independent of N.
# Matches scripts/amin_et_al/simplified_amin.py for like-for-like behaviour.
MAX_TOTAL_TOKENS = 80

K_DEFAULT = 4
SEED_DEFAULT = 0
# Max sensitive prompts per Gemma forward pass. The Amin average is identical
# whether logits are computed in one batch or accumulated over chunks; chunking
# only caps peak GPU memory (batch=100 can exceed 30 GiB).
GEMMA_CHUNK_SIZE_DEFAULT = 8
# InvisibleInk defaults. EPSILON_II_DEFAULT is the authors' recommended
# operating point (invink README); Amin needs EPSILON_DEFAULT=200 to stay
# coherent, so matched-budget comparisons should pass --ii-epsilon 200.
EPSILON_II_DEFAULT = 10.0
TOP_K_II_DEFAULT = 100
TAU_II_DEFAULT = 1.0
DELTA_II_DEFAULT = 1e-5
ALL_METHODS: tuple[str, ...] = (
    "amin_nodp",
    "amin_dp",
    "kmeans",
    "bertopic",
    "bertopic_pure",
    "invisible_ink",
)

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
OUTPUTS_PATH = Path("scripts/amin_et_al/outputs/cluster_comparison_extensive.jsonl")
TOY_REPEAT = 10

MODEL_NAME = "google/gemma-2-2b-it"


@dataclass(frozen=True)
class ClusteringResult:
    """One row in the side-by-side comparison.

    ``labels[i]`` is the label string for cluster ``i`` and
    ``assignments[j]`` is the cluster index (in ``[0, k)``) that the
    ``j``-th input text was assigned to. ``extra`` captures method-specific
    metadata: privacy account for the Amin DP variant, inertia for k-means,
    topic-info for BERTopic, raw decoded text for both Amin variants.
    """

    method: str
    labels: list[str]
    assignments: list[int]
    extra: dict[str, Any] = field(default_factory=dict)


def _assign_by_cosine(
    text_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
) -> list[int]:
    """Argmax cosine similarity per text against the label embeddings.

    L2-normalises both tensors and returns the per-text argmax over the
    ``(N_texts, N_labels)`` cosine-similarity matrix. Used by the two
    Amin variants to mirror the production round-2 batch-assignment
    step in :func:`agent_memories.generalisation.assign_memories_to_labels`
    without needing :class:`MemoryEntry` objects.
    """
    text_norms = np.linalg.norm(text_embeddings, axis=1)
    text_norms = np.where(text_norms == 0, 1e-12, text_norms)
    text_unit = text_embeddings / text_norms[:, None]

    label_norms = np.linalg.norm(label_embeddings, axis=1)
    label_norms = np.where(label_norms == 0, 1e-12, label_norms)
    label_unit = label_embeddings / label_norms[:, None]

    sims = text_unit @ label_unit.T
    return [int(np.argmax(row)) for row in sims]


def _truncate_words(text: str, n: int) -> str:
    """First ``n`` whitespace-delimited tokens of ``text``, re-joined."""
    return " ".join(text.split()[:n])


def _laplace(scale: float) -> float:
    return float(np.random.laplace(0.0, scale))


def _stack_next_token_logits_microbatched(
    prompt_ids: list[list[int]],
    x_ids: list[int],
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Stacked next-token logits for every sensitive prompt, bounded memory.

    Runs at most ``chunk_size`` Gemma forwards per call instead of one forward
    over the full batch. The returned tensor has shape
    ``(len(prompt_ids), vocab_size)`` and is bit-for-bit equivalent to a
    single :func:`tg.get_next_token_logits_from_ids` call over all rows —
    only peak activation memory changes.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")

    rows: list[torch.Tensor] = []
    for start in range(0, len(prompt_ids), chunk_size):
        chunk = prompt_ids[start : start + chunk_size]
        batch = [ids + x_ids for ids in chunk]
        rows.append(tg.get_next_token_logits_from_ids(batch))
    return torch.cat(rows, dim=0)


def _next_public_logits(public_ids: list[int], x_ids: list[int]) -> torch.Tensor:
    """Single-row next-token logits for the SVT public prompt."""
    return tg.get_next_token_logits_from_ids([public_ids + x_ids])[0]


def _generate_privacy_free_labels(
    items_blocks: Sequence[str],
    k: int,
    *,
    c: float = C,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    gemma_chunk_size: int = GEMMA_CHUNK_SIZE_DEFAULT,
) -> str:
    """Run the simplified_amin clip+mean+softmax+multinomial loop with wrap_label.

    Equivalent to :func:`scripts.amin_et_al.simplified_amin.generate` but
    swaps the round-2 :func:`~agent_memories.agent.privacy.prompts.wrap`
    template for the round-1
    :func:`~agent_memories.agent.privacy.prompts.wrap_label` template, so
    the decoded output is a JSON array of labels rather than a content
    paragraph. No Laplace noise, no SVT, no ``r`` budget; this is the
    structural baseline that isolates "labels from an LLM averaging
    inputs" from any DP-noise effect. Returns the raw decoded string for
    downstream :func:`parse_json_labels` parsing.
    """
    prompts = [wrap_label(items=block, k=k) for block in items_blocks]
    prompt_ids = [tg.encode_chat(p) for p in prompts]

    stop = tg.stop_ids()
    x_ids: list[int] = []
    step = 0
    while len(x_ids) < max_total_tokens:
        Z = _stack_next_token_logits_microbatched(prompt_ids, x_ids, chunk_size=gemma_chunk_size)
        z_bar = clip_recenter(Z, c).mean(dim=0)
        probs = torch.softmax(z_bar, dim=-1)
        tok = int(torch.multinomial(probs, num_samples=1).item())
        x_ids.append(tok)
        print(f"  amin_nodp token {step} sampled")
        if tok in stop:
            break
        step += 1
    return tg.decode(x_ids)


def _generate_dp_microbatched(
    texts: list[str],
    *,
    k: int,
    s: int,
    c: float,
    tau: float,
    tau_public: float,
    sigma: float,
    theta: float,
    r: int,
    delta: float,
    gemma_chunk_size: int,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> tuple[str, PrivacyAccount]:
    """Amin Algorithm 1 with micro-batched sensitive forwards.

    Algorithmically identical to
    :func:`agent_memories.agent.privacy.privatisation.generate`, but never
    stacks all sensitive prompts into one ``prefill_padded`` batch. Each
    token step gathers ``Z`` via :func:`_stack_next_token_logits_microbatched`
    and runs the public branch as a single-row forward. Re-tokenises the
    running suffix each step (no KV-cache reuse) so memory stays
    ``O(gemma_chunk_size * seq_len)`` rather than ``O(N * seq_len)``.
    """
    prompts = [wrap_label(items=text, k=k) for text in texts]
    public_prompt = wrap_label(k=k)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)

    stop = tg.stop_ids()
    x_ids: list[int] = []
    t = 0
    n_public = 0
    theta_hat = theta + _laplace(sigma)

    while t < r and len(x_ids) < max_total_tokens:
        Z = _stack_next_token_logits_microbatched(prompt_ids, x_ids, chunk_size=gemma_chunk_size)
        z_public = _next_public_logits(public_ids, x_ids)

        d_hat = softmax_l1_distance(Z, z_public, s) + _laplace(2.0 * sigma)
        if d_hat >= theta_hat:
            tok = sample_private(Z, c, tau, s)
            t += 1
            theta_hat = theta + _laplace(sigma)
        else:
            tok = sample_public(z_public, tau_public)
            n_public += 1

        x_ids.append(tok)
        if tok in stop:
            break

    rho = rho_for(r, s, c, tau, sigma)
    account = PrivacyAccount(
        epsilon=epsilon_from_rho(rho, delta),
        delta=delta,
        rho=rho,
        r=r,
        s=s,
        c=c,
        tau=tau,
        sigma=sigma,
        private_tokens_used=t,
        public_tokens_used=n_public,
    )
    return tg.decode(x_ids), account


def _is_oom_error(exc: RuntimeError) -> bool:
    """True when ``exc`` looks like a GPU / MPS memory exhaustion."""
    msg = str(exc).lower()
    return any(needle in msg for needle in ("out of memory", "buffer size", "oom", "mps backend"))


def _generate_dp(
    texts: list[str],
    *,
    k: int,
    r: int,
    delta: float,
    gemma_chunk_size: int,
) -> tuple[str, PrivacyAccount, str]:
    """DP Amin via production KV-cache path, microbatched fallback on OOM.

    Returns ``(decoded_output, privacy_account, engine)`` where ``engine``
    is ``"production"`` or ``"microbatched"`` for the JSONL ``extra`` field.
    """
    gen_kwargs = {
        "s": S,
        "c": C,
        "tau": TAU,
        "tau_public": TAU_PUBLIC,
        "sigma": SIGMA,
        "theta": THETA,
        "r": r,
        "delta": delta,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "wrap_fn": wrap_label,
        "k": k,
    }
    print("  amin_dp: trying production generate (KV-cache path)...")
    try:
        raw_output, account = generate(texts, **gen_kwargs)
        return raw_output, account, "production"
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        print(
            f"  amin_dp: KV-cache path OOM ({exc}); "
            f"falling back to microbatched (chunk_size={gemma_chunk_size})..."
        )
        raw_output, account = _generate_dp_microbatched(
            texts,
            k=k,
            s=S,
            c=C,
            tau=TAU,
            tau_public=TAU_PUBLIC,
            sigma=SIGMA,
            theta=THETA,
            r=r,
            delta=delta,
            gemma_chunk_size=gemma_chunk_size,
        )
        return raw_output, account, "microbatched"


def _generate_ii(
    texts: list[str],
    *,
    k: int,
    epsilon: float,
    top_k: int,
    b: int,
    tau: float,
    max_tokens: int,
    delta: float,
    gemma_chunk_size: int,
) -> tuple[str, InvisibleInkAccount, str]:
    """InvisibleInk via production KV-cache path, microbatched fallback on OOM.

    Returns ``(decoded_output, account, engine)`` where ``engine`` is
    ``"production"`` or ``"microbatched"`` for the JSONL ``extra`` field.
    """
    gen_kwargs: dict[str, Any] = {
        "b": b,
        "tau": tau,
        "top_k": top_k,
        "max_total_tokens": max_tokens,
        "target_epsilon": epsilon,
        "delta": delta,
        "wrap_fn": wrap_label,
        "k": k,
    }
    print("  invisible_ink: trying production generate (KV-cache path)...")
    try:
        raw_output, account = generate_invisible_ink(texts, **gen_kwargs)
        return raw_output, account, "production"
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        print(
            f"  invisible_ink: KV-cache path OOM ({exc}); "
            f"falling back to microbatched (chunk_size={gemma_chunk_size})..."
        )
        raw_output, account = generate_invisible_ink_microbatched(
            texts,
            chunk_size=gemma_chunk_size,
            **gen_kwargs,
        )
        return raw_output, account, "microbatched"


def run_amin_privacy_free(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
    gemma_chunk_size: int,
) -> ClusteringResult:
    """Variant 1: privacy-free Amin, parsed via the production JSON parser.

    Loads Gemma 2 IT, runs the no-noise label-generation loop, parses the
    result with :func:`parse_json_labels` (same parser the production
    round-1 path uses), embeds the parsed labels with ``embedder``, and
    cosine-assigns every input text to its nearest label.
    """
    tg.set_model(MODEL_NAME)
    raw_output = _generate_privacy_free_labels(texts, k=k, gemma_chunk_size=gemma_chunk_size)

    parsed = parse_json_labels(raw_output, k)
    labels = [text for text, _ in parsed]
    label_parsed = [flag for _, flag in parsed]

    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    label_embeddings = np.asarray(embedder.embed_batch(labels), dtype=np.float32)
    assignments = _assign_by_cosine(text_embeddings, label_embeddings)

    extra: dict[str, Any] = {
        "model": MODEL_NAME,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "clip_c": C,
        "gemma_chunk_size": gemma_chunk_size,
        "n_batch_members": len(texts),
        "raw_output": raw_output,
        "label_parsed": label_parsed,
    }
    return ClusteringResult(
        method="amin_nodp",
        labels=labels,
        assignments=assignments,
        extra=extra,
    )


def run_amin_dp(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
    epsilon: float | None,
    r_override: int | None,
    gemma_chunk_size: int,
) -> ClusteringResult:
    """Variant 2: production DP Amin round 1, parsed via the production JSON parser.

    Mirrors the call shape in :mod:`scripts.amin_et_al.WP2_8`. Exactly one
    of ``epsilon`` and ``r_override`` must be set; ``epsilon`` is the
    target privacy budget passed to :func:`solve_r`, ``r_override`` pins
    the private-token budget directly. The realised ``(epsilon, delta)``
    are reported via :func:`epsilon_from_rho` / :func:`check_delta` and
    captured in the ``extra`` field of the returned result so the JSONL
    record carries the full privacy account.
    """
    tg.set_model(MODEL_NAME)

    delta = 1.0 / len(texts)

    if r_override is not None:
        r = r_override
        rho = rho_for(r, S, C, TAU, SIGMA)
        realised_epsilon = epsilon_from_rho(rho, delta)
        print(
            f"  amin_dp: --r supplied directly, r = {r}, "
            f"s={S}, c={C}, tau={TAU}, sigma={SIGMA}, delta={delta} "
            f"-> realised epsilon = {realised_epsilon:.4f}"
        )
    else:
        target_epsilon = epsilon if epsilon is not None else EPSILON_DEFAULT
        r = solve_r(target_epsilon, delta, s=S, c=C, tau=TAU, sigma=SIGMA, r_max=R_MAX)
        rho = rho_for(r, S, C, TAU, SIGMA)
        realised_epsilon = epsilon_from_rho(rho, delta)
        print(
            f"  amin_dp: target epsilon = {target_epsilon}, delta = {delta}, "
            f"s={S}, c={C}, tau={TAU}, sigma={SIGMA} "
            f"-> r = {r}, realised epsilon = {realised_epsilon:.4f}"
        )

    chk = check_delta(rho, realised_epsilon, delta, n=len(texts))
    print(
        f"  amin_dp delta validity: chosen={chk.delta_chosen:.3e}, "
        f"min={chk.delta_min:.3e}, 1/n={chk.delta_max_convention:.3e}, "
        f"valid={chk.valid}"
    )
    if not chk.valid:
        print(
            "  amin_dp: WARNING chosen delta does not satisfy Amin Theorem 1 "
            "/ Appendix C; continuing anyway for the sense-check comparison."
        )

    if r == 0:
        print(
            "  amin_dp: budget too tight to sample even one private token; "
            "returning empty labels."
        )
        labels = [f"label_{i + 1}" for i in range(k)]
        assignments = [0] * len(texts)
        extra: dict[str, Any] = {
            "model": MODEL_NAME,
            "realised_epsilon": realised_epsilon,
            "realised_epsilon_tight": float(cdp_eps(rho, delta)),
            "r": r,
            "delta": delta,
            "rho": rho,
            "private_tokens_used": 0,
            "public_tokens_used": 0,
            "raw_output": "",
            "label_parsed": [False] * k,
            "delta_valid": chk.valid,
        }
        return ClusteringResult(
            method="amin_dp",
            labels=labels,
            assignments=assignments,
            extra=extra,
        )

    raw_output, account, dp_engine = _generate_dp(
        texts,
        k=k,
        r=r,
        delta=delta,
        gemma_chunk_size=gemma_chunk_size,
    )
    print(f"  amin_dp: completed via {dp_engine} engine.")

    parsed = parse_json_labels(raw_output, k)
    labels = [text for text, _ in parsed]
    label_parsed = [flag for _, flag in parsed]

    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    label_embeddings = np.asarray(embedder.embed_batch(labels), dtype=np.float32)
    assignments = _assign_by_cosine(text_embeddings, label_embeddings)

    extra = {
        "model": MODEL_NAME,
        "realised_epsilon": account.epsilon,
        "realised_epsilon_tight": float(cdp_eps(account.rho, account.delta)),
        "r": account.r,
        "delta": account.delta,
        "rho": account.rho,
        "private_tokens_used": account.private_tokens_used,
        "public_tokens_used": account.public_tokens_used,
        "dp_engine": dp_engine,
        "gemma_chunk_size": gemma_chunk_size,
        "n_batch_members": len(texts),
        "raw_output": raw_output,
        "label_parsed": label_parsed,
        "delta_valid": chk.valid,
    }
    return ClusteringResult(
        method="amin_dp",
        labels=labels,
        assignments=assignments,
        extra=extra,
    )


def run_invisible_ink(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
    epsilon: float,
    top_k: int,
    b: int,
    tau: float,
    max_tokens: int,
    delta: float,
    gemma_chunk_size: int,
) -> ClusteringResult:
    """Variant 6: InvisibleInk Algorithm 1, parsed via the production JSON parser.

    Same prompt / model / k / cosine-assignment surface as
    :func:`run_amin_dp`; the mechanism underneath is Vinod et al. 2025
    (DClip + Top-k+, every token paid via Theorem 2). ``b`` is the paper
    private-reference count and defaults to ``len(texts)`` so the whole
    fixture contributes, matching ``amin_dp``'s ``S``.
    """
    tg.set_model(MODEL_NAME)
    print(
        f"  invisible_ink: target epsilon = {epsilon}, delta = {delta}, "
        f"b={b}, tau={tau}, top_k={top_k}, max_tokens={max_tokens}"
    )
    raw_output, account, ii_engine = _generate_ii(
        texts,
        k=k,
        epsilon=epsilon,
        top_k=top_k,
        b=b,
        tau=tau,
        max_tokens=max_tokens,
        delta=delta,
        gemma_chunk_size=gemma_chunk_size,
    )
    print(f"  invisible_ink: completed via {ii_engine} engine.")
    print(
        f"  invisible_ink: tokens_used={account.tokens_used}/{account.t}, "
        f"C={account.c:.6f}, spent epsilon={account.epsilon:.4f}, "
        f"mean |V_k+|={account.topk_plus_mean:.1f}"
    )

    parsed = parse_json_labels(raw_output, k)
    labels = [text for text, _ in parsed]
    label_parsed = [flag for _, flag in parsed]

    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    label_embeddings = np.asarray(embedder.embed_batch(labels), dtype=np.float32)
    assignments = _assign_by_cosine(text_embeddings, label_embeddings)

    extra: dict[str, Any] = {
        "model": MODEL_NAME,
        "realised_epsilon": account.epsilon,
        "target_epsilon": epsilon,
        "delta": account.delta,
        "rho": account.rho,
        "t": account.t,
        "b": account.b,
        "c": account.c,
        "tau": account.tau,
        "top_k": account.top_k,
        "tokens_used": account.tokens_used,
        "topk_plus_mean": account.topk_plus_mean,
        "topk_plus_std": account.topk_plus_std,
        "expansion_set_count": account.expansion_set_count,
        "ii_engine": ii_engine,
        "gemma_chunk_size": gemma_chunk_size,
        "n_batch_members": len(texts),
        "raw_output": raw_output,
        "label_parsed": label_parsed,
    }
    return ClusteringResult(
        method="invisible_ink",
        labels=labels,
        assignments=assignments,
        extra=extra,
    )


def run_kmeans(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
    seed: int,
) -> ClusteringResult:
    """Variant 3: spherical k-means on L2-normalised sentence embeddings.

    Embeds the input texts via the project's standard
    ``all-MiniLM-L6-v2`` :class:`Embedder`, L2-normalises so Euclidean
    distance approximates cosine, fits :class:`sklearn.cluster.KMeans`
    with ``random_state=seed`` for reproducibility. The label for each
    cluster is the input text nearest the cluster's centroid (cosine
    nearest, since both are L2-normalised), truncated to the first four
    whitespace-delimited tokens for visual parity with the Amin labels.
    """
    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    norms = np.linalg.norm(text_embeddings, axis=1)
    norms = np.where(norms == 0, 1e-12, norms)
    text_unit = text_embeddings / norms[:, None]

    kmeans = SklearnKMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = [int(a) for a in kmeans.fit_predict(text_unit)]

    centers = kmeans.cluster_centers_
    center_norms = np.linalg.norm(centers, axis=1)
    center_norms = np.where(center_norms == 0, 1e-12, center_norms)
    center_unit = centers / center_norms[:, None]
    sims_to_centers = text_unit @ center_unit.T  # (N, k)

    labels: list[str] = []
    for cluster_idx in range(k):
        nearest_text_idx = int(np.argmax(sims_to_centers[:, cluster_idx]))
        labels.append(_truncate_words(texts[nearest_text_idx], 4))

    extra: dict[str, Any] = {
        "embedding_model": embedder.model_name,
        "seed": seed,
        "inertia": float(kmeans.inertia_),
        "geometry": "spherical (L2-normalised + Euclidean)",
    }
    return ClusteringResult(
        method="kmeans",
        labels=labels,
        assignments=assignments,
        extra=extra,
    )


def run_bertopic(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
    seed: int,
) -> ClusteringResult:
    """Variant 4: BERTopic with KMeans inside and UMAP disabled.

    Swaps BERTopic's default HDBSCAN clusterer for
    :class:`sklearn.cluster.KMeans` and its default UMAP reducer for
    :class:`bertopic.dimensionality.BaseDimensionalityReduction` (an
    identity reducer). The clustering itself then matches
    :func:`run_kmeans` exactly when run on the same pre-computed
    embeddings; the only difference vs ``kmeans`` is that BERTopic
    labels each cluster via c-TF-IDF (top-four discriminative terms in
    the cluster's documents) rather than the nearest-text-to-centroid
    heuristic. Disabling UMAP avoids the small-N pathology where the
    default UMAP ``n_neighbors=15`` cannot run on a 10-row fixture.
    """
    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)

    topic_model = BERTopic(
        embedding_model=embedder.model_name,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=SklearnKMeans(n_clusters=k, random_state=seed, n_init=10),
        top_n_words=4,
        calculate_probabilities=False,
        verbose=False,
    )
    raw_topics, _ = topic_model.fit_transform(texts, embeddings=text_embeddings)
    assignments = [int(t) for t in raw_topics]

    labels: list[str] = []
    for cluster_idx in range(k):
        words = topic_model.get_topic(cluster_idx)
        if words and isinstance(words, list):
            labels.append(" ".join(word for word, _score in words[:4]))
        else:
            labels.append(f"label_{cluster_idx + 1}")

    topic_info_records: list[dict[str, Any]] = []
    try:
        info_df = topic_model.get_topic_info()
        for row in info_df.to_dict(orient="records"):
            topic_info_records.append({key: _jsonable(value) for key, value in row.items()})
    except Exception as exc:  # noqa: BLE001 - get_topic_info is best-effort metadata
        topic_info_records = [{"error": repr(exc)}]

    extra: dict[str, Any] = {
        "embedding_model": embedder.model_name,
        "seed": seed,
        "umap": "disabled (BaseDimensionalityReduction)",
        "clusterer": "sklearn.cluster.KMeans",
        "topic_info": topic_info_records,
    }
    return ClusteringResult(
        method="bertopic",
        labels=labels,
        assignments=assignments,
        extra=extra,
    )


def run_bertopic_pure(
    texts: list[str],
    *,
    embedder: Embedder,
    seed: int,
) -> ClusteringResult:
    """Variant 5: BERTopic with its default UMAP + HDBSCAN backends.

    Mirrors a hands-off ``BERTopic()`` call, with the minimum parameter
    overrides needed for the pipeline to run at small N: ``UMAP`` falls
    back to ``n_neighbors=N - 1`` and ``n_components=min(5, N - 1)`` when
    ``N < 16``; ``HDBSCAN`` falls back to ``min_cluster_size=max(2, N //
    3)`` when ``N < 30``. At ``N >= 30`` both stages run at BERTopic's
    published defaults (``UMAP(n_neighbors=15, n_components=5,
    min_dist=0.0, metric='cosine')`` and ``HDBSCAN(min_cluster_size=10,
    metric='euclidean')``).

    Unlike the other four methods, the cluster count is auto-discovered
    by HDBSCAN rather than pinned to a caller-supplied ``k``. Documents
    that HDBSCAN flags as outliers (topic id ``-1``) are bucketed as a
    synthetic last cluster labelled ``"outliers (HDBSCAN)"`` so the
    print and JSONL paths can iterate over a fixed cluster list without
    a special case. The result's ``labels`` length is the discovered
    cluster count and may differ from the ``--k`` the user passed; the
    JSONL record reports both ``k`` (actual) and ``k_requested`` (the
    flag value) so a downstream reader can spot the mismatch.

    Expect a single-cluster collapse or an all-outlier result on the
    10-row toy fixture: BERTopic's defaults are tuned for hundreds of
    documents, and this method exists to register that fact. The
    meaningful comparison against the other four methods happens once
    the input scales to the WP3.4 pilot corpus size.
    """
    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    n = len(texts)
    n_neighbors = max(2, min(15, n - 1))
    n_components = max(2, min(5, n - 1))
    min_cluster_size = max(2, min(10, n // 3 if n >= 6 else 2))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedder.model_name,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=4,
        calculate_probabilities=False,
        verbose=False,
    )
    raw_topics, _ = topic_model.fit_transform(texts, embeddings=text_embeddings)
    raw_topics_int = [int(t) for t in raw_topics]

    # HDBSCAN may emit -1 (outlier) plus a set of dense topic ids. Re-map
    # those ids to a contiguous 0..K_actual-1 range, with -1 (if present)
    # pinned to the last slot so the "outlier" bucket is always at the
    # tail of the cluster list rather than buried among real topics.
    unique_topics = sorted(set(raw_topics_int))
    has_outliers = -1 in unique_topics
    real_topics = [t for t in unique_topics if t != -1]
    ordered_topics: list[int] = list(real_topics)
    if has_outliers:
        ordered_topics.append(-1)
    topic_to_dense = {t: i for i, t in enumerate(ordered_topics)}
    assignments = [topic_to_dense[t] for t in raw_topics_int]

    labels: list[str] = []
    for orig_topic in ordered_topics:
        if orig_topic == -1:
            labels.append("outliers (HDBSCAN)")
            continue
        words = topic_model.get_topic(orig_topic)
        if words and isinstance(words, list):
            labels.append(" ".join(word for word, _score in words[:4]))
        else:
            labels.append(f"topic_{orig_topic}")

    topic_info_records: list[dict[str, Any]] = []
    try:
        info_df = topic_model.get_topic_info()
        for row in info_df.to_dict(orient="records"):
            topic_info_records.append({key: _jsonable(value) for key, value in row.items()})
    except Exception as exc:  # noqa: BLE001 - get_topic_info is best-effort metadata
        topic_info_records = [{"error": repr(exc)}]

    extra: dict[str, Any] = {
        "embedding_model": embedder.model_name,
        "seed": seed,
        "umap": {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "metric": "cosine",
        },
        "hdbscan": {
            "min_cluster_size": min_cluster_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
        "discovered_topics": ordered_topics,
        "n_outliers": sum(1 for t in raw_topics_int if t == -1),
        "topic_info": topic_info_records,
    }
    return ClusteringResult(
        method="bertopic_pure",
        labels=labels,
        assignments=assignments,
        extra=extra,
    )


def _jsonable(value: Any) -> Any:
    """Coerce numpy / pandas scalars into JSON-serialisable Python primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    return value


def _members_per_cluster(assignments: list[int], k: int) -> list[list[int]]:
    """Bucket input indices by their cluster assignment, preserving input order."""
    buckets: list[list[int]] = [[] for _ in range(k)]
    for idx, cluster_idx in enumerate(assignments):
        if 0 <= cluster_idx < k:
            buckets[cluster_idx].append(idx)
    return buckets


_FALLBACK_PREFIX = "label_"


def _is_fallback_label(label: str) -> bool:
    """Recognise the ``label_<i>`` literal emitted by parse_json_labels fallbacks."""
    if not label.startswith(_FALLBACK_PREFIX):
        return False
    suffix = label[len(_FALLBACK_PREFIX) :]
    return suffix.isdigit()


def print_clusters(result: ClusteringResult, texts: list[str]) -> None:
    """Pretty-print one method's clusters: label per cluster, then member texts."""
    print()
    print("=" * 72)
    print(f"Method: {result.method}  (k={len(result.labels)})")
    print("-" * 72)
    buckets = _members_per_cluster(result.assignments, len(result.labels))
    for cluster_idx, label in enumerate(result.labels):
        tag = "  [fallback]" if _is_fallback_label(label) else ""
        print(f"Cluster {cluster_idx}: {label!r}{tag}")
        members = buckets[cluster_idx]
        if not members:
            print("  (empty)")
            continue
        for member_idx in members:
            print(f"  - {texts[member_idx]!r}")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _result_to_record(
    result: ClusteringResult,
    *,
    k_requested: int,
    n_texts: int,
    created_at: str,
) -> dict[str, Any]:
    """Serialise one :class:`ClusteringResult` into the JSONL record shape.

    ``k_requested`` is the ``--k`` the user passed; ``k`` in the
    returned record is the *actual* cluster count (``len(result.labels)``),
    which for every method except ``bertopic_pure`` equals
    ``k_requested`` by construction. The mismatch flag
    ``k_matches_requested`` makes the downstream JSONL reader's job
    explicit.
    """
    k_actual = len(result.labels)
    buckets = _members_per_cluster(result.assignments, k_actual)
    return {
        "method": result.method,
        "k": k_actual,
        "k_requested": k_requested,
        "k_matches_requested": k_actual == k_requested,
        "n_texts": n_texts,
        "labels": result.labels,
        "assignments": result.assignments,
        "members_per_cluster": buckets,
        "extra": result.extra,
        "created_at": created_at,
    }


def _write_jsonl(records: list[dict[str, Any]]) -> None:
    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUTS_PATH.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help=f"Cluster count for every method (default: {K_DEFAULT}).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        choices=list(ALL_METHODS),
        help=f"Subset of methods to run (default: all of {ALL_METHODS}).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            f"Target epsilon for amin_dp; solve_r picks the largest r that fits. "
            f"Mutually exclusive with --r. Default (if neither given): {EPSILON_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
        help=(
            "Private-token budget for amin_dp (fixed). Mutually exclusive with "
            "--epsilon. Realised epsilon is reported via epsilon_from_rho."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help=f"Seed for k-means and BERTopic clusterers (default: {SEED_DEFAULT}).",
    )
    parser.add_argument(
        "--gemma-chunk-size",
        type=int,
        default=GEMMA_CHUNK_SIZE_DEFAULT,
        help=(
            "Max sensitive prompts per Gemma forward pass for amin_nodp, "
            "for amin_dp when the production KV-cache path OOMs, and for "
            "invisible_ink when its production KV-cache path OOMs "
            f"(default: {GEMMA_CHUNK_SIZE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--ii-epsilon",
        type=float,
        default=EPSILON_II_DEFAULT,
        help=(
            f"Target epsilon for invisible_ink (default: {EPSILON_II_DEFAULT}). "
            "Pass 200 to match amin_dp's default budget for a like-for-like "
            "comparison; 10 is the authors' recommended operating point."
        ),
    )
    parser.add_argument(
        "--ii-topk",
        type=int,
        default=TOP_K_II_DEFAULT,
        help=f"Top-k+ truncation parameter for invisible_ink (default: {TOP_K_II_DEFAULT}).",
    )
    parser.add_argument(
        "--ii-tau",
        type=float,
        default=TAU_II_DEFAULT,
        help=f"Sampling temperature for invisible_ink (default: {TAU_II_DEFAULT}).",
    )
    parser.add_argument(
        "--ii-batch-size",
        type=int,
        default=None,
        help=(
            "Private-reference count B for invisible_ink. Defaults to "
            "len(texts) so the whole fixture contributes (matches amin_dp's S)."
        ),
    )
    parser.add_argument(
        "--ii-max-tokens",
        type=int,
        default=MAX_TOTAL_TOKENS,
        help=(
            f"Token budget T for invisible_ink (also used to calibrate C; "
            f"default: {MAX_TOTAL_TOKENS}, matching the Amin max)."
        ),
    )
    parser.add_argument(
        "--ii-delta",
        type=float,
        default=DELTA_II_DEFAULT,
        help=f"Delta for invisible_ink (epsilon, delta)-DP (default: {DELTA_II_DEFAULT}).",
    )
    args = parser.parse_args()

    if args.epsilon is not None and args.r is not None:
        parser.error("Pass --epsilon OR --r, not both.")
    if args.k < 1:
        parser.error("--k must be >= 1.")
    if args.gemma_chunk_size < 1:
        parser.error("--gemma-chunk-size must be >= 1.")
    if args.ii_topk < 1:
        parser.error("--ii-topk must be >= 1.")
    if args.ii_max_tokens < 1:
        parser.error("--ii-max-tokens must be >= 1.")
    if not (0.0 < args.ii_delta < 1.0):
        parser.error("--ii-delta must be in (0, 1).")

    df = pd.read_csv(EXAMPLES_PATH)
    if "text" not in df.columns:
        parser.error(f"{EXAMPLES_PATH} must have a `text` column.")
    texts: list[str] = [str(t) for t in df["text"].tolist()] * TOY_REPEAT
    if not texts:
        parser.error(f"{EXAMPLES_PATH} is empty.")

    ii_batch_size = args.ii_batch_size if args.ii_batch_size is not None else len(texts)
    if ii_batch_size < 1:
        parser.error("--ii-batch-size must be >= 1.")
    if ii_batch_size > len(texts):
        parser.error(f"--ii-batch-size ({ii_batch_size}) exceeds number of texts ({len(texts)}).")
    ii_texts = texts[:ii_batch_size]

    print(
        f"Loaded {len(texts)} texts from {EXAMPLES_PATH} "
        f"({len(texts) // TOY_REPEAT} unique rows × {TOY_REPEAT})."
    )
    print(
        f"Running methods: {args.methods} at k={args.k} "
        f"(gemma_chunk_size={args.gemma_chunk_size})."
    )

    embedder = Embedder()
    results: list[ClusteringResult] = []

    if "amin_nodp" in args.methods:
        print()
        print("Running amin_nodp (privacy-free Amin)...")
        results.append(
            run_amin_privacy_free(
                texts,
                k=args.k,
                embedder=embedder,
                gemma_chunk_size=args.gemma_chunk_size,
            )
        )

    if "amin_dp" in args.methods:
        print()
        print("Running amin_dp (production DP Amin)...")
        results.append(
            run_amin_dp(
                texts,
                k=args.k,
                embedder=embedder,
                epsilon=args.epsilon,
                r_override=args.r,
                gemma_chunk_size=args.gemma_chunk_size,
            )
        )

    if "kmeans" in args.methods:
        print()
        print("Running kmeans (spherical KMeans on sentence embeddings)...")
        results.append(run_kmeans(texts, k=args.k, embedder=embedder, seed=args.seed))

    if "bertopic" in args.methods:
        print()
        print("Running bertopic (KMeans inside, UMAP disabled)...")
        results.append(run_bertopic(texts, k=args.k, embedder=embedder, seed=args.seed))

    if "bertopic_pure" in args.methods:
        print()
        print("Running bertopic_pure (default UMAP + HDBSCAN, --k ignored)...")
        results.append(run_bertopic_pure(texts, embedder=embedder, seed=args.seed))

    if "invisible_ink" in args.methods:
        print()
        print(
            f"Running invisible_ink (DClip + Top-k+, "
            f"b={ii_batch_size}, epsilon={args.ii_epsilon})..."
        )
        results.append(
            run_invisible_ink(
                ii_texts,
                k=args.k,
                embedder=embedder,
                epsilon=args.ii_epsilon,
                top_k=args.ii_topk,
                b=ii_batch_size,
                tau=args.ii_tau,
                max_tokens=args.ii_max_tokens,
                delta=args.ii_delta,
                gemma_chunk_size=args.gemma_chunk_size,
            )
        )

    for result in results:
        # invisible_ink may run on a prefix of texts when --ii-batch-size < N.
        member_texts = ii_texts if result.method == "invisible_ink" else texts
        print_clusters(result, member_texts)

    created_at = _utcnow_iso()
    records = [
        _result_to_record(
            r,
            k_requested=args.k,
            n_texts=len(ii_texts) if r.method == "invisible_ink" else len(texts),
            created_at=created_at,
        )
        for r in results
    ]
    _write_jsonl(records)
    print()
    print(f"Wrote {len(records)} records to {OUTPUTS_PATH} (overwrite mode).")


if __name__ == "__main__":
    main()
