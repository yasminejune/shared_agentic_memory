"""Bucketing ablation on the 10-row toy fixture.

Same texts, same k, four labelling methods plus a hands-off BERTopic
run. I wanted to see whether Amin labels (with and without DP noise)
look anything like k-means or c-TF-IDF before committing to DP labels
as the round-1 output.

* amin_nodp - Amin Algorithm 1 with wrap_label and parse_json_labels,
  but no Laplace, no SVT, no r budget. Labels from averaging the
  inputs, nothing else. Same clip-mean-softmax-multinomial loop as
  scripts/amin_et_al/simplified_amin.py, with wrap_label instead of
  wrap.
* amin_dp - production round 1 (same call shape as
  scripts/amin_et_al/WP2_8.py). Pair with amin_nodp to see how much
  the clusters move under DP noise alone.
* kmeans - spherical k-means on all-MiniLM-L6-v2 embeddings
  (L2-normalised, so Euclidean ~ cosine, same geometry as round-2
  assignment). Cluster label is the nearest input text, truncated
  to four words.
* bertopic - BERTopic with KMeans(n_clusters=k) and UMAP disabled.
  Clustering matches kmeans on the same embeddings; the extra is
  c-TF-IDF labels (top four terms). Difference vs kmeans is the
  labelling layer, not the partition.
* bertopic_pure - default UMAP + HDBSCAN, k not pinned. Fine on
  N >= 30; on the toy fixture it usually collapses to one cluster
  or all outliers. That is the point of keeping it.

TopicDP is out of scope. This harness is cluster structure on a
small fixture at matched k, not a privacy-utility sweep.

Point --examples at a larger id,text CSV if you want; retune S to
the new N. R_MAX and MAX_TOTAL_TOKENS do not depend on N.

Writes one JSONL record per method to OUTPUTS_PATH (overwrite).
Does not touch src/.
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
from sklearn.cluster import KMeans as SklearnKMeans
from umap import UMAP

from agent_memories.agent.amin_et_al import (
    check_delta,
    epsilon_from_rho,
    generate,
    rho_for,
    solve_r,
)
from agent_memories.agent.amin_et_al.privatisation import clip_recenter
from agent_memories.agent.lm import token_generation as tg
from agent_memories.agent.lm.prompts import wrap_label
from agent_memories.generalisation import parse_json_labels
from agent_memories.memory import Embedder

# Same operating point as scripts/amin_et_al/WP2_8.py. Raise S if
# --examples points at a larger CSV (Amin bounds use expected batch
# size). R_MAX does not track N.
S = 10
C = 20.0
TAU = 1.0
TAU_PUBLIC = 1.5
SIGMA = 0.1
THETA = 0.0
R_MAX = 80
EPSILON_DEFAULT = 200.0

# Cap on decoded length for the no-DP Amin loop. Independent of N;
# matches scripts/amin_et_al/simplified_amin.py.
MAX_TOTAL_TOKENS = 80

K_DEFAULT = 4
SEED_DEFAULT = 0
ALL_METHODS: tuple[str, ...] = (
    "amin_nodp",
    "amin_dp",
    "kmeans",
    "bertopic",
    "bertopic_pure",
)

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
OUTPUTS_PATH = Path("scripts/labels/outputs/cluster_comparison.jsonl")

MODEL_NAME = "google/gemma-2-2b-it"


@dataclass(frozen=True)
class ClusteringResult:
    """One method's labels, assignments, and leftover metadata.

    ``labels[i]`` is the string for cluster i; ``assignments[j]`` is
    which cluster the j-th input landed in. ``extra`` is method-specific:
    privacy account for amin_dp, inertia for k-means, topic-info for
    BERTopic, raw decoded text for both Amin variants.
    """

    method: str
    labels: list[str]
    assignments: list[int]
    extra: dict[str, Any] = field(default_factory=dict)


def _assign_by_cosine(
    text_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
) -> list[int]:
    """Nearest label per text by cosine similarity.

    L2-normalises both matrices, then argmax over the (N_texts,
    N_labels) sim matrix. Same geometry as
    assign_memories_to_labels, without needing MemoryEntry objects.
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
    """First ``n`` whitespace tokens of ``text``, joined back."""
    return " ".join(text.split()[:n])


def _generate_privacy_free_labels(
    items_blocks: Sequence[str],
    k: int,
    *,
    c: float = C,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> str:
    """No-noise Amin loop, wrap_label instead of wrap.

    Same clip-mean-softmax-multinomial as simplified_amin.generate,
    but the prompt asks for a JSON array of labels rather than a
    content paragraph. No Laplace, no SVT, no r. Returns the raw
    decoded string for parse_json_labels.
    """
    prompts = [wrap_label(items=block, k=k) for block in items_blocks]
    prompt_ids = [tg.encode_chat(p) for p in prompts]

    stop = tg.stop_ids()
    x_ids: list[int] = []
    step = 0
    while len(x_ids) < max_total_tokens:
        Z = tg.get_next_token_logits_from_ids([ids + x_ids for ids in prompt_ids])
        z_bar = clip_recenter(Z, c).mean(dim=0)
        probs = torch.softmax(z_bar, dim=-1)
        tok = int(torch.multinomial(probs, num_samples=1).item())
        x_ids.append(tok)
        print(f"  amin_nodp token {step} sampled")
        if tok in stop:
            break
        step += 1
    return tg.decode(x_ids)


def run_amin_privacy_free(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
) -> ClusteringResult:
    """Privacy-free Amin, then cosine-assign texts to the parsed labels."""
    tg.set_model(MODEL_NAME)
    raw_output = _generate_privacy_free_labels(texts, k=k)

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
) -> ClusteringResult:
    """Production DP Amin round 1, then cosine-assign.

    Same call shape as scripts/amin_et_al/WP2_8.py. Pass either
    ``epsilon`` (solve_r picks r) or ``r_override`` (pin r, report the
    realised epsilon). The account lands in ``extra``.
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

    raw_output, account = generate(
        texts,
        s=S,
        c=C,
        tau=TAU,
        tau_public=TAU_PUBLIC,
        sigma=SIGMA,
        theta=THETA,
        r=r,
        delta=delta,
        wrap_fn=wrap_label,
        k=k,
    )

    parsed = parse_json_labels(raw_output, k)
    labels = [text for text, _ in parsed]
    label_parsed = [flag for _, flag in parsed]

    text_embeddings = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    label_embeddings = np.asarray(embedder.embed_batch(labels), dtype=np.float32)
    assignments = _assign_by_cosine(text_embeddings, label_embeddings)

    extra = {
        "model": MODEL_NAME,
        "realised_epsilon": account.epsilon,
        "r": account.r,
        "delta": account.delta,
        "rho": account.rho,
        "private_tokens_used": account.private_tokens_used,
        "public_tokens_used": account.public_tokens_used,
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


def run_kmeans(
    texts: list[str],
    *,
    k: int,
    embedder: Embedder,
    seed: int,
) -> ClusteringResult:
    """Spherical k-means on L2-normalised sentence embeddings.

    Embed with all-MiniLM-L6-v2, L2-normalise so Euclidean ~ cosine,
    fit sklearn KMeans. Cluster label is the nearest input text to
    the centroid, truncated to four words so it sits next to Amin
    labels without drowning them.
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
    """BERTopic with KMeans inside and UMAP off.

    Identity reducer + sklearn KMeans, so the partition matches
    run_kmeans on the same embeddings. The only extra is c-TF-IDF
    labels (top four terms). UMAP is off because n_neighbors=15
    cannot run on a 10-row fixture.
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
    """BERTopic with its own UMAP + HDBSCAN, k not pinned.

    Small-N fallbacks so the pipeline still runs: UMAP uses
    n_neighbors=N-1 when N < 16; HDBSCAN uses min_cluster_size =
    max(2, N//3) when N < 30. At N >= 30 both stages are the
    published defaults.

    Cluster count is whatever HDBSCAN finds. Outliers (topic -1) are
    a synthetic last cluster labelled "outliers (HDBSCAN)". JSONL
    records both k (actual) and k_requested.

    On the 10-row fixture this usually collapses. That is expected;
    BERTopic's defaults want hundreds of documents.
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

    # HDBSCAN may emit -1 (outlier) plus a set of dense topic ids.
    # Remap to 0..K_actual-1, with -1 last so the outlier bucket sits
    # at the tail rather than among real topics.
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
    """numpy / pandas scalars -> JSON-safe Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    return value


def _members_per_cluster(assignments: list[int], k: int) -> list[list[int]]:
    """Input indices grouped by cluster, input order kept."""
    buckets: list[list[int]] = [[] for _ in range(k)]
    for idx, cluster_idx in enumerate(assignments):
        if 0 <= cluster_idx < k:
            buckets[cluster_idx].append(idx)
    return buckets


_FALLBACK_PREFIX = "label_"


def _is_fallback_label(label: str) -> bool:
    """True if this is the ``label_<i>`` fallback from parse_json_labels."""
    if not label.startswith(_FALLBACK_PREFIX):
        return False
    suffix = label[len(_FALLBACK_PREFIX) :]
    return suffix.isdigit()


def print_clusters(result: ClusteringResult, texts: list[str]) -> None:
    """Print one method: label per cluster, then the member texts."""
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
    """One ClusteringResult as a JSONL record.

    ``k_requested`` is the ``--k`` flag; ``k`` in the record is
    ``len(result.labels)``. They match for every method except
    bertopic_pure. ``k_matches_requested`` makes that obvious.
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
        "--examples",
        type=Path,
        default=EXAMPLES_PATH,
        help=f"CSV with an `id,text` schema (default: {EXAMPLES_PATH}).",
    )
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
    args = parser.parse_args()

    if args.epsilon is not None and args.r is not None:
        parser.error("Pass --epsilon OR --r, not both.")
    if args.k < 1:
        parser.error("--k must be >= 1.")

    df = pd.read_csv(args.examples)
    if "text" not in df.columns:
        parser.error(f"{args.examples} must have a `text` column.")
    texts: list[str] = [str(t) for t in df["text"].tolist()]
    if not texts:
        parser.error(f"{args.examples} is empty.")

    print(f"Loaded {len(texts)} texts from {args.examples}.")
    print(f"Running methods: {args.methods} at k={args.k}.")

    embedder = Embedder()
    results: list[ClusteringResult] = []

    if "amin_nodp" in args.methods:
        print()
        print("Running amin_nodp (privacy-free Amin)...")
        results.append(run_amin_privacy_free(texts, k=args.k, embedder=embedder))

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

    for result in results:
        print_clusters(result, texts)

    created_at = _utcnow_iso()
    records = [
        _result_to_record(r, k_requested=args.k, n_texts=len(texts), created_at=created_at)
        for r in results
    ]
    _write_jsonl(records)
    print()
    print(f"Wrote {len(records)} records to {OUTPUTS_PATH} (overwrite mode).")


if __name__ == "__main__":
    main()
