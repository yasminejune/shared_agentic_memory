"""Batch JSON vs sequential EOS label generation under Amin DP.

Same toy memories and the same private-token budget r. json_batch is one
generate call asking for a JSON array. sequential_eos follows Amin
Algorithm 1's nested loop and stops after k sequences.

Writes scripts/amin_et_al/outputs/label_generation_mode_comparison.jsonl.
Harness only.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    sample_private,
    sample_public,
    softmax_l1_distance,
)

S = 100  # expected Amin batch size, not the toy CSV row count
C = 50.0  # logit clip
TAU = 1.0  # private temperature
TAU_PUBLIC = 1.5  # public temperature
SIGMA = 0.1  # SVT noise scale (same as WP2_8.py)
THETA = 0.0  # SVT threshold (same as WP2_8.py)
R_MAX = 80  # cap passed to solve_r
EPSILON_DEFAULT = 200.0
K_DEFAULT = 3
RUNS_DEFAULT = 1
MAX_TOTAL_TOKENS = 256
SEQUENCE_SEPARATOR = "\n---\n"

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
OUTPUTS_PATH = Path("scripts/amin_et_al/outputs/label_generation_mode_comparison.jsonl")

# Frozen snapshot of compare_label_prompts.JSON_SPECIFIC_PROMPT.
JSON_SPECIFIC_PROMPT = (
    "Return exactly {k} labels, one of which must reflect the memory item below.\n"
    "Each label must be 1-4 words.\n"
    "Return only a JSON array of strings.\n"
    "No explanations, numbering, markdown, or extra text.\n"
    "\n"
    "Example:\n"
    '["pricing risk", "customer churn", "supply delay", '
    '"budget pressure", "staff training", "data quality"]\n'
    "\n"
    "Memory items:\n"
    "{items}"
)

SINGLE_SPECIFIC_PROMPT = (
    "Return one short label (1-4 words) that reflects the memory items below.\n"
    "Output only the label text - no JSON, numbering, markdown, or extra text.\n"
    "\n"
    "Memory items:\n"
    "{items}"
)

_QUOTED_STRING = re.compile(r'"([^"\n]+)"')


def wrap_json_specific(items: str = "(no examples)", *, k: int) -> str:
    """Fill JSON_SPECIFIC_PROMPT for one batch member or the public prompt."""
    return JSON_SPECIFIC_PROMPT.format(k=k, items=items)


def wrap_single_specific(items: str = "(no examples)", *, k: int) -> str:
    """Fill SINGLE_SPECIFIC_PROMPT. k is unused (signature parity)."""
    del k
    return SINGLE_SPECIFIC_PROMPT.format(items=items)


def parse_json(raw: str, k: int) -> list[tuple[str, bool]]:
    """JSON-array parser with fallback layers, then label_<i> padding."""
    raw_stripped = raw.strip()
    candidates: list[str] = []
    for repair in ("", "]"):
        try:
            parsed = json.loads(raw_stripped + repair)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            candidates = [x.strip() for x in parsed if isinstance(x, str) and x.strip()]
            if candidates:
                break
    if not candidates:
        candidates = [m.group(1).strip() for m in _QUOTED_STRING.finditer(raw_stripped)]
        candidates = [c for c in candidates if c]
    candidates = candidates[:k]
    return [
        (candidates[i], True) if i < len(candidates) else (f"label_{i + 1}", False)
        for i in range(k)
    ]


def parse_single_label(raw: str) -> tuple[str, bool]:
    """Parse one EOS-terminated sequential label."""
    text = raw.strip().strip('"').strip("'")
    for line in text.splitlines():
        candidate = line.strip().strip('"').strip("'")
        if candidate:
            return candidate, True
    return "", False


def parse_sequential_labels(raw_labels: list[str], k: int) -> list[tuple[str, bool]]:
    """Take the first k sequences from X and pad if fewer were emitted."""
    parsed: list[tuple[str, bool]] = []
    for raw in raw_labels[:k]:
        label, ok = parse_single_label(raw)
        if ok:
            parsed.append((label, True))
        else:
            parsed.append((f"label_{len(parsed) + 1}", False))
    while len(parsed) < k:
        parsed.append((f"label_{len(parsed) + 1}", False))
    return parsed


def _laplace(scale: float) -> float:
    return float(np.random.laplace(0.0, scale))


def _sequence_ends_with_eos(x_ids: list[int], stop: set[int]) -> bool:
    """Whether x ends with eos (or IT turn-end)."""
    return bool(x_ids) and x_ids[-1] in stop


def _clone_prefill_state(state: tg.PrefillState) -> tg.PrefillState:
    """Reset point for a new sequence without re-running prompt prefill."""
    return tg.PrefillState(
        past_key_values=copy.deepcopy(state.past_key_values),
        attention_mask=state.attention_mask.clone(),
    )


def generate_sequential_labels(
    texts: list[str],
    *,
    wrap_fn: Callable[..., str],
    k: int,
    s: int,
    c: float,
    tau: float,
    tau_public: float,
    sigma: float,
    theta: float,
    r: int,
    delta: float,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    **wrap_kwargs: Any,
) -> tuple[list[str], PrivacyAccount]:
    """Amin et al. Algorithm 1 nested loop, adapted for this comparison.

    Stops at k EOS-terminated sequences rather than burning the full r
    budget, caps total tokens, reuses the prompt KV cache, prints progress.
    """
    wrap_kwargs = {**wrap_kwargs, "k": k}
    prompts = [wrap_fn(items=text, **wrap_kwargs) for text in texts]
    public_prompt = wrap_fn(**wrap_kwargs)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)
    stop = tg.stop_ids()
    harness_token_cap = r

    # Line 4: theta_hat = theta + Laplace(sigma)
    theta_hat = theta + _laplace(sigma)
    # Line 5: t = 0
    t = 0
    n_public = 0
    total_tokens = 0
    raw_labels: list[str] = []

    prefill_logits, prefill_state = tg.prefill_padded(prompt_ids + [public_ids])
    print(
        f"  [sequential_eos] prefill done ({len(prompt_ids) + 1} prompts); "
        f"target k={k}, harness token cap={harness_token_cap}"
    )

    # Line 6: while t < r (harness: also stop at k sequences or token cap)
    while t < r and len(raw_labels) < k and total_tokens < harness_token_cap:
        seq_idx = len(raw_labels) + 1
        print(f"  [sequential_eos] sequence {seq_idx}/{k} (private {t}/{r})")

        # Line 7: x = empty token sequence; reuse prompt KV cache
        x_ids: list[int] = []
        logits = prefill_logits.clone()
        state = _clone_prefill_state(prefill_state)

        # Line 8: while x does not end with <eos>
        while not _sequence_ends_with_eos(x_ids, stop):
            if len(x_ids) >= max_total_tokens:
                break
            if total_tokens >= harness_token_cap:
                break

            # Lines 9-10
            Z = logits[: len(prompt_ids)]
            z_public = logits[len(prompt_ids)]

            # Line 11
            d_hat = softmax_l1_distance(Z, z_public, s) + _laplace(2.0 * sigma)

            # Lines 12-16 (private) or 17-18 (public)
            if d_hat >= theta_hat and t < r:
                # Lines 13-14 via sample_private; lines 15-16 inline
                tok = sample_private(Z, c, tau, s)
                t += 1
                theta_hat = theta + _laplace(sigma)
            else:
                tok = sample_public(z_public, tau_public)
                n_public += 1

            # Line 19: append x to x
            x_ids.append(tok)
            total_tokens += 1

            if not _sequence_ends_with_eos(x_ids, stop):
                logits, state = tg.continue_batched(state, [tok] * (len(prompt_ids) + 1))

        # Line 20: X = X union {x}
        decoded = tg.decode(x_ids)
        raw_labels.append(decoded)
        print(
            f"  [sequential_eos] sequence {seq_idx} done "
            f"({len(x_ids)} tokens, private {t}/{r}): {decoded!r}"
        )

    if len(raw_labels) < k:
        print(
            f"  [sequential_eos] stopped early: {len(raw_labels)}/{k} sequences, "
            f"{total_tokens}/{harness_token_cap} harness tokens, private {t}/{r}"
        )

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
    return raw_labels, account


@dataclass(frozen=True)
class GenerationMode:
    """One comparison row: name, runner, parser."""

    name: str
    runner: Callable[..., tuple[str | list[str], PrivacyAccount, int]]
    parser: Callable[..., list[tuple[str, bool]]]


def _run_json_batch(
    items_blocks: list[str],
    *,
    k: int,
    r: int,
    delta: float,
) -> tuple[str, PrivacyAccount, int]:
    synthetic, account = generate(
        items_blocks,
        s=S,
        c=C,
        tau=TAU,
        tau_public=TAU_PUBLIC,
        sigma=SIGMA,
        theta=THETA,
        r=r,
        delta=delta,
        wrap_fn=wrap_json_specific,
        k=k,
    )
    return synthetic, account, 1


def _run_sequential_eos(
    items_blocks: list[str],
    *,
    k: int,
    r: int,
    delta: float,
) -> tuple[str, PrivacyAccount, int]:
    raw_labels, account = generate_sequential_labels(
        items_blocks,
        wrap_fn=wrap_single_specific,
        k=k,
        s=S,
        c=C,
        tau=TAU,
        tau_public=TAU_PUBLIC,
        sigma=SIGMA,
        theta=THETA,
        r=r,
        delta=delta,
    )
    return SEQUENCE_SEPARATOR.join(raw_labels), account, len(raw_labels)


MODES: list[GenerationMode] = [
    GenerationMode("json_batch", _run_json_batch, parse_json),
    GenerationMode(
        "sequential_eos",
        _run_sequential_eos,
        lambda raw, k: parse_sequential_labels(raw.split(SEQUENCE_SEPARATOR) if raw else [], k),
    ),
]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _print_parsed(parsed: list[tuple[str, bool]], k: int) -> None:
    for idx, (text, was_parsed) in enumerate(parsed, start=1):
        tag = "" if was_parsed else "  [fallback]"
        print(f"  {idx}. {text}{tag}")
    n_fallback = sum(1 for _, p in parsed if not p)
    if n_fallback:
        print(f"Note: {n_fallback} / {k} labels filled with `label_<i>` fallback.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help=f"Number of labels per mode (default: {K_DEFAULT}).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            f"Target epsilon; solve_r picks the largest r that fits. "
            f"Mutually exclusive with --r. Default (if neither given): {EPSILON_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
        help=(
            "Private-token budget (fixed). Mutually exclusive with --epsilon. "
            "Realised epsilon is reported via epsilon_from_rho."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_DEFAULT,
        help=f"Runs per mode for stability signal (default: {RUNS_DEFAULT}).",
    )
    args = parser.parse_args()

    if args.epsilon is not None and args.r is not None:
        parser.error("Pass --epsilon OR --r, not both.")
    if args.runs < 1:
        parser.error("--runs must be >= 1.")

    df = pd.read_csv(EXAMPLES_PATH)
    items_blocks = df["text"].tolist()
    items_blocks = items_blocks * 10
    delta = 1.0 / len(items_blocks)

    print(f"Delta: {delta}")

    if args.r is not None:
        r = args.r
        rho = rho_for(r, S, C, TAU, SIGMA)
        realised_epsilon = epsilon_from_rho(rho, delta)
        print(
            f"Privacy assessment: --r supplied directly, r = {r}, "
            f"s={S}, c={C}, tau={TAU}, sigma={SIGMA}, delta={delta} "
            f"-> realised epsilon = {realised_epsilon:.4f}"
        )
    else:
        target_epsilon = args.epsilon if args.epsilon is not None else EPSILON_DEFAULT
        r = solve_r(target_epsilon, delta, s=S, c=C, tau=TAU, sigma=SIGMA, r_max=R_MAX)
        rho = rho_for(r, S, C, TAU, SIGMA)
        realised_epsilon = epsilon_from_rho(rho, delta)
        print(
            f"Privacy assessment: target epsilon = {target_epsilon}, delta = {delta}, "
            f"s={S}, c={C}, tau={TAU}, sigma={SIGMA} "
            f"-> r = {r}, realised epsilon = {realised_epsilon:.4f}"
        )

    if r == 0:
        print("Budget too tight to sample even one private token; aborting.")
        return

    chk = check_delta(rho, realised_epsilon, delta, n=len(items_blocks))
    print()
    print("Amin delta validity (Theorem 1 + Appendix C):")
    print(f"  delta_chosen             = {chk.delta_chosen:.3e}")
    print(f"  delta_min (Theorem 1)    = {chk.delta_min:.3e}")
    print(f"  1 / n     (Appendix C, n={chk.n}) " f"= {chk.delta_max_convention:.3e}")
    print(f"  valid (all three)        = {chk.valid}")
    if not chk.valid:
        print("  -> WARNING: chosen delta does not satisfy Amin Theorem 1 / Appendix C.")
        print("     Continuing anyway so the mode comparison still runs;")
        print("     re-tune sigma / r / epsilon for a privacy-clean run.")

    print()
    print(
        f"Comparing {len(MODES)} generation mode(s) x {args.runs} run(s) "
        f"= {len(MODES) * args.runs} runs."
    )

    all_records: list[dict] = []

    for mode in MODES:
        print()
        print("=" * 72)
        print(f"Mode: {mode.name}")
        print("=" * 72)
        for run_idx in range(args.runs):
            print()
            print(f"--- Run {run_idx + 1} / {args.runs} ---")
            raw_output, account, n_sequences = mode.runner(
                items_blocks,
                k=args.k,
                r=r,
                delta=delta,
            )

            print("Raw DP output:")
            print(repr(raw_output))

            parsed = mode.parser(raw_output, args.k)
            print()
            print(f"Parsed labels (k={args.k}):")
            _print_parsed(parsed, args.k)

            print(
                f"Tokens spent: private = {account.private_tokens_used} / {account.r}, "
                f"public = {account.public_tokens_used}"
            )
            if mode.name == "sequential_eos":
                print(f"EOS-terminated sequences: {n_sequences}")

            all_records.append(
                {
                    "mode": mode.name,
                    "run_idx": run_idx,
                    "k": args.k,
                    "r": r,
                    "realised_epsilon": realised_epsilon,
                    "delta": delta,
                    "raw_output": raw_output,
                    "parsed_labels": [
                        {"index": i, "label": t, "parsed": p}
                        for i, (t, p) in enumerate(parsed, start=1)
                    ],
                    "private_tokens_used": account.private_tokens_used,
                    "public_tokens_used": account.public_tokens_used,
                    "n_sequences": n_sequences,
                    "created_at": _utcnow_iso(),
                }
            )

    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUTS_PATH.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print()
    print(f"Wrote {len(all_records)} records to {OUTPUTS_PATH} (overwrite mode).")


if __name__ == "__main__":
    main()
