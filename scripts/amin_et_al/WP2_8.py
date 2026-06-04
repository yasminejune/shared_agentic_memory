"""WP2 round-1 label-generation standalone harness (WP2-plan §3.2).

Reads the 10 toy memories from ``scripts/amin_et_al/examples.csv``,
runs the existing Amin et al. Algorithm 1 mechanism
(:func:`agent_memories.agent.privacy.generate`) once, and produces
``k`` DP-released topic labels that summarise what the batch has in
common. Each invocation generates fresh labels and overwrites
``scripts/amin_et_al/outputs/labels.jsonl`` so a downstream round-2
caller always sees the most recent label set.

Round 1 reuses the same Amin per-token loop as round 2; the only
difference is the prompt template. ``generate`` is called with
``wrap_fn=wrap_label`` and ``k=K``, which selects the WP2-plan §3.2
numbered-list template :data:`agent_memories.agent.privacy.prompts.LABEL_PROMPT`
instead of the round-2 :data:`~agent_memories.agent.privacy.prompts.GENERIC_PROMPT`
lesson template. The label template ends literally on ``Topics:\\n1.``
so the first sampled token continues label 1's text; subsequent
labels are produced by the model emitting ``\\n2.``, ``\\n3.`` and
so on.

Parsing: the raw DP output is the continuation after the consumed
``Topics:\\n1.`` anchor, so the parser prepends the literal ``"1."``
back, splits on newlines, and matches each line against
``^\\s*(\\d+)\\.\\s*(.+?)\\s*$``. Slots that do not yield a clean
label fall back to the literal string ``label_<i>`` per WP2-plan
§3.2; the round-2 pipeline still receives ``k`` entries so its
batch assignment does not silently lose the round-1 privacy spend.

Production WP2 round 1 will live in
``src/agent_memories/generalisation/labelling.py`` (WP2-plan §6.1)
and consume real WP1.6 per-user memories. This script is the
demo-harness equivalent that ``WP2_3.py`` already is for round 2;
the prompt template, the parser, and the fallback policy are all
reusable from here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent_memories.agent.privacy import (
    check_delta,
    epsilon_from_rho,
    generate,
    rho_for,
    solve_r,
)
from agent_memories.agent.privacy.prompts import wrap_label
from agent_memories.generalisation import parse_numbered_labels as parse_labels

S = 10  # batch size (10 examples)
C = 50.0  # logit clip
TAU = 1.0  # private temperature
TAU_PUBLIC = 1.5  # public temperature
SIGMA = 0.1  # SVT noise scale; see module docstring for rationale
THETA = 0.0  # SVT threshold; see module docstring for rationale
R_MAX = 80  # cap passed to solve_r
EPSILON_DEFAULT = 200.0
K_DEFAULT = 5

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
OUTPUTS_PATH = Path("scripts/amin_et_al/outputs/labels.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help=f"Number of labels to generate (default: {K_DEFAULT}).",
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
            "The realised epsilon is reported via epsilon_from_rho."
        ),
    )
    args = parser.parse_args()

    if args.epsilon is not None and args.r is not None:
        parser.error("Pass --epsilon OR --r, not both.")

    df = pd.read_csv(EXAMPLES_PATH)
    items_blocks = df["text"].tolist()

    delta = 1.0 / len(items_blocks)  # Appendix C convention: delta = 1/n

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
    print(f"  in (0, 1]                = {chk.delta_in_domain}")
    print(f"  meets Theorem 1 tight    = {chk.delta_meets_theorem1}")
    print(f"  within Appendix C bound  = {chk.delta_within_n_bound}")
    print(f"  valid (all three)        = {chk.valid}")
    if not chk.valid:
        if not chk.delta_in_domain:
            print("  -> delta must be in (0, 1]; check the delta computation.")
        if not chk.delta_meets_theorem1:
            print("  -> Reduce r, increase sigma/tau, or raise the privacy budget.")
        if not chk.delta_within_n_bound:
            print(
                "  -> delta is too large for the dataset size; "
                f"choose delta <= 1/n = {chk.delta_max_convention:.3e}."
            )

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
        wrap_fn=wrap_label,
        k=args.k,
    )

    print()
    print("Raw DP output (continuation after the prompt's `1.` anchor):")
    print(repr(synthetic))

    parsed_labels = parse_labels(synthetic, args.k)

    print()
    print(f"Parsed labels (k={args.k}):")
    for idx, (text, was_parsed) in enumerate(parsed_labels, start=1):
        tag = "" if was_parsed else "  [fallback]"
        print(f"  {idx}. {text}{tag}")
    n_fallback = sum(1 for _, p in parsed_labels if not p)
    if n_fallback:
        print(f"Note: {n_fallback} / {args.k} labels filled with `label_<i>` fallback.")

    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with OUTPUTS_PATH.open("w", encoding="utf-8") as fh:
        for idx, (text, was_parsed) in enumerate(parsed_labels, start=1):
            record = {
                "index": idx,
                "label": text,
                "parsed": was_parsed,
                "created_at": created_at,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print()
    print(f"Wrote {args.k} labels to {OUTPUTS_PATH} (overwrite mode).")

    print()
    print("Privacy account:")
    print(f"  realised epsilon       = {account.epsilon:.4f}")
    print(f"  rho                    = {account.rho:.4f}")
    print(f"  private tokens used    = {account.private_tokens_used} / {account.r}")
    print(f"  public tokens used     = {account.public_tokens_used}")


if __name__ == "__main__":
    main()
