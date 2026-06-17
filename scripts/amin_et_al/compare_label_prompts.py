"""Side-by-side comparison of round-1 label-generation prompts under DP.

Runs the same Amin et al. Algorithm 1 mechanism
(:func:`agent_memories.agent.privacy.generate`) on the same 10 toy
memories with the same privacy hyperparameters, swapping only the
prompt template. Each variant runs ``--runs`` times (default 1) so
the qualitative reviewer can see both the raw DP output and the
parsed label list per draw, and judge stability across draws.

All three prompt templates live as script-local constants
(:data:`NUMBERED_PROMPT`, :data:`JSON_PROMPT`,
:data:`JSON_DISCRIMINATIVE_PROMPT`) rather than being imported from
:mod:`agent_memories.agent.privacy.prompts`. This is a deliberate
snapshot: if the production ``LABEL_PROMPT`` is later edited or
replaced when one of the JSON variants is promoted into production,
the comparison harness still holds a frozen copy of every prompt it
has ever tested, so the side-by-side comparison stays reproducible
against the historical baselines. ``NUMBERED_PROMPT`` is a verbatim
snapshot of ``LABEL_PROMPT`` as of 2026-06-04.

The three variants registered here are:

* ``numbered_wp2plan_3.2`` — verbatim snapshot of the WP2-plan §3.2
  numbered-list template (the production
  :data:`agent_memories.agent.privacy.prompts.LABEL_PROMPT` as of
  2026-06-04). Parser: numbered ``N. <label>`` lines with
  ``label_<i>`` fallback.
* ``json_array`` — an experimental template that asks the model for
  a JSON array of strings, with a 6-item example. The ``"exactly six"``
  instruction in the source prompt is templated to ``"exactly {k}"``
  so every variant runs at the same ``--k``; apples-to-apples
  comparison. Parser: ``json.loads`` first, then a fallback regex
  over double-quoted substrings, then ``label_<i>`` padding.
* ``json_discriminative`` — the ``json_array`` template with one
  added anti-redundancy clause in the instruction: each label must
  be on a distinct facet, no synonyms or paraphrases of any other
  label. Same JSON output format, same example, same parser as
  ``json_array``; only the instruction prose differs, so any
  qualitative gap between the two isolates the effect of the
  discriminativity instruction itself (not an example change and
  not an output-format change).

Fairness controls: privacy hyperparameters (``s``, ``c``, ``tau``,
``tau_public``, ``sigma``, ``theta``) are fixed at the same values
as ``WP2_8.py``; ``r`` is computed once from ``--epsilon`` (via
:func:`solve_r`) or supplied directly via ``--r`` and then reused
across every variant and every run. So all variants spend the same
private-token budget on the same data, and only the prompt template
varies.

Output file ``scripts/amin_et_al/outputs/label_prompt_comparison.jsonl``
is overwritten on every invocation (one JSONL record per
(variant, run)) so the reviewer's eye can move between the printed
console block and the persisted record.

This script is a comparison harness and does not promote either
prompt into production. The chosen variant should be added to
:mod:`agent_memories.agent.privacy.prompts` in the same shape as
the existing :data:`agent_memories.agent.privacy.prompts.LABEL_PROMPT`
once the choice is made.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
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

S = 100  # batch size (100 examples)
C = 50.0  # logit clip
TAU = 1.0  # private temperature
TAU_PUBLIC = 1.5  # public temperature
SIGMA = 0.1  # SVT noise scale (matches WP2_8.py rationale)
THETA = 0.0  # SVT threshold (matches WP2_8.py rationale)
R_MAX = 80  # cap passed to solve_r
EPSILON_DEFAULT = 200.0
K_DEFAULT = 3  # matches the JSON variant's native example count
RUNS_DEFAULT = 1

EXAMPLES_PATH = Path("scripts/amin_et_al/examples.csv")
OUTPUTS_PATH = Path("scripts/amin_et_al/outputs/label_prompt_comparison.jsonl")


NUMBERED_PROMPT = (
    "You will be given memory items distilled from one web-navigation\n"
    "trajectory. Suggest k = {k} short topic labels (one to four words\n"
    "each) that together cover the kinds of tasks this trajectory\n"
    "belongs to. Output exactly k lines, each in the format:\n"
    "N. <label>\n"
    "where N is the line number starting at 1.\n"
    "\n"
    "Memory items:\n"
    "{items}\n"
    "\n"
    "Topics:\n"
    "1."
)


JSON_PROMPT = (
    "Return exactly {k} labels.\n"
    "Each label must be 1–4 words.\n"
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


JSON_DISCRIMINATIVE_PROMPT = (
    "Return exactly {k} labels that summarise the memory items below.\n"
    "Each label must be 1–4 words and on a DISTINCT facet — no two labels\n"
    "may be synonyms, paraphrases, or describe the same aspect.\n"
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

JSON_SPECIFIC_PROMPT = (
    "Return exactly {k} labels, one of which must reflect the memory item below.\n"
    "Each label must be 1–4 words.\n"
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


def wrap_numbered(items: str = "(no examples)", *, k: int) -> str:
    """Format :data:`NUMBERED_PROMPT` for one batch member or the SVT public prompt.

    Local counterpart of
    :func:`agent_memories.agent.privacy.prompts.wrap_label` against
    the frozen :data:`NUMBERED_PROMPT` snapshot. Behaviourally
    identical to the production ``wrap_label`` as of 2026-06-04; the
    point of the local copy is to keep this comparison harness
    independent of future edits to the production template.
    """
    return NUMBERED_PROMPT.format(k=k, items=items)


def wrap_json(items: str = "(no examples)", *, k: int) -> str:
    """Format :data:`JSON_PROMPT` for one batch member or the SVT public prompt.

    Sibling of :func:`wrap_numbered`. The default empty-items body
    (``"(no examples)"``) yields the SVT public prompt; passing a
    rendered items block yields the matching private prompt for that
    batch member. ``k`` substitutes into the ``"Return exactly {k}
    labels."`` line so the count instruction matches the ``--k`` the
    user passes.
    """
    return JSON_PROMPT.format(k=k, items=items)


def wrap_json_discriminative(items: str = "(no examples)", *, k: int) -> str:
    """Format :data:`JSON_DISCRIMINATIVE_PROMPT` for one batch member or SVT public.

    Identical shape to :func:`wrap_json` so the comparison harness
    passes one through the other transparently. The only material
    difference vs. :data:`JSON_PROMPT` is the added anti-redundancy
    clause asking for distinct facets; the JSON output format and
    the worked example are unchanged so any qualitative effect is
    attributable to the instruction prose alone.
    """
    return JSON_DISCRIMINATIVE_PROMPT.format(k=k, items=items)


_NUMBERED_LINE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
_QUOTED_STRING = re.compile(r'"([^"\n]+)"')


def wrap_json_specific(items: str = "(no examples)", *, k: int) -> str:
    """Format :data:`JSON_SPECIFIC_PROMPT` for one batch member or SVT public."""
    return JSON_SPECIFIC_PROMPT.format(k=k, items=items)


def parse_numbered(raw: str, k: int) -> list[tuple[str, bool]]:
    """Numbered-list parser (mirrors ``WP2_8.py``).

    The :data:`~agent_memories.agent.privacy.prompts.LABEL_PROMPT`
    ends literally on ``Topics:\\n1.``, so the first sampled token
    is the continuation of label 1. Prepend ``"1."`` back, split on
    newlines, match ``^\\s*(\\d+)\\.\\s*(.+?)\\s*$`` per line, pad
    missing indices with ``label_<i>``.
    """
    full_text = "1." + raw
    found: dict[int, str] = {}
    for line in full_text.splitlines():
        match = _NUMBERED_LINE.match(line)
        if match is None:
            continue
        idx = int(match.group(1))
        text = match.group(2).strip()
        if 1 <= idx <= k and idx not in found and text:
            found[idx] = text
    return [(found[i], True) if i in found else (f"label_{i}", False) for i in range(1, k + 1)]


def parse_json(raw: str, k: int) -> list[tuple[str, bool]]:
    """JSON-array parser with two fallback layers, then ``label_<i>`` padding.

    Layer 1: strict :func:`json.loads` on the stripped raw output.
    Layer 2: same with a trailing ``"]"`` appended (the DP output may
    have been cut off at ``r`` tokens before the closing bracket).
    Layer 3: regex-extract every double-quoted substring as a label
    candidate.

    The first layer that returns at least one non-empty string wins;
    the result is truncated to ``k`` candidates and any missing
    slots are padded with ``label_<i>``.
    """
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


@dataclass(frozen=True)
class PromptVariant:
    """One row in the comparison: name, wrap function, parser.

    ``wrap_fn`` has the same shape as
    :func:`agent_memories.agent.privacy.prompts.wrap_label` so the
    Amin ``generate`` call site is identical across variants; only
    the template content differs.
    """

    name: str
    wrap_fn: Callable[..., str]
    parser: Callable[[str, int], list[tuple[str, bool]]]


VARIANTS: list[PromptVariant] = [
    PromptVariant("numbered_wp2plan_3.2", wrap_numbered, parse_numbered),
    PromptVariant("json_array", wrap_json, parse_json),
    PromptVariant("json_discriminative", wrap_json_discriminative, parse_json),
    PromptVariant("json_specific", wrap_json_specific, parse_json),
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
        help=f"Number of labels per variant (default: {K_DEFAULT}).",
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
        help=f"Runs per variant for stability signal (default: {RUNS_DEFAULT}).",
    )
    args = parser.parse_args()

    if args.epsilon is not None and args.r is not None:
        parser.error("Pass --epsilon OR --r, not both.")
    if args.runs < 1:
        parser.error("--runs must be >= 1.")

    df = pd.read_csv(EXAMPLES_PATH)
    items_blocks = df["text"].tolist()
    items_blocks = items_blocks * 10  # Each example now appears 10 times
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
        print("     Continuing anyway so the prompt comparison still runs;")
        print("     re-tune sigma / r / epsilon for a privacy-clean run.")

    print()
    print(
        f"Comparing {len(VARIANTS)} prompt variant(s) x {args.runs} run(s) "
        f"= {len(VARIANTS) * args.runs} Amin generate calls."
    )

    all_records: list[dict] = []

    for variant in VARIANTS:
        print()
        print("=" * 72)
        print(f"Variant: {variant.name}")
        print("=" * 72)
        for run_idx in range(args.runs):
            print()
            print(f"--- Run {run_idx + 1} / {args.runs} ---")
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
                wrap_fn=variant.wrap_fn,
                k=args.k,
            )

            print("Raw DP output:")
            print(repr(synthetic))

            parsed = variant.parser(synthetic, args.k)
            print()
            print(f"Parsed labels (k={args.k}):")
            _print_parsed(parsed, args.k)

            print(
                f"Tokens spent: private = {account.private_tokens_used} / {account.r}, "
                f"public = {account.public_tokens_used}"
            )

            all_records.append(
                {
                    "variant": variant.name,
                    "run_idx": run_idx,
                    "k": args.k,
                    "r": r,
                    "realised_epsilon": realised_epsilon,
                    "delta": delta,
                    "raw_output": synthetic,
                    "parsed_labels": [
                        {"index": i, "label": t, "parsed": p}
                        for i, (t, p) in enumerate(parsed, start=1)
                    ],
                    "private_tokens_used": account.private_tokens_used,
                    "public_tokens_used": account.public_tokens_used,
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
