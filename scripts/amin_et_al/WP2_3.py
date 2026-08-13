"""WP2.3 standalone smoke harness for Amin et al. Algorithm 1.

Reads 10 toy examples from ``scripts/amin_et_al/examples.csv`` and
calls :func:`agent_memories.agent.privacy.generate` once to produce a
single differentially-private synthetic sentence at the script's
``EPSILON`` constant and ``delta = 1 / n`` (the Amin et al. 2024
Appendix C convention; with ``n = 10`` toy examples this is
``delta = 0.1``). Prints the resulting text and the
privacy account. Prompt wrapping is handled inside ``generate`` via
:data:`agent_memories.agent.privacy.prompts.GENERIC_PROMPT`; this
script pre-renders each CSV row as a one-line memory-items block
(the raw review text, since the toy fixture has no title or
description fields under the WP2-plan §3.3 content-only template)
and supplies the static :data:`LABEL` that stands in for what WP2
round 1 would otherwise produce.

This script is intentionally not wired into the real per-user memories
or the WP2 generalisation pipeline; the CSV loader exists *only* to
exercise the mechanism end-to-end on data Gemma can chew on.
Production WP2.3 logic lives in ``src/agent_memories/agent/privacy/``.
"""

from __future__ import annotations

import pandas as pd

from agent_memories.agent.privacy import check_delta, generate, rho_for, solve_r

S = 10  # batch size
C = 10.0  # clip value
TAU = 1.0  # tao private
TAU_PUBLIC = 1.5  # tao public
SIGMA = 0.5  # noise scale
THETA = 0.3  # threshold, the higher the less private the text is
R_MAX = 80  # maximum number of private tokens

EPSILON = 10.0
LABEL = "attending a recent event"


def main() -> None:
    df = pd.read_csv("scripts/amin_et_al/examples.csv")
    items_blocks = df["text"].tolist()

    DELTA = 1 / len(items_blocks)  # setting delta to max possible

    print(f"Delta: {DELTA}")

    r = solve_r(EPSILON, DELTA, s=S, c=C, tau=TAU, sigma=SIGMA, r_max=R_MAX)
    print(
        f"Privacy assessment: target epsilon={EPSILON}, delta={DELTA}, "
        f"s={S}, c={C}, tau={TAU}, sigma={SIGMA} -> r = {r}"
    )
    if r == 0:
        print("Budget too tight to sample even one private token; aborting.")
        return

    rho = rho_for(r, S, C, TAU, SIGMA)
    chk = check_delta(rho, EPSILON, DELTA, n=len(items_blocks))
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
            print("  -> delta must be in (0, 1]; check the demo's DELTA constant.")
        if not chk.delta_meets_theorem1:
            print("  -> Reduce r, increase sigma/tau, or raise the privacy budget.")
        if not chk.delta_within_n_bound:
            print(
                "  -> delta is too large for the dataset size; "
                f"choose delta <= 1/n = {chk.delta_max_convention:.3e}."
            )

    synthetic, account = generate(
        items_blocks,
        label=LABEL,
        s=S,
        c=C,
        tau=TAU,
        tau_public=TAU_PUBLIC,
        sigma=SIGMA,
        theta=THETA,
        r=r,
        delta=DELTA,
    )

    print()
    print("Synthetic text:")
    print(repr(synthetic))
    print()
    print("Privacy account:")
    print(f"  realised epsilon       = {account.epsilon:.4f}")
    print(f"  rho                    = {account.rho:.4f}")
    print(f"  private tokens used    = {account.private_tokens_used} / {account.r}")
    print(f"  public tokens used     = {account.public_tokens_used}")


if __name__ == "__main__":
    main()
