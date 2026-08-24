"""Amin et al. Algorithm 1 on the 10 toy memories in examples.csv.

One DP synthetic sentence plus the privacy account. Diagnostic only;
the deployed pipeline is src/agent_memories/generalisation (InvisibleInk).
"""

from __future__ import annotations

import pandas as pd

from agent_memories.agent.amin_et_al import check_delta, generate, rho_for, solve_r

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
