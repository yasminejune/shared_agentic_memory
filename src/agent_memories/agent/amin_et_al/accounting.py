"""zCDP accounting for Amin et al. (2024) Algorithm 1 (Theorem 1).

There are two forms of the privacy bound:
* Closed form, used by epsilon_from_rho and solve_r:

    epsilon(rho, delta) = rho + sqrt(4 * rho * log(1 / delta))

* Tight form, used by delta_from_rho_epsilon and check_delta:

    delta(rho, epsilon) = inf_{alpha > 1}
        exp((alpha - 1) * (alpha * rho - epsilon)) / (alpha - 1)
        * (1 - 1 / alpha) ** alpha

solve_r inverts the closed form for the largest integer r inside
(target_epsilon, delta). check_delta tests the Theorem 1 domain, the
tight bound, and the Appendix C convention delta <= 1/n.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyAccount:
    """Result of one Amin Algorithm 1 run.

    The first block is the bound's hyperparameters (``r``, ``s``, ``c``,
    ``tau``, ``sigma``); the second is realised token counts, which may
    be below ``r`` if generation stopped early.
    """

    epsilon: float
    delta: float
    rho: float
    r: int
    s: int
    c: float
    tau: float
    sigma: float
    private_tokens_used: int
    public_tokens_used: int


def rho_for(r: int, s: int, c: float, tau: float, sigma: float) -> float:
    """zCDP cost of Algorithm 1: ``r * ((c/(s*tau))^2 / 2 + 2/(s*sigma)^2)``."""
    exp_mech_term = 0.5 * (c / (s * tau)) ** 2
    svt_term = 2.0 / (s * sigma) ** 2
    return r * (exp_mech_term + svt_term)


def get_epsilon(r: int, s: int, c: float, tau: float, sigma: float) -> tuple[float, float]:
    rho = rho_for(r, s, c, tau, sigma)

    delta = 1 / s

    epsilon = epsilon_from_rho(rho, delta)

    return (epsilon, delta)


def epsilon_from_rho(rho: float, delta: float) -> float:
    """Closed form: ``epsilon = rho + sqrt(4 * rho * log(1 / delta))``."""
    if rho <= 0.0:
        return 0.0
    return rho + math.sqrt(4.0 * rho * math.log(1.0 / delta))


def solve_r(
    target_epsilon: float,
    delta: float,
    *,
    s: int,
    c: float,
    tau: float,
    sigma: float,
    r_max: int | None = 80,
) -> int:
    """Largest integer r with Theorem 1 epsilon at most target_epsilon.

    r_max caps the search (default 80). Pass r_max=None to stop
    only when the next r would exceed the budget. Returns 0 when
    even r = 1 is too large.
    """
    best_r = 0
    r = 1
    if r_max is None:
        while True:
            eps = epsilon_from_rho(rho_for(r, s, c, tau, sigma), delta)
            if eps <= target_epsilon:
                best_r = r
                r += 1
            else:
                break
    else:
        while r <= r_max:
            eps = epsilon_from_rho(rho_for(r, s, c, tau, sigma), delta)
            if eps <= target_epsilon:
                best_r = r
                r += 1
            else:
                break
    return best_r


def _log_amin_delta(alpha: float, rho: float, epsilon: float) -> float:
    """Log of the per-alpha summand in Amin Theorem 1's tight delta.

    f(alpha) = exp((alpha-1)(alpha*rho - epsilon)) / (alpha-1)
                 * (1 - 1/alpha) ** alpha``

    Log space keeps the minimisation stable when the exponent is large.
    """
    return (
        (alpha - 1.0) * (alpha * rho - epsilon)
        - math.log(alpha - 1.0)
        + alpha * math.log(1.0 - 1.0 / alpha)
    )


def delta_from_rho_epsilon(rho: float, epsilon: float) -> float:
    """Tight delta from Amin Theorem 1 first statement.

    Smallest delta for which Algorithm 1 is (epsilon, delta)-DP
    at this rho.
    """
    if rho <= 0.0:
        return 0.0
    a, b = 1.0 + 1e-9, 1.0e6
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        c1 = b - phi * (b - a)
        c2 = a + phi * (b - a)
        if _log_amin_delta(c1, rho, epsilon) < _log_amin_delta(c2, rho, epsilon):
            b = c2
        else:
            a = c1
        if (b - a) < 1e-9:
            break
    alpha_star = 0.5 * (a + b)
    return math.exp(_log_amin_delta(alpha_star, rho, epsilon))


@dataclass(frozen=True)
class DeltaCheck:
    """Diagnostic result for check_delta."""

    delta_chosen: float
    rho: float
    epsilon: float
    n: int
    delta_min: float
    delta_max_convention: float
    delta_in_domain: bool
    delta_meets_theorem1: bool
    delta_within_n_bound: bool

    @property
    def valid(self) -> bool:
        # Checks considitions of delta:
        # valid is the conjunction.
        # delta_min is the Theorem 1 tight bound;
        # delta_max_convention is Appendix C's 1 / n
        return self.delta_in_domain and self.delta_meets_theorem1 and self.delta_within_n_bound


def check_delta(rho: float, epsilon: float, delta: float, n: int) -> DeltaCheck:
    """Validate ``delta`` against Amin Theorem 1 and Appendix C.

    Three independent conditions: delta in (0, 1] (Theorem 1
    domain); delta >= delta_from_rho_epsilon(rho, epsilon) (tight
    bound); delta <= 1 / n (Appendix C convention). Returns a
    DeltaCheck; the caller decides whether to abort, warn, or proceed.
    """
    if n <= 0:
        raise ValueError("n (dataset size) must be a positive integer")
    delta_in_domain = 0.0 < delta <= 1.0
    delta_min = delta_from_rho_epsilon(rho, epsilon) if delta_in_domain else math.nan
    delta_max_convention = 1.0 / n
    return DeltaCheck(
        delta_chosen=delta,
        rho=rho,
        epsilon=epsilon,
        n=n,
        delta_min=delta_min,
        delta_max_convention=delta_max_convention,
        delta_in_domain=delta_in_domain,
        delta_meets_theorem1=delta_in_domain and delta >= delta_min,
        delta_within_n_bound=delta <= delta_max_convention,
    )


if __name__ == "__main__":
    # Example script for testing the accounting functions
    s, c, tau, sigma = 10, 10.0, 1.5, 1.0
    delta = 1.0 / s
    n = 10
    print(
        f"Amin et al. Theorem 1 sanity table  (s={s}, c={c}, "
        f"tau={tau}, sigma={sigma}, delta={delta}, n={n})"
    )
    print(f"{'r':>3}  {'rho':>10}  {'epsilon':>10}  {'delta_min':>14}")
    for r in range(1, 21):
        rho = rho_for(r, s, c, tau, sigma)
        eps = epsilon_from_rho(rho, delta)
        d_min = delta_from_rho_epsilon(rho, eps)
        print(f"{r:>3}  {rho:>10.4f}  {eps:>10.4f}  {d_min:>14.3e}")
    print()
    for target in (1.0, 3.0, 5.0, 10.0):
        r = solve_r(target, delta, s=s, c=c, tau=tau, sigma=sigma)
        print(f"solve_r(target_epsilon={target}) -> r = {r}")
    print()
    r = solve_r(10.0, delta, s=s, c=c, tau=tau, sigma=sigma)
    rho = rho_for(r, s, c, tau, sigma)
    eps = epsilon_from_rho(rho, delta)
    chk = check_delta(rho, eps, delta, n=n)
    print(f"check_delta at r={r}, epsilon={eps:.4f}, delta={delta}, n={n}:")
    print(f"  delta_min                 = {chk.delta_min:.3e}")
    print(f"  delta_max_convention 1/n  = {chk.delta_max_convention:.3e}")
    print(f"  delta_in_domain           = {chk.delta_in_domain}")
    print(f"  delta_meets_theorem1      = {chk.delta_meets_theorem1}")
    print(f"  delta_within_n_bound      = {chk.delta_within_n_bound}")
    print(f"  valid                     = {chk.valid}")
