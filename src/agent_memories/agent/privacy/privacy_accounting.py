"""Privacy accounting for Amin et al. (2024) Algorithm 1.

Implements the zCDP-to-(epsilon, delta)-DP conversion in Theorem 1 of
Amin et al. 2024 (arXiv:2407.12108).

Theorem 1 has two equivalent statements:

* Closed form (second statement), used by :func:`epsilon_from_rho` and
  :func:`solve_r`::

      epsilon(rho, delta) = rho + sqrt(4 * rho * log(1 / delta))

  Valid for any ``delta in (0, 1]``. Slightly looser than the tight
  bound; convenient because it inverts to an analytic ``r``.

* Tight form (first statement), used by :func:`delta_from_rho_epsilon`
  and :func:`check_delta`::

      delta(rho, epsilon) = inf_{alpha > 1}
          exp((alpha - 1) * (alpha * rho - epsilon)) / (alpha - 1)
          * (1 - 1 / alpha) ** alpha

  Valid for any ``epsilon >= 0``. Numerically minimised over ``alpha``
  via golden-section search on the log-objective.

The :func:`solve_r` function inverts the closed-form bound and returns
the largest integer number of private tokens ``r`` that fits inside the
user's ``(target_epsilon, delta)`` budget. The cap ``r_max=80`` reflects
the project-level default in ``.claude/thesis/WP2-plan.md`` Section 5.

The :func:`check_delta` function verifies that a chosen ``delta``
satisfies three independent conditions:

1. ``delta in (0, 1]`` (Theorem 1 second-form domain).
2. ``delta >= delta_from_rho_epsilon(rho, epsilon)`` (Theorem 1 tight).
3. ``delta <= 1 / n`` where ``n`` is the input dataset size
   (Amin Appendix C "Privacy checklist" convention).

Single-round private prediction only. The two-round composed
(round 1 labelling + round 2 generation) budget covered by WP2-plan
Section 4 lives in the WP2 generalisation pipeline, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyAccount:
    """Result of one run of Amin Algorithm 1.

    Pure-data record. The first block captures the configuration that
    determined the bound (``r``, ``s``, ``c``, ``tau``, ``sigma``); the
    second block captures the realised counts on this particular run
    (``private_tokens_used``, ``public_tokens_used``). The latter may
    be lower than ``r`` if the model emitted an end-of-sequence token
    before the private-token budget was exhausted.
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
    """zCDP cost for Algorithm 1 with the given hyperparameters."""
    exp_mech_term = 0.5 * (c / (s * tau)) ** 2
    svt_term = 2.0 / (s * sigma) ** 2
    return r * (exp_mech_term + svt_term)


def get_epsilon(r: int, s: int, c: float, tau: float, sigma: float) -> tuple[float, float]:
    rho = rho_for(r, s, c, tau, sigma)

    delta = 1 / s

    epsilon = epsilon_from_rho(rho, delta)

    return (epsilon, delta)


def epsilon_from_rho(rho: float, delta: float) -> float:
    """zCDP-to-(epsilon, delta)-DP conversion used by Amin Theorem 1."""
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
    """Return the largest integer ``r`` such that the realised epsilon
    is at most ``target_epsilon``.

    ``r_max`` caps the search when set (default 80, WP2 demo convention).
    Pass ``r_max=None`` for an uncapped search that stops when the next
    ``r`` would exceed ``target_epsilon``.

    Returns 0 when even ``r = 1`` exceeds the budget.
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

    The original expression is::

        f(alpha) = exp((alpha-1)(alpha*rho - epsilon)) / (alpha-1)
                   * (1 - 1/alpha) ** alpha

    Working in log space keeps the minimisation numerically stable when
    ``(alpha-1)(alpha*rho - epsilon)`` is large.
    """
    return (
        (alpha - 1.0) * (alpha * rho - epsilon)
        - math.log(alpha - 1.0)
        + alpha * math.log(1.0 - 1.0 / alpha)
    )


def delta_from_rho_epsilon(rho: float, epsilon: float) -> float:
    """Tight delta from Amin Theorem 1 first statement.

    Returns the smallest ``delta`` for which Algorithm 1 satisfies
    ``(epsilon, delta)``-DP at this ``rho``, computed by minimising the
    expression over ``alpha > 1`` via golden-section search on the
    log-objective. The objective is unimodal on ``(1, infinity)``: it
    diverges to ``+infinity`` at both ends and has a single interior
    minimum.
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
    """Diagnostic result for :func:`check_delta`.

    Each ``bool`` field reports one of the three independent conditions
    on ``delta``; :attr:`valid` is the aggregate. ``delta_min`` is the
    Theorem 1 tight bound and ``delta_max_convention`` is the Appendix C
    convention ``1 / n``.
    """

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
        return self.delta_in_domain and self.delta_meets_theorem1 and self.delta_within_n_bound


def check_delta(rho: float, epsilon: float, delta: float, n: int) -> DeltaCheck:
    """Validate ``delta`` against Amin Theorem 1 and Appendix C.

    Reports three independent conditions:

    1. ``delta in (0, 1]`` (Theorem 1 second-form domain).
    2. ``delta >= delta_from_rho_epsilon(rho, epsilon)`` (Theorem 1
       tight bound is achievable).
    3. ``delta <= 1 / n`` (Amin Appendix C "Privacy checklist"
       convention: the guarantee is only meaningful when ``delta`` is
       below the per-record leakage threshold).

    Returns a :class:`DeltaCheck` so the caller can decide whether to
    abort, warn, or proceed.
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
