"""zCDP accounting for Amin et al. Theorem 1: rho, epsilon, r and the delta check."""

from __future__ import annotations

import math

import pytest

from agent_memories.agent.amin_et_al.accounting import (
    check_delta,
    delta_from_rho_epsilon,
    epsilon_from_rho,
    get_epsilon,
    rho_for,
    solve_r,
)

pytestmark = pytest.mark.unit

S = 10
C = 10.0
TAU = 1.5
SIGMA = 1.0
DELTA = 1.0 / S


def _eps(r: int) -> float:
    return epsilon_from_rho(rho_for(r, S, C, TAU, SIGMA), DELTA)


def test_rho_for_matches_theorem1_closed_form() -> None:
    r = 7
    expected = r * (0.5 * (C / (S * TAU)) ** 2 + 2.0 / (S * SIGMA) ** 2)
    assert rho_for(r, S, C, TAU, SIGMA) == pytest.approx(expected, rel=1e-12)


def test_rho_is_linear_in_r() -> None:
    assert rho_for(6, S, C, TAU, SIGMA) == pytest.approx(3 * rho_for(2, S, C, TAU, SIGMA))


def test_epsilon_from_rho_matches_closed_form() -> None:
    rho = 0.25
    expected = rho + math.sqrt(4.0 * rho * math.log(1.0 / DELTA))
    assert epsilon_from_rho(rho, DELTA) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("rho", [0.0, -1.0])
def test_non_positive_rho_costs_nothing(rho: float) -> None:
    assert epsilon_from_rho(rho, DELTA) == 0.0
    assert delta_from_rho_epsilon(rho, 1.0) == 0.0


def test_get_epsilon_uses_one_over_s_as_delta() -> None:
    epsilon, delta = get_epsilon(5, S, C, TAU, SIGMA)

    assert delta == pytest.approx(1.0 / S)
    assert epsilon == pytest.approx(_eps(5))


def test_solve_r_returns_the_largest_r_inside_the_budget() -> None:
    target = 10.0
    r = solve_r(target, DELTA, s=S, c=C, tau=TAU, sigma=SIGMA)

    assert _eps(r) <= target
    assert _eps(r + 1) > target


def test_solve_r_returns_zero_when_even_one_token_is_too_costly() -> None:
    assert solve_r(_eps(1) / 2, DELTA, s=S, c=C, tau=TAU, sigma=SIGMA) == 0


def test_solve_r_stops_at_r_max() -> None:
    generous = 1e9
    assert solve_r(generous, DELTA, s=S, c=C, tau=TAU, sigma=SIGMA) == 80
    assert solve_r(generous, DELTA, s=S, c=C, tau=TAU, sigma=SIGMA, r_max=5) == 5


def test_solve_r_uncapped_agrees_with_the_cap_when_the_budget_binds_first() -> None:
    kwargs = {"s": S, "c": C, "tau": TAU, "sigma": SIGMA}
    assert solve_r(10.0, DELTA, r_max=None, **kwargs) == solve_r(10.0, DELTA, **kwargs)


@pytest.mark.parametrize("r", [1, 5, 20])
def test_tight_delta_is_below_the_closed_form_delta(r: int) -> None:
    """The closed form is a valid but loose bound, so its delta must have slack."""
    rho = rho_for(r, S, C, TAU, SIGMA)
    assert delta_from_rho_epsilon(rho, epsilon_from_rho(rho, DELTA)) <= DELTA


def test_tight_delta_falls_as_epsilon_rises() -> None:
    rho = rho_for(5, S, C, TAU, SIGMA)
    assert delta_from_rho_epsilon(rho, 8.0) < delta_from_rho_epsilon(rho, 4.0)


def test_check_delta_accepts_a_delta_meeting_all_three_conditions() -> None:
    rho = rho_for(5, S, C, TAU, SIGMA)
    check = check_delta(rho, _eps(5), DELTA, n=S)

    assert check.valid
    assert check.delta_in_domain
    assert check.delta_meets_theorem1
    assert check.delta_within_n_bound
    assert check.delta_max_convention == pytest.approx(1.0 / S)


def test_check_delta_rejects_delta_above_the_one_over_n_convention() -> None:
    rho = rho_for(5, S, C, TAU, SIGMA)
    check = check_delta(rho, _eps(5), DELTA, n=1000)

    assert not check.delta_within_n_bound
    assert not check.valid


def test_check_delta_rejects_delta_below_the_theorem1_bound() -> None:
    rho = rho_for(5, S, C, TAU, SIGMA)
    check = check_delta(rho, _eps(5), 1e-300, n=S)

    assert not check.delta_meets_theorem1
    assert not check.valid


@pytest.mark.parametrize("delta", [0.0, -0.1, 1.5])
def test_check_delta_rejects_delta_outside_the_unit_interval(delta: float) -> None:
    rho = rho_for(5, S, C, TAU, SIGMA)
    check = check_delta(rho, _eps(5), delta, n=S)

    assert not check.delta_in_domain
    assert math.isnan(check.delta_min), "the tight bound is undefined off the domain"
    assert not check.valid


@pytest.mark.parametrize("n", [0, -1])
def test_check_delta_rejects_a_non_positive_dataset_size(n: int) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        check_delta(0.5, 1.0, DELTA, n=n)
