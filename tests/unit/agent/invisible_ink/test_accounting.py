"""Tests for InvisibleInk privacy accounting."""

from __future__ import annotations

import pytest

from agent_memories.agent.invisible_ink.accounting import (
    clip_for_budget,
    epsilon_for_tokens,
    epsilon_from_rho,
    rho_for_tokens,
    rho_from_epsilon,
)

pytestmark = pytest.mark.unit

T = 80
B = 10
TAU = 1.0
DELTA = 1e-5
EPSILON = 10.0


def test_clip_then_epsilon_recovers_target() -> None:
    c = clip_for_budget(EPSILON, DELTA, T, B, TAU)
    realised = epsilon_for_tokens(T, c, B, TAU, DELTA)
    assert realised == pytest.approx(EPSILON, rel=1e-4, abs=1e-4)


def test_rho_for_tokens_matches_theorem2_closed_form() -> None:
    c = 2.5
    expected = float(T) * 0.5 * (c / (float(B) * TAU)) ** 2
    assert rho_for_tokens(T, c, B, TAU) == pytest.approx(expected, rel=1e-12)


def test_epsilon_for_tokens_agrees_with_cdp_eps_of_rho() -> None:
    c = clip_for_budget(EPSILON, DELTA, T, B, TAU)
    rho = rho_for_tokens(T, c, B, TAU)
    via_adapters = epsilon_for_tokens(T, c, B, TAU, DELTA)
    via_rho = epsilon_from_rho(rho, DELTA)
    assert via_adapters == pytest.approx(via_rho, rel=1e-12)


def test_rho_from_epsilon_round_trips() -> None:
    rho = rho_from_epsilon(EPSILON, DELTA)
    assert epsilon_from_rho(rho, DELTA) == pytest.approx(EPSILON, rel=1e-4, abs=1e-4)


def test_clip_for_budget_scales_with_sqrt_b() -> None:
    c_small = clip_for_budget(EPSILON, DELTA, T, B, TAU)
    c_large = clip_for_budget(EPSILON, DELTA, T, 2 * B, TAU)
    assert c_large / c_small == pytest.approx(2.0, rel=1e-6)
