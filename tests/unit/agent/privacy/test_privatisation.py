"""Tests for Amin et al. Algorithm 1 with monkeypatched token plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from agent_memories.agent.privacy import privatisation as pv
from agent_memories.agent.privacy.privacy_accounting import (
    PrivacyAccount,
    epsilon_from_rho,
    rho_for,
    solve_r,
)

pytestmark = pytest.mark.unit

VOCAB = 16
STOP_ID = 0
FAVOURITE_ID = 7


@dataclass
class _FakeState:
    past_key_values: Any = None
    attention_mask: torch.Tensor | None = None


def _wrap(items: str = "(no examples)", **kwargs: Any) -> str:
    return f"{kwargs.get('label')}:{items}"


def _row(favourite: int = FAVOURITE_ID) -> torch.Tensor:
    """One logit row where favourite dominates every other token.

    A 40-logit gap only survives clip_recenter when c is wide, so the
    tests that assert on token identity pass c=50.0 (the value the
    Amin comparison scripts use).
    """
    row = torch.full((VOCAB,), -20.0)
    row[favourite] = 20.0
    return row


def _install_fake_tg(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_after: int | None = None,
) -> dict[str, int]:
    """Patch token_generation so generate runs without a model.

    Every row favours FAVOURITE_ID, so sampling never draws the stop
    token unless stop_after is set, which flips every row to STOP_ID
    once that many continuations have happened.
    """
    steps = {"continues": 0}

    def fake_encode_chat(prompt: str) -> list[int]:
        return [1, 2, 3]

    def fake_stop_ids() -> set[int]:
        return {STOP_ID}

    def fake_decode(ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def fake_prefill(prompt_ids: list[list[int]]) -> tuple[torch.Tensor, _FakeState]:
        logits = torch.stack([_row() for _ in prompt_ids])
        return logits, _FakeState(attention_mask=torch.ones(len(prompt_ids), 3))

    def fake_continue(
        state: _FakeState, new_token_ids: list[int]
    ) -> tuple[torch.Tensor, _FakeState]:
        steps["continues"] += 1
        favourite = FAVOURITE_ID
        if stop_after is not None and steps["continues"] >= stop_after:
            favourite = STOP_ID
        logits = torch.stack([_row(favourite) for _ in new_token_ids])
        return logits, state

    monkeypatch.setattr(pv.tg, "encode_chat", fake_encode_chat)
    monkeypatch.setattr(pv.tg, "stop_ids", fake_stop_ids)
    monkeypatch.setattr(pv.tg, "decode", fake_decode)
    monkeypatch.setattr(pv.tg, "prefill_padded", fake_prefill)
    monkeypatch.setattr(pv.tg, "continue_batched", fake_continue)
    return steps


def _no_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pv, "_laplace", lambda scale: 0.0)


def test_clip_recenter_maps_row_max_to_c() -> None:
    Z = torch.tensor([[0.0, 5.0, -100.0]])  # noqa: N806
    clipped = pv.clip_recenter(Z, c=2.0)
    assert clipped.shape == Z.shape
    assert clipped[0, 1].item() == pytest.approx(2.0)
    assert clipped[0, 0].item() == pytest.approx(-2.0)
    assert clipped[0, 2].item() == pytest.approx(-2.0)


def test_clip_recenter_keeps_every_component_in_band() -> None:
    Z = torch.randn(4, VOCAB) * 50.0  # noqa: N806
    clipped = pv.clip_recenter(Z, c=3.0)
    assert clipped.min().item() >= -3.0 - 1e-6
    assert clipped.max().item() <= 3.0 + 1e-6


def test_clip_recenter_is_per_row() -> None:
    Z = torch.tensor([[0.0, 1.0], [100.0, 101.0]])  # noqa: N806
    clipped = pv.clip_recenter(Z, c=1.0)
    assert clipped[0].tolist() == pytest.approx(clipped[1].tolist())


def test_softmax_l1_distance_is_zero_when_batch_matches_public() -> None:
    z_public = torch.tensor([1.0, 2.0, 3.0])
    Z = z_public.repeat(2, 1)  # noqa: N806
    assert pv.softmax_l1_distance(Z, z_public, s=2) == pytest.approx(0.0, abs=1e-6)


def test_softmax_l1_distance_divides_by_s_not_row_count() -> None:
    """Equation 2 averages over the expected batch size s."""
    z_public = torch.tensor([1.0, 2.0, 3.0])
    Z = z_public.repeat(2, 1)  # noqa: N806
    assert pv.softmax_l1_distance(Z, z_public, s=4) == pytest.approx(0.5, abs=1e-6)


def test_softmax_l1_distance_is_two_for_disjoint_mass() -> None:
    z_public = _row(favourite=1)
    Z = _row(favourite=2).repeat(3, 1)  # noqa: N806
    assert pv.softmax_l1_distance(Z, z_public, s=3) == pytest.approx(2.0, abs=1e-4)


def test_sample_private_picks_the_dominant_token() -> None:
    Z = _row().repeat(3, 1)  # noqa: N806
    assert pv.sample_private(Z, c=50.0, tau=1.0, s=3) == FAVOURITE_ID


def test_sample_public_picks_the_dominant_token() -> None:
    assert pv.sample_public(_row(), tau_public=1.0) == FAVOURITE_ID


def test_laplace_with_zero_scale_is_zero() -> None:
    assert pv._laplace(0.0) == 0.0


def test_generate_rejects_both_r_and_target_epsilon() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        pv.generate(
            ["a"],
            s=2,
            c=1.0,
            tau=1.0,
            tau_public=1.0,
            sigma=0.1,
            theta=0.0,
            r=4,
            target_epsilon=10.0,
            wrap_fn=_wrap,
            label="topic",
        )


def test_generate_rejects_neither_r_nor_target_epsilon() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        pv.generate(
            ["a"],
            s=2,
            c=1.0,
            tau=1.0,
            tau_public=1.0,
            sigma=0.1,
            theta=0.0,
            wrap_fn=_wrap,
            label="topic",
        )


def test_generate_account_is_self_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)

    raw, account = pv.generate(
        ["memory a", "memory b"],
        s=2,
        c=1.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=-10.0,
        r=3,
        delta=1e-5,
        max_total_tokens=20,
        wrap_fn=_wrap,
        label="topic",
    )

    assert isinstance(raw, str)
    assert isinstance(account, PrivacyAccount)
    expected_rho = rho_for(3, 2, 1.0, 1.0, 0.1)
    assert account.rho == pytest.approx(expected_rho, rel=1e-9)
    assert account.epsilon == pytest.approx(epsilon_from_rho(expected_rho, 1e-5), rel=1e-9)
    assert (account.r, account.s, account.c) == (3, 2, 1.0)
    assert (account.tau, account.sigma, account.delta) == (1.0, 0.1, 1e-5)


def test_generate_delta_defaults_to_one_over_s(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)

    _, account = pv.generate(
        ["a"],
        s=8,
        c=1.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=-10.0,
        r=1,
        wrap_fn=_wrap,
        label="topic",
    )
    assert account.delta == pytest.approx(1.0 / 8)


def test_generate_derives_r_from_target_epsilon(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)

    _, account = pv.generate(
        ["a", "b"],
        s=2,
        c=1.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.5,
        theta=-10.0,
        target_epsilon=50.0,
        delta=1e-5,
        max_total_tokens=200,
        wrap_fn=_wrap,
        label="topic",
    )
    expected_r = solve_r(50.0, 1e-5, s=2, c=1.0, tau=1.0, sigma=0.5)
    assert account.r == expected_r
    assert expected_r > 0


def test_generate_takes_private_branch_when_threshold_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """theta below any achievable d_hat forces every token private."""
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)

    raw, account = pv.generate(
        ["a", "b"],
        s=2,
        c=50.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=-10.0,
        r=3,
        delta=1e-5,
        max_total_tokens=20,
        wrap_fn=_wrap,
        label="topic",
    )
    assert account.private_tokens_used == 3
    assert account.public_tokens_used == 0
    assert raw.split() == [str(FAVOURITE_ID)] * 3


def test_generate_takes_public_branch_when_threshold_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """d_hat is an L1 distance between distributions, so it never exceeds 2."""
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)

    raw, account = pv.generate(
        ["a", "b"],
        s=2,
        c=1.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=10.0,
        r=5,
        delta=1e-5,
        max_total_tokens=4,
        wrap_fn=_wrap,
        label="topic",
    )
    assert account.private_tokens_used == 0
    assert account.public_tokens_used == 4
    assert len(raw.split()) == 4


def test_generate_stops_at_max_total_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)

    raw, account = pv.generate(
        ["a", "b"],
        s=2,
        c=1.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=10.0,
        r=100,
        delta=1e-5,
        max_total_tokens=6,
        wrap_fn=_wrap,
        label="topic",
    )
    assert len(raw.split()) == 6
    assert account.private_tokens_used + account.public_tokens_used == 6


def test_generate_breaks_on_stop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tg(monkeypatch, stop_after=2)
    _no_noise(monkeypatch)

    raw, account = pv.generate(
        ["a", "b"],
        s=2,
        c=50.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=-10.0,
        r=50,
        delta=1e-5,
        max_total_tokens=50,
        wrap_fn=_wrap,
        label="topic",
    )
    tokens = raw.split()
    assert tokens == [str(FAVOURITE_ID), str(FAVOURITE_ID), str(STOP_ID)]
    assert account.private_tokens_used == 3


def test_generate_builds_one_public_prompt_beside_the_private_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public prompt uses the wrap_fn default, so it carries no items."""
    _install_fake_tg(monkeypatch)
    _no_noise(monkeypatch)
    seen: list[str] = []

    def recording_wrap(items: str = "(no examples)", **kwargs: Any) -> str:
        seen.append(items)
        return f"{kwargs.get('label')}:{items}"

    pv.generate(
        ["a", "b"],
        s=2,
        c=1.0,
        tau=1.0,
        tau_public=1.0,
        sigma=0.1,
        theta=-10.0,
        r=1,
        delta=1e-5,
        wrap_fn=recording_wrap,
        label="topic",
    )
    assert seen == ["a", "b", "(no examples)"]
