"""Smoke test for InvisibleInk Algorithm 1 with monkeypatched token plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from agent_memories.agent.invisible_ink import generation as gen
from agent_memories.agent.invisible_ink.accounting import (
    InvisibleInkAccount,
    epsilon_for_tokens,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeState:
    past_key_values: Any = None
    attention_mask: torch.Tensor | None = None


def test_generate_account_is_self_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate returns an account whose spent epsilon matches tokens_used."""
    vocab = 32
    stop_id = 0
    # Force an early stop after a few tokens so we don't need a real model.
    step = {"n": 0}

    def fake_encode_chat(prompt: str) -> list[int]:
        return [1, 2, 3]

    def fake_stop_ids() -> set[int]:
        return {stop_id}

    def fake_decode(ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def fake_prefill(prompt_ids: list[list[int]]) -> tuple[torch.Tensor, _FakeState]:
        b_plus_pub = len(prompt_ids)
        logits = torch.randn(b_plus_pub, vocab, dtype=torch.float32)
        # Make token 7 the clear favourite on the public row so Top-k+ is non-empty.
        logits[-1, 7] = 10.0
        return logits, _FakeState(attention_mask=torch.ones(b_plus_pub, 3))

    def fake_continue(
        state: _FakeState, new_token_ids: list[int]
    ) -> tuple[torch.Tensor, _FakeState]:
        step["n"] += 1
        b_plus_pub = len(new_token_ids)
        logits = torch.randn(b_plus_pub, vocab, dtype=torch.float32)
        logits[-1, 7] = 10.0
        # After 3 continuations, make the stop token win so the loop ends.
        if step["n"] >= 3:
            logits[:, stop_id] = 20.0
        return logits, state

    monkeypatch.setattr(gen.tg, "encode_chat", fake_encode_chat)
    monkeypatch.setattr(gen.tg, "stop_ids", fake_stop_ids)
    monkeypatch.setattr(gen.tg, "decode", fake_decode)
    monkeypatch.setattr(gen.tg, "prefill_padded", fake_prefill)
    monkeypatch.setattr(gen.tg, "continue_batched", fake_continue)

    texts = ["memory a", "memory b", "memory c"]
    raw, account = gen.generate(
        texts,
        b=len(texts),
        tau=1.0,
        top_k=5,
        max_total_tokens=20,
        target_epsilon=10.0,
        delta=1e-5,
        wrap_fn=lambda items="(no examples)", **kw: f"k={kw.get('k')}:{items}",
        k=3,
    )

    assert isinstance(raw, str)
    assert isinstance(account, InvisibleInkAccount)
    assert account.tokens_used <= account.t
    assert account.tokens_used >= 1
    assert account.b == len(texts)
    expected_eps = epsilon_for_tokens(
        account.tokens_used, account.c, account.b, account.tau, account.delta
    )
    assert account.epsilon == pytest.approx(expected_eps, rel=1e-6)
    assert account.topk_plus_mean >= 1.0
