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


def test_generate_microbatched_account_is_self_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_microbatched matches generate's account self-consistency."""
    vocab = 32
    stop_id = 0
    public_steps = {"n": 0}

    def fake_encode_chat(prompt: str) -> list[int]:
        return [1, 2, 3]

    def fake_stop_ids() -> set[int]:
        return {stop_id}

    def fake_decode(ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def fake_from_ids(prompt_ids: list[list[int]]) -> torch.Tensor:
        logits = torch.randn(len(prompt_ids), vocab, dtype=torch.float32)
        logits[:, 7] = 10.0
        if public_steps["n"] >= 3:
            logits[:, stop_id] = 20.0
        if len(prompt_ids) == 1:
            public_steps["n"] += 1
            if public_steps["n"] >= 3:
                logits[:, stop_id] = 20.0
        return logits

    monkeypatch.setattr(gen.tg, "encode_chat", fake_encode_chat)
    monkeypatch.setattr(gen.tg, "stop_ids", fake_stop_ids)
    monkeypatch.setattr(gen.tg, "decode", fake_decode)
    monkeypatch.setattr(gen.tg, "get_next_token_logits_from_ids", fake_from_ids)

    texts = ["memory a", "memory b", "memory c"]
    raw, account = gen.generate_microbatched(
        texts,
        b=len(texts),
        tau=1.0,
        top_k=5,
        max_total_tokens=20,
        target_epsilon=10.0,
        delta=1e-5,
        chunk_size=2,
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


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_generate_microbatched_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    """chunk_size < 1 raises before any token plumbing is touched."""
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        gen.generate_microbatched(
            ["a"],
            b=1,
            tau=1.0,
            top_k=5,
            max_total_tokens=20,
            target_epsilon=10.0,
            delta=1e-5,
            chunk_size=chunk_size,
        )


def test_generate_allows_more_texts_than_accounting_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dissertation §5.2.3: extra bucket members are allowed; account uses b."""
    vocab = 32
    stop_id = 0
    step = {"n": 0}

    def fake_encode_chat(prompt: str) -> list[int]:
        return [1, 2, 3]

    def fake_stop_ids() -> set[int]:
        return {stop_id}

    def fake_decode(ids: list[int]) -> str:
        return "ok"

    def fake_prefill(prompt_ids: list[list[int]]) -> tuple[torch.Tensor, _FakeState]:
        logits = torch.randn(len(prompt_ids), vocab, dtype=torch.float32)
        logits[-1, 7] = 10.0
        return logits, _FakeState(attention_mask=torch.ones(len(prompt_ids), 3))

    def fake_continue(
        state: _FakeState, new_token_ids: list[int]
    ) -> tuple[torch.Tensor, _FakeState]:
        step["n"] += 1
        logits = torch.randn(len(new_token_ids), vocab, dtype=torch.float32)
        logits[-1, 7] = 10.0
        if step["n"] >= 1:
            logits[:, stop_id] = 20.0
        return logits, state

    monkeypatch.setattr(gen.tg, "encode_chat", fake_encode_chat)
    monkeypatch.setattr(gen.tg, "stop_ids", fake_stop_ids)
    monkeypatch.setattr(gen.tg, "decode", fake_decode)
    monkeypatch.setattr(gen.tg, "prefill_padded", fake_prefill)
    monkeypatch.setattr(gen.tg, "continue_batched", fake_continue)

    texts = ["a", "b", "c", "d"]
    _, account = gen.generate(
        texts,
        b=2,
        tau=1.0,
        top_k=5,
        max_total_tokens=8,
        target_epsilon=10.0,
        delta=1e-5,
        wrap_fn=lambda items="(no examples)", **kw: items,
        label="topic",
    )
    assert account.b == 2
    assert account.tokens_used >= 1


def test_generate_rejects_fewer_texts_than_accounting_b() -> None:
    with pytest.raises(ValueError, match="must be >= b="):
        gen.generate(
            ["only-one"],
            b=2,
            tau=1.0,
            top_k=5,
            max_total_tokens=8,
            target_epsilon=10.0,
            delta=1e-5,
            wrap_fn=lambda items="(no examples)", **kw: items,
            label="topic",
        )


def test_generate_with_oom_fallback_uses_microbatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = InvisibleInkAccount(
        epsilon=1.0,
        delta=1e-5,
        rho=0.1,
        t=8,
        b=2,
        c=1.0,
        tau=1.0,
        top_k=5,
        tokens_used=2,
        topk_plus_mean=4.0,
        topk_plus_std=0.0,
        expansion_set_count=0,
    )

    def boom(*_args: Any, **_kwargs: Any) -> tuple[str, InvisibleInkAccount]:
        raise RuntimeError("MPS backend out of memory")

    def ok(*_args: Any, **_kwargs: Any) -> tuple[str, InvisibleInkAccount]:
        return "fallback-text", account

    monkeypatch.setattr(gen, "generate", boom)
    monkeypatch.setattr(gen, "generate_microbatched", ok)

    raw, got, engine = gen.generate_with_oom_fallback(
        ["a", "b"],
        b=2,
        tau=1.0,
        top_k=5,
        max_total_tokens=8,
        target_epsilon=10.0,
        delta=1e-5,
        wrap_fn=lambda items="(no examples)", **kw: items,
        label="topic",
    )
    assert raw == "fallback-text"
    assert got is account
    assert engine == "microbatched"


def test_generate_with_oom_fallback_reraises_other_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> tuple[str, InvisibleInkAccount]:
        raise RuntimeError("something else broke")

    monkeypatch.setattr(gen, "generate", boom)
    with pytest.raises(RuntimeError, match="something else broke"):
        gen.generate_with_oom_fallback(
            ["a"],
            b=1,
            tau=1.0,
            top_k=5,
            max_total_tokens=8,
            target_epsilon=10.0,
            delta=1e-5,
        )
