"""Tests for MistralClient; SDK errors including 429 propagate without retry."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from mistralai.client.errors import SDKError

from agent_memories.services.mistral_client import (
    MissingMistralApiKeyError,
    MistralClient,
)


def _sdk_error(status_code: int) -> SDKError:
    """Build a real SDKError with an httpx response of the given status code."""
    response = httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=b'{"message":"rate limited"}',
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    return SDKError("API error occurred", response)


def _make_client() -> MistralClient:
    """Build a MistralClient that skips the .env / API-key check."""
    return MistralClient(api_key="dummy")


@pytest.mark.unit
def test_chat_returns_completion_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    calls: list[dict[str, Any]] = []

    def fake_complete(messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return "click [e1]"

    monkeypatch.setattr(client, "_complete", fake_complete)

    result = client.chat("sys", "usr")

    assert result == "click [e1]"
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 64
    assert calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


@pytest.mark.unit
def test_chat_max_tokens_kwarg_overrides_constructor_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    seen: dict[str, Any] = {}

    def fake_complete(messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        seen["max_tokens"] = max_tokens
        return "ok"

    monkeypatch.setattr(client, "_complete", fake_complete)

    client.chat("sys", "usr", max_tokens=512)

    assert seen["max_tokens"] == 512


@pytest.mark.unit
def test_chat_propagates_429_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    attempts = {"n": 0}

    def fake_complete(messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        attempts["n"] += 1
        raise _sdk_error(429)

    monkeypatch.setattr(client, "_complete", fake_complete)

    with pytest.raises(SDKError):
        client.chat("sys", "usr")

    assert attempts["n"] == 1, "chat() must call _complete exactly once and re-raise"


@pytest.mark.unit
def test_chat_propagates_non_429_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    attempts = {"n": 0}

    def fake_complete(messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        attempts["n"] += 1
        raise _sdk_error(500)

    monkeypatch.setattr(client, "_complete", fake_complete)

    with pytest.raises(SDKError):
        client.chat("sys", "usr")

    assert attempts["n"] == 1


@pytest.mark.unit
def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # MistralClient does not load .env; the runner does. Unset the env var
    # and check that the constructor refuses to proceed.
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(MissingMistralApiKeyError):
        MistralClient()
