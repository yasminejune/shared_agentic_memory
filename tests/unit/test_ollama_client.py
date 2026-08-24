"""Request payload and error propagation for OllamaClient, over a mock transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent_memories.services.ollama_client import OllamaClient

pytestmark = pytest.mark.unit


def _client_with(handler: Any, **kwargs: Any) -> OllamaClient:
    """Client whose HTTP transport is swapped for the given handler."""
    kwargs.setdefault("model", "qwen3:8b")
    kwargs.setdefault("seed", 3006)
    client = OllamaClient(**kwargs)
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _capture(captured: dict[str, Any], content: str = "ok") -> Any:
    """Handler that records the outgoing request and replies with content."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    return handler


def test_chat_returns_completion_on_success() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_capture(captured, "click [e1]"))

    result = client.chat("sys", "usr")

    assert result == "click [e1]"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["model"] == "qwen3:8b"
    assert captured["json"]["think"] is False
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"] == {"num_predict": 64, "temperature": 0.0, "seed": 3006}
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_constructor_options_reach_the_payload() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_capture(captured), seed=99, num_predict=128, think=True)

    client.chat("sys", "usr")

    assert captured["json"]["think"] is True
    assert captured["json"]["options"]["num_predict"] == 128
    assert captured["json"]["options"]["seed"] == 99


def test_max_tokens_kwarg_overrides_constructor_default() -> None:
    captured: dict[str, Any] = {}
    client = _client_with(_capture(captured))

    client.chat("sys", "usr")
    assert captured["json"]["options"]["num_predict"] == 64

    client.chat("sys", "usr", max_tokens=512)
    assert captured["json"]["options"]["num_predict"] == 512


def test_chat_ignores_thinking_field_and_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "thinking": "I should click the search button.",
                    "content": "click('53')",
                }
            },
        )

    assert _client_with(handler).chat("sys", "usr") == "click('53')"


def test_chat_returns_empty_string_when_content_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant"}})

    assert _client_with(handler).chat("sys", "usr") == ""


@pytest.mark.parametrize("status_code", [404, 500])
def test_chat_propagates_http_errors_without_retry(status_code: int) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(status_code, json={"error": "boom"})

    client = _client_with(handler)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat("sys", "usr")

    assert attempts["n"] == 1
