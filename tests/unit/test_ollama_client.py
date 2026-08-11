"""Unit tests for ``OllamaClient``.

The HTTP layer is the seam: we swap the client's ``_http`` for an
``httpx.Client`` backed by ``httpx.MockTransport`` so the tests cover
the request payload, the success path, and the error paths without
needing a live Ollama server. The structure mirrors
``tests/unit/test_mistral_client.py`` so the two clients' failure
behaviour stays observably similar.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent_memories.services.ollama_client import OllamaClient


def _make_client_with_mock(
    handler: Any,
) -> OllamaClient:
    client = OllamaClient(model="qwen3:8b", seed=3006)
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.unit
def test_chat_returns_completion_on_success() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "click [e1]"}},
        )

    client = _make_client_with_mock(handler)

    result = client.chat("sys", "usr")

    assert result == "click [e1]"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["model"] == "qwen3:8b"
    assert captured["json"]["think"] is False
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"]["num_predict"] == 64
    assert captured["json"]["options"]["temperature"] == 0.0
    assert captured["json"]["options"]["seed"] == 3006
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


@pytest.mark.unit
def test_think_true_is_sent_in_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "click('53')"}},
        )

    client = OllamaClient(model="qwen3:8b", seed=3006, think=True)
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(handler))

    client.chat("sys", "usr")

    assert captured["json"]["think"] is True


@pytest.mark.unit
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

    client = _make_client_with_mock(handler)

    assert client.chat("sys", "usr") == "click('53')"


@pytest.mark.unit
def test_chat_returns_empty_string_when_content_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant"}})

    client = _make_client_with_mock(handler)

    assert client.chat("sys", "usr") == ""


@pytest.mark.unit
def test_chat_propagates_http_404_without_retry() -> None:
    """A missing Ollama model returns 404; the Think node must see it."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            404,
            json={"error": "model 'qwen3:8b' not found"},
        )

    client = _make_client_with_mock(handler)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat("sys", "usr")

    assert attempts["n"] == 1


@pytest.mark.unit
def test_chat_propagates_http_500_without_retry() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = _make_client_with_mock(handler)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat("sys", "usr")

    assert attempts["n"] == 1


@pytest.mark.unit
def test_num_predict_is_configurable() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "stop"}},
        )

    client = OllamaClient(model="qwen3:8b", seed=3006, num_predict=128)
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(handler))

    client.chat("sys", "usr")

    assert captured["json"]["options"]["num_predict"] == 128


@pytest.mark.unit
def test_chat_max_tokens_kwarg_overrides_constructor_default() -> None:
    """The WP1.6 pipeline raises ``max_tokens`` per call; the constructor cap stays at 64."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

    client = _make_client_with_mock(handler)

    client.chat("sys", "usr", max_tokens=512)

    assert captured["json"]["options"]["num_predict"] == 512


@pytest.mark.unit
def test_chat_max_tokens_none_falls_back_to_constructor_default() -> None:
    """Omitting ``max_tokens`` must use the constructor's ``num_predict`` (Think default)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

    client = _make_client_with_mock(handler)

    client.chat("sys", "usr")

    assert captured["json"]["options"]["num_predict"] == 64


@pytest.mark.unit
def test_chat_includes_seed_in_options() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

    client = OllamaClient(model="qwen3:8b", seed=99)
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(handler))

    client.chat("sys", "usr")

    assert captured["json"]["options"]["seed"] == 99
