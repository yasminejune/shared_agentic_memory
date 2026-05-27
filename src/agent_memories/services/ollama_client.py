"""Native Ollama chat wrapper for the WP1.3 Think node.

Posts directly to Ollama's ``/api/chat`` endpoint. The bridge silently
drops the ``think: false`` field, so thinking-capable models like
Qwen3 always emit a ``<think>...</think>`` reasoning block and we
have to size ``max_tokens`` for the reasoning trace rather than for
the short grammar line we actually want. Talking to ``/api/chat``
directly lets us pass ``think: false`` and stay at ``num_predict: 64``
-- one short grammar line, no wasted reasoning tokens, ~0.5 s per
call instead of tens of seconds.

The class satisfies the ``ChatClient`` Protocol defined in
``agent_memories.types`` so it is interchangeable with
:class:`MistralClient` from the Think node's point of view.

Errors: a missing model (Ollama tag not pulled) returns HTTP 404; a
hung request hits ``request_timeout`` and raises ``httpx.TimeoutException``.
Both propagate via ``raise_for_status`` (or the underlying transport)
so the Think node fails the run loudly rather than spinning on a
silent error.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_NUM_PREDICT = 64
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """Minimal native-Ollama chat wrapper.

    The model name (e.g. ``qwen3:8b``) is bound at construction time
    because the WP1 loop sends the same kind of prompt every step.
    ``base_url`` defaults to the local Ollama server; override only
    when running against a remote host.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.num_predict = num_predict
        self._http = httpx.Client(timeout=request_timeout)

    def chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        """Send a single system+user turn and return the assistant text.

        Any ``httpx.HTTPStatusError`` (e.g. 404 for an unpulled model)
        or ``httpx.TimeoutException`` propagates so the Think node
        surfaces the failure rather than silently looping.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.num_predict,
            },
        }
        response = self._http.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        return content or ""
