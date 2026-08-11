"""Native Ollama chat wrapper for the Think node and short structured calls.

Posts directly to Ollama's ``/api/chat`` endpoint so the ``think`` field is
honoured (the OpenAI-compatible bridge silently drops it). Thinking is off
by default: the judge, extractor and label post-processing are short
structured replies that must not spend their ``num_predict`` budget on a
reasoning trace. The WebArena agent runners construct the client with
``think=True`` and raise ``num_predict`` so the trace and the action line
both fit. When thinking is on, ``message.thinking`` holds the trace and
``message.content`` holds the answer; ``chat`` returns ``content`` only.

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

    The model name (e.g. ``qwen3:8b``) and reproducibility ``seed`` are
    bound at construction time because the WP1 loop sends the same kind
    of prompt every step. ``seed`` is always sent in the Ollama
    ``options`` payload. ``base_url`` defaults to the local Ollama
    server; override only when running against a remote host.
    """

    def __init__(
        self,
        *,
        model: str,
        seed: int,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        num_predict: int = DEFAULT_NUM_PREDICT,
        think: bool = False,
    ) -> None:
        """Bind model and reproducibility seed at construction time.

        ``seed`` is always forwarded in the Ollama ``options`` payload.
        Callers must pass an explicit value (typically from
        :func:`agent_memories.config.load_random_seed`); there is no
        silent default.
        """
        self.model = model
        self.seed = seed
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.num_predict = num_predict
        self.think = think
        self._http = httpx.Client(timeout=request_timeout)

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Send a single system+user turn and return the assistant text.

        ``max_tokens`` overrides the constructor's ``num_predict`` for
        a single call. When ``think`` is True, ``num_predict`` must cover
        the reasoning trace and the answer together; ``message.thinking``
        is discarded and only ``message.content`` is returned.

        Any ``httpx.HTTPStatusError`` (e.g. 404 for an unpulled model)
        or ``httpx.TimeoutException`` propagates so the Think node
        surfaces the failure rather than silently looping.
        """
        num_predict = self.num_predict if max_tokens is None else max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": self.think,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "seed": self.seed,
            },
        }
        response = self._http.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        return content or ""
