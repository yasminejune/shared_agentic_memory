"""Ollama chat client. Posts to /api/chat so the think field is honoured.

Thinking is off by default (short structured replies). WebArena Think
turns construct the client with think=True.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_NUM_PREDICT = 64
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """Ollama /api/chat client.

    model and seed are bound at construction. base_url defaults to the
    local server.
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
        """Bind model, seed, and optional think flag.

        seed is always sent in the Ollama options payload. There is no
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
        """Send one system+user turn and return the assistant text.

        max_tokens overrides num_predict for this call. When think is
        on, only message.content is returned; the trace is discarded.
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
