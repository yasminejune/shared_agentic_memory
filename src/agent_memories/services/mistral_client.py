"""Mistral chat client. Same ChatClient shape as OllamaClient.

The runner loads the environment; this class only reads MISTRAL_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any

from mistralai.client import Mistral


class MissingMistralApiKeyError(RuntimeError):
    """Raised when MISTRAL_API_KEY is not set."""


DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RESPONSE_TOKENS = 64


class MistralClient:
    """Mistral chat client.

    Reads MISTRAL_API_KEY if no api_key is given. max_response_tokens
    defaults to 64, matching OllamaClient's short-reply cap.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-small-latest",
        max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        resolved_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not resolved_key:
            raise MissingMistralApiKeyError(
                "MISTRAL_API_KEY is not set. Add it to .env or export it before running."
            )
        self._client = Mistral(
            api_key=resolved_key,
            timeout_ms=int(request_timeout * 1000),
        )
        self.model = model
        self.max_response_tokens = max_response_tokens

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Send one system+user turn and return the assistant text.

        max_tokens overrides max_response_tokens for this call. Think
        leaves it at the 64-token cap; the judge and extractor raise it.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        effective_max_tokens = self.max_response_tokens if max_tokens is None else max_tokens
        return self._complete(messages, temperature, effective_max_tokens)

    def _complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response: Any = self._client.chat.complete(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)
