"""Thin Mistral chat wrapper for the WP1.3 Think node.

Keeps the rest of the codebase free of ``mistralai`` types so that the
agent loop can be unit-tested with a fake client (any object exposing
``chat(system, user)`` is enough).

Environment-key loading is the runner's job, not this client's: the
runner calls ``load_dotenv()`` once at startup so the wrapper never
has to remember. ``MISTRAL_API_KEY`` is read here because the SDK
requires a key string, but the value comes from the already-loaded
environment.

Rate-limit handling: a 429 from Mistral propagates as the underlying
``SDKError`` rather than being silently retried. Backoff was previously
done here, but it masked free-tier quota exhaustion as a "slow run"
instead of a clear failure. The agent loop should switch to
``--model qwen`` (local Ollama) on a sustained 429.

Timeouts: ``request_timeout`` is converted to milliseconds and passed
to the SDK as ``timeout_ms`` so a hung Mistral call does not freeze
the LangGraph loop indefinitely.

Missing-model / auth handling: a bad model name or invalid key
surfaces as the SDK's own ``NotFoundError`` / ``AuthenticationError``
on the first chat call. The Think node lets it propagate so the
runner's ``finally`` block closes the browser cleanly and the user
sees a real error rather than 30 iterations of identical failures.
"""

from __future__ import annotations

import os
from typing import Any

from mistralai.client import Mistral


class MissingMistralApiKeyError(RuntimeError):
    """Raised when ``MISTRAL_API_KEY`` is not available in the environment."""


DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RESPONSE_TOKENS = 64


class MistralClient:
    """Minimal Mistral chat wrapper.

    Reads ``MISTRAL_API_KEY`` from the environment if no ``api_key`` is
    given; environment loading itself happens once in the runner, not
    here. The model name is bound at construction time because the WP1
    loop sends the same kind of prompt every step.

    ``max_response_tokens`` matches :class:`OllamaClient`'s
    ``num_predict`` default of 64: the agent grammar is one short
    line, so capping the response stops a confused model from rambling.

    ``request_timeout`` (seconds) is forwarded to the SDK as
    ``timeout_ms`` so a hung chat call surfaces as an SDK error in
    bounded time rather than freezing the agent loop.
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
        """Send a single system+user turn and return the assistant text.

        ``max_tokens`` overrides the constructor's ``max_response_tokens``
        for a single call; the Think node leaves it at ``None`` (64-token
        cap), the WP1.6 pipeline raises it for the judge and extractor.

        Any ``SDKError`` (including HTTP 429 rate-limit) propagates
        immediately so the caller can switch model or surface the
        failure rather than wait through a silent backoff.

        Temperature defaults to ``0.0`` so WP1.3 development runs are
        as deterministic as the API allows; raise it later when we want
        the agent to explore.
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
