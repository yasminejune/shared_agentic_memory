"""Shared types used across the agent loop and the services layer.

Putting the ``ChatClient`` Protocol here keeps the agent and services
packages from importing from each other just to define the same shape
twice. Both layers depend on this neutral module instead.
"""

from __future__ import annotations

from typing import Protocol


class ChatClient(Protocol):
    """Structural type for any object that can complete a chat turn.

    ``MistralClient`` and ``OllamaClient`` satisfy this, and so does
    any test double exposing a matching ``chat`` method. The optional
    ``max_tokens`` override is used by the WP1.6 ReasoningBank
    pipeline (judge ~256, extractor ~768) to bypass the 64-token cap
    that the Think node relies on.
    """

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
    ) -> str: ...
