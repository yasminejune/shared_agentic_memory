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
    any test double exposing a matching ``chat`` method.
    """

    def chat(self, system: str, user: str, *, temperature: float = ...) -> str: ...
