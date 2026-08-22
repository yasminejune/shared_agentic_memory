"""ChatClient protocol shared by the agent and the services layer.

MistralClient and OllamaClient both match this shape, so neither
package has to import the other.
"""

from __future__ import annotations

from typing import Protocol


class ChatClient(Protocol):
    """Anything that can complete a chat turn.

    MistralClient and OllamaClient satisfy this, and so does a test
    double with a matching chat method. max_tokens is optional so the
    ReasoningBank judge and extractor can raise the 64-token Think cap.
    """

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
    ) -> str: ...
