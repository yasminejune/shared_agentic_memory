"""Doubles shared across the test suite.

Both are queue- or mapping-driven so a test declares what it expects back
and then asserts on what the code under test asked for.
"""

from __future__ import annotations


class FakeChatClient:
    """Returns the queued replies in order and records every call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self._replies.pop(0)


class FakeEmbedder:
    """Maps known text to fixed vectors, and anything else to zeros.

    Dimension comes from the first mapped vector, so a test only has to
    pin the texts whose similarity it actually asserts on.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None, *, dim: int = 3) -> None:
        self._mapping = {k: list(v) for k, v in (mapping or {}).items()}
        first = next(iter(self._mapping.values()), None)
        self._dim = len(first) if first is not None else dim

    def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0] * self._dim))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
