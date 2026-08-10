"""Item-level records used by the WP2 privacy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_memories.memory import MemoryEntry, MemoryItem
from agent_memories.memory.store import Outcome

ITEM_BUFFER_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CycleItem:
    """One protected example in the WP2 item-level privacy mechanism."""

    user_id: str
    query: str
    outcome: Outcome
    item: MemoryItem
    item_index: int
    embedding: list[float]
    created_at: str

    @property
    def item_id(self) -> str:
        """Return the stable identifier used by checkpoints and audit records."""
        return f"{self.user_id}::{self.query}::{self.created_at}::{self.item_index}"

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return the versioned item-level carry-over representation."""
        return {
            "schema_version": ITEM_BUFFER_SCHEMA_VERSION,
            "user_id": self.user_id,
            "query": self.query,
            "outcome": self.outcome,
            "item": self.item.to_dict(),
            "item_index": self.item_index,
            "embedding": self.embedding,
            "created_at": self.created_at,
        }

    @classmethod
    def from_jsonl_dict(cls, data: dict[str, Any]) -> CycleItem:
        """Hydrate one item-level carry-over record."""
        if data.get("schema_version") != ITEM_BUFFER_SCHEMA_VERSION:
            raise ValueError(
                "Carry-over buffer uses the legacy entry-level schema. "
                "Reset the WP2 shared state before running item-level privacy."
            )
        raw_outcome = data.get("outcome", "failed")
        outcome: Outcome = (
            raw_outcome if raw_outcome in {"successful", "failed", "shared"} else "failed"
        )
        embedding = data.get("embedding")
        if embedding is None:
            raise ValueError("Item-level carry-over record has no query embedding.")
        return cls(
            user_id=str(data["user_id"]),
            query=str(data.get("query", "")),
            outcome=outcome,
            item=MemoryItem.from_dict(data.get("item", {})),
            item_index=int(data["item_index"]),
            embedding=[float(value) for value in embedding],
            created_at=str(data.get("created_at", "")),
        )


def flatten_entry(entry: MemoryEntry) -> list[CycleItem]:
    """Expand one trajectory entry into one protected record per memory item."""
    if entry.embedding is None:
        raise ValueError(
            f"Memory entry for user_id={entry.user_id!r}, query={entry.query!r} "
            "has no query embedding."
        )
    return [
        CycleItem(
            user_id=entry.user_id,
            query=entry.query,
            outcome=entry.outcome,
            item=item,
            item_index=item_index,
            embedding=list(entry.embedding),
            created_at=entry.created_at,
        )
        for item_index, item in enumerate(entry.items)
    ]
