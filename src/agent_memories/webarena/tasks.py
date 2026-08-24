"""WebArena task metadata."""

from __future__ import annotations

import json

from agent_memories.config import REPO_ROOT


def load_intent_template_ids() -> dict[int, int]:
    path = REPO_ROOT / "evaluation" / "task_descriptions_all.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["task_id"]): int(row["intent_template_id"]) for row in raw}
