"""Shared paths, artefacts, and CSV loading for the four pipeline steps.

The memories CSV contract matches agent_memories.webarena.constants
(extracted rows only, embedding re-attached)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_memories.agent.invisible_ink.accounting import InvisibleInkAccount
from agent_memories.config import DEFAULT_MEMORY_DIR, REPO_ROOT
from agent_memories.memory import MemoryEntry

RUN = 1
K_LABELS = 116
BUCKET_SIZE = 7
EPSILON = 10.0
DELTA = 1e-5
TAU = 1.0
TOP_K = 100
STEP1_MAX_TOKENS = 1024
STEP3_MAX_TOKENS = 100
GEMMA_CHUNK_SIZE = 8
QWEN_MODEL = "qwen3.5:4b-nvfp4"

LABELS_FILENAME = "labels.json"
ASSIGNMENTS_FILENAME = "assignments.json"
CONTENTS_FILENAME = "contents.jsonl"
BUFFER_FILENAME = ".shared_buffer.jsonl"
DEFAULT_SHARED_STORE = DEFAULT_MEMORY_DIR / "shared.jsonl"

MEMORIES_CSV = REPO_ROOT / "data" / "webarena" / "trajectories_reasoningbank_private_memories.csv"
WORK_DIR = REPO_ROOT / "data" / "webarena" / f"shared_memory_run_{RUN}"


def labels_path(work_dir: Path) -> Path:
    return work_dir / LABELS_FILENAME


def assignments_path(work_dir: Path) -> Path:
    return work_dir / ASSIGNMENTS_FILENAME


def contents_path(work_dir: Path) -> Path:
    return work_dir / CONTENTS_FILENAME


def buffer_path(work_dir: Path) -> Path:
    return work_dir / BUFFER_FILENAME


def render_item_block(entry: MemoryEntry) -> str:
    """Render one trajectory entry's items for a Step 3 prompt row."""
    return "\n".join(
        f"Memory {index}: {item.title} | {item.description} | {item.content}"
        for index, item in enumerate(entry.items, start=1)
    )


def load_memory_entries_from_csv(csv_path: Path) -> list[tuple[int, MemoryEntry]]:
    """Load (task_id, MemoryEntry) pairs from the memories CSV.

    Rows without an extracted memory or
    without a stored embedding are skipped.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Memories file not found: {csv_path}")

    pairs: list[tuple[int, MemoryEntry]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task_id_raw = row.get("task_id", "").strip()
            if not task_id_raw.isdigit():
                continue
            if row.get("memory_extracted", "").strip() != "True":
                continue
            memory_data = json.loads(row.get("memory", "{}") or "{}")
            embedding = json.loads(row.get("embedding", "[]") or "[]")
            if not memory_data or not embedding:
                continue
            entry = MemoryEntry.from_jsonl_dict(memory_data)
            entry.embedding = [float(x) for x in embedding]
            pairs.append((int(task_id_raw), entry))
    return pairs


def account_to_dict(account: InvisibleInkAccount) -> dict[str, Any]:
    """JSON-serialise an InvisibleInkAccount."""
    return asdict(account)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def entry_from_payload(data: dict[str, Any]) -> MemoryEntry:
    """Hydrate a MemoryEntry from an assignments-file record."""
    return MemoryEntry.from_jsonl_dict(data)


__all__ = [
    "ASSIGNMENTS_FILENAME",
    "BUCKET_SIZE",
    "BUFFER_FILENAME",
    "CONTENTS_FILENAME",
    "DEFAULT_SHARED_STORE",
    "DELTA",
    "EPSILON",
    "GEMMA_CHUNK_SIZE",
    "K_LABELS",
    "LABELS_FILENAME",
    "MEMORIES_CSV",
    "QWEN_MODEL",
    "RUN",
    "STEP1_MAX_TOKENS",
    "STEP3_MAX_TOKENS",
    "TAU",
    "TOP_K",
    "WORK_DIR",
    "account_to_dict",
    "assignments_path",
    "buffer_path",
    "contents_path",
    "entry_from_payload",
    "labels_path",
    "load_memory_entries_from_csv",
    "read_json",
    "read_jsonl",
    "render_item_block",
    "write_json",
    "write_jsonl",
]
