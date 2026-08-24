"""InvisibleInk shared-memory pipeline: run Steps 1-4 in order.

Counterpart to Amin scripts/memories/WP2_pipeline.py. Each step is still
independently runnable via python -m agent_memories.generalisation.stepN_*.
"""

from __future__ import annotations

from agent_memories.generalisation.io import (
    DEFAULT_SHARED_STORE,
    MEMORIES_CSV,
    WORK_DIR,
)
from agent_memories.generalisation.step1_labels import main as step1_main
from agent_memories.generalisation.step2_assignment import main as step2_main
from agent_memories.generalisation.step3_content import main as step3_main
from agent_memories.generalisation.step4_store import main as step4_main


def main() -> None:
    print(f"[run] memories CSV: {MEMORIES_CSV}")
    print(f"[run] work dir:     {WORK_DIR}")
    print(f"[run] shared store: {DEFAULT_SHARED_STORE}")

    step1_main()
    step2_main()
    step3_main()
    step4_main()
    print("[run] Steps 1-4 complete.")


if __name__ == "__main__":
    main()
