"""WP2 checkpoint defaults and the x-per-label gate."""

from __future__ import annotations

import pytest
from scripts.memories.WP2_pipeline import _load_checkpoint

from agent_memories.generalisation import select_round2_inputs

pytestmark = pytest.mark.unit


def test_fresh_checkpoint_uses_entry_level_schema(tmp_path) -> None:
    checkpoint = _load_checkpoint(tmp_path / ".shared_state.json")

    assert checkpoint["schema_version"] == 3
    assert checkpoint["step1_processed_entry_ids"] == []
    assert checkpoint["consumed_entry_ids"] == []
    assert checkpoint["cumulative_delta"] == 0.0


def test_gating_passes_every_entry_from_a_qualifying_label() -> None:
    result = select_round2_inputs(
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
        x_per_label=5,
    )

    assert result.label_inputs == [[0, 1, 2, 3, 4, 5, 6], None]
    assert result.carry_over == [7, 8, 9, 10]
