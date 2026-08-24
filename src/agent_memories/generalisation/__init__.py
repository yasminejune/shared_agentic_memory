"""Shared-memory pipeline: DP labels, assignment, content, then title and description.

Step 1 synthesises InvisibleInk labels from task queries. Step 2 assigns
each private entry to one label by cosine similarity. Step 3 generates
the content field only. Step 4 writes title and description from that
released content and stores the shared record.
"""

from .assignment import LabelAssignment, assign_memories_to_labels, group_by_label
from .buffer import (
    GatingResult,
    entry_id,
    read_buffer,
    select_round2_inputs,
    write_buffer,
)
from .io import load_memory_entries_from_csv, render_item_block
from .labelling import parse_json_labels, parse_numbered_labels
from .post_processing import save_intermediate, title_and_description

__all__ = [
    "GatingResult",
    "LabelAssignment",
    "assign_memories_to_labels",
    "entry_id",
    "group_by_label",
    "load_memory_entries_from_csv",
    "parse_json_labels",
    "parse_numbered_labels",
    "read_buffer",
    "render_item_block",
    "save_intermediate",
    "select_round2_inputs",
    "title_and_description",
    "write_buffer",
]
