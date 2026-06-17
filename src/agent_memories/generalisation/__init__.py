"""WP2 cross-user generalisation pipeline (post-processing arm).

This subpackage will house the full WP2 pipeline described in
``.claude/thesis/WP2-plan.md``. The first piece landed is §3.5
post-processing: a standard Qwen chat call that turns a DP-released
``content`` paragraph (from the round-2 Amin mechanism in §3.3) and a
DP-released round-1 label into the matching ``title`` and
``description`` of a shared :class:`MemoryItem`.

The post-processing call is off the differential-privacy path: by the
post-processing property of differential privacy (Dwork and Roth 2014,
Proposition 2.1) any function computed on DP-released artifacts is
itself DP-released at the same level, so this call contributes zero
to the privacy account (see WP2-plan §4.7).
"""

from .assignment import LabelAssignment, assign_memories_to_labels, group_by_label
from .buffer import GatingResult, read_buffer, select_round2_inputs, write_buffer
from .labelling import parse_json_labels, parse_numbered_labels
from .post_processing import save_intermediate, title_and_description

__all__ = [
    "GatingResult",
    "LabelAssignment",
    "assign_memories_to_labels",
    "group_by_label",
    "parse_json_labels",
    "parse_numbered_labels",
    "read_buffer",
    "save_intermediate",
    "select_round2_inputs",
    "title_and_description",
    "write_buffer",
]
