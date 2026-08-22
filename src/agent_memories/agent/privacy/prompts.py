"""Prompt templates shared by the Amin sampler and InvisibleInk.

GENERIC_PROMPT / wrap() ask for a short lesson paragraph given a label
and memory items. LABEL_PROMPT / wrap_label() ask for k topic labels as
a JSON array. The public prompt is the same template with items
defaulting to "(no examples)", so format tokens align and only content
tokens spend privacy budget.
"""

from __future__ import annotations

import json

EXAMPLE_LABELS = (
    "pricing risk",
    "customer churn",
    "supply delay",
    "budget pressure",
    "staff training",
    "data quality",
)

GENERIC_PROMPT = (
    "You will be given one or more memory items distilled from a single "
    "web-navigation trajectory related to {label}. Produce a single short "
    "paragraph (1 to 3 sentences) that captures the most generalisable "
    "lesson from these memory items. Output only the lesson text — no "
    "title, no header, no markdown formatting, no list, no explanation of "
    "what you are doing.\n"
    "\n"
    "Memory items:\n"
    "{items}\n"
    "\n"
    "Lesson:"
)

LABEL_PROMPT = (
    "Return exactly {k} labels that summarise the task query below.\n"
    "Each label must be 1–4 words and on a DISTINCT facet — no two labels\n"
    "may be synonyms, paraphrases, or describe the same aspect.\n"
    "Return only a JSON array of strings.\n"
    "No explanations, numbering, markdown, or extra text.\n"
    "\n"
    "Example:\n"
    "{example}\n"
    "\n"
    "Task query:\n"
    "{items}"
)


def wrap(items: str = "(no examples)", *, label: str) -> str:
    """Fill GENERIC_PROMPT with ``items`` and ``label``.

    Default ``items="(no examples)"`` is the public prompt. ``label``
    sits in the same slot on both branches and does not spend extra
    budget (post-processing of round-1 DP output).
    """
    return GENERIC_PROMPT.format(label=label, items=items)


def wrap_label(items: str = "(no examples)", *, k: int) -> str:
    """Fill LABEL_PROMPT with ``items`` and ``k``.

    Default ``items="(no examples)"`` is the public prompt. The worked
    example is EXAMPLE_LABELS sliced to ``k`` so the demonstration
    matches the "exactly k" instruction. No ``label`` argument: this
    round produces labels, it does not consume them.
    """
    example = json.dumps(list(EXAMPLE_LABELS[:k]))
    return LABEL_PROMPT.format(k=k, items=items, example=example)
