"""Round-2 generation prompt for the Amin et al. WP2.3 sampling mechanism.

Implements the WP2-plan §3.3 content-only template (2026-06-02
revision, plan §11 deviation 9). The privacy-spending DP path emits
only the ``content`` field of a single ReasoningBank ``MemoryItem``
as a short paragraph of plain prose; the matching ``title`` and
``description`` are produced downstream by the §3.5 Qwen
post-processing call on the DP-released content (free under the
post-processing property of differential privacy). The previous
markdown-scaffolded template (``# [User] / Memory Item: / # [Assistant]``
with a ``## Title`` anchor) was retired because (a) the schema
tokens consumed a non-trivial fraction of the per-output ``r``-token
budget without contributing to the synthesised insight, (b) schema
tokens are the most fragile decoding location under Amin's stochastic
exponential-mechanism sampling, and (c) the ``## Title`` anchor was
incompatible with chat-template instruction-tuned models.

Both the per-user batch prompts and the SVT public prompt are
produced by formatting :data:`GENERIC_PROMPT` via :func:`wrap`:

* ``wrap(items=..., label=...)`` for each batch member, where ``items``
  is that user's memory items rendered as one item per line; on
  production WP1.6 data each line is ``TITLE | DESCRIPTION | CONTENT``
  per the original schema, on toy harnesses it is the raw text since
  the fixtures have no title or description fields.
* ``wrap(label=...)`` (no items) for the public prompt; the items
  block defaults to the literal ``"(no examples)"``, matching
  WP2-plan §3.4 ("public prompt is the same template with the content
  block emptied").

The label string is the DP-released output of round 1 (WP2-plan
§4.3); by the post-processing property of differential privacy it is
public input to round 2 and incurs no additional privacy cost. It
appears in both the batch prompts and the public prompt in the same
position so SVT format-token alignment is preserved.

The template ends literally on ``Lesson:`` so the first sampled token
continues the lesson paragraph directly. The ``Lesson:`` anchor is a
single-line marker (two Gemma tokens) that gives the model a
deterministic continuation point without committing to a markdown
schema. The DP output is consumed as-is (after stripping leading and
trailing whitespace) and becomes the ``content`` field of one shared
``MemoryItem``; no markdown parser is needed on the privacy-spending
output. Because the template is identical between the batch prompts
and the public prompt apart from the items block, SVT format-token
alignment is preserved across both branches and the privacy budget is
spent on the lesson text itself.
"""

from __future__ import annotations

GENERIC_PROMPT = (
    "You will be given memory items distilled from one web-navigation "
    "trajectory related to {label}. Produce a single short paragraph "
    "(1 to 3 sentences) that captures the most generalisable lesson "
    "from these memory items. Output only the lesson text — no title, "
    "no header, no markdown formatting, no list, no explanation of "
    "what you are doing.\n"
    "\n"
    "Memory items:\n"
    "{items}\n"
    "\n"
    "Lesson:"
)


def wrap(items: str = "(no examples)", *, label: str) -> str:
    """Apply :data:`GENERIC_PROMPT` to ``items`` and ``label``.

    The default empty-items body (``"(no examples)"``) yields the
    public prompt used by Amin Algorithm 1's SVT branch; passing a
    rendered items block yields the matching private prompt for that
    batch member. ``label`` is the round-1 DP-released label string;
    it appears in the same position in both branches and incurs no
    additional privacy cost by the post-processing property.
    """
    return GENERIC_PROMPT.format(label=label, items=items)
