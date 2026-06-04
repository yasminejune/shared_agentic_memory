"""Generation prompts for the Amin et al. WP2 sampling mechanism.

Two templates live here, one per WP2 round, both consumed by the
same per-token loop in :mod:`agent_memories.agent.privacy.privatisation`:

* :data:`GENERIC_PROMPT` / :func:`wrap` — round 2 (shared-memory
  synthesis, WP2-plan §3.3 content-only template, 2026-06-02 revision,
  plan §11 deviation 9). The privacy-spending DP path emits only the
  ``content`` field of a single ReasoningBank ``MemoryItem`` as a
  short paragraph of plain prose; the matching ``title`` and
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
* :data:`LABEL_PROMPT` / :func:`wrap_label` — round 1 (label
  synthesis, WP2-plan §3.2). Asks the model for exactly ``k`` short
  topic labels as a JSON array of strings, with an explicit
  anti-redundancy clause requiring each label to be on a distinct
  facet (no synonyms or paraphrases). Promoted from the
  ``json_discriminative`` variant of
  ``scripts/amin_et_al/compare_label_prompts.py`` after the
  numbered-list template (``N. <label>`` lines ending on
  ``Topics:\\n1.``) was retired: the JSON output format degrades
  more gracefully under Amin's stochastic exponential-mechanism
  sampling than the numbered-list anchor, and the distinct-facet
  clause reduces near-duplicate labels that would later collapse
  into the same round-2 batch under cosine similarity. No
  ``label`` placeholder because labels are the *output* of round 1,
  not an input. The DP-released label list becomes the public input
  that round 2 consumes via :func:`wrap`.

Both rounds share the same SVT alignment principle: the public
prompt for the SVT branch is the same template with the items block
replaced by the literal ``"(no examples)"`` (the default of
:func:`wrap` and :func:`wrap_label`), so format tokens align between
the public and private branches and only content tokens consume
privacy budget (WP2-plan §3.4).

The round-2 ``label`` argument is the DP-released output of round 1
(WP2-plan §4.3); by the post-processing property of differential
privacy it is public input to round 2 and incurs no additional
privacy cost.
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

LABEL_PROMPT = (
    "Return exactly {k} labels that summarise the memory items below.\n"
    "Each label must be 1–4 words and on a DISTINCT facet — no two labels\n"
    "may be synonyms, paraphrases, or describe the same aspect.\n"
    "Return only a JSON array of strings.\n"
    "No explanations, numbering, markdown, or extra text.\n"
    "\n"
    "Example:\n"
    '["pricing risk", "customer churn", "supply delay", '
    '"budget pressure", "staff training", "data quality"]\n'
    "\n"
    "Memory items:\n"
    "{items}"
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


def wrap_label(items: str = "(no examples)", *, k: int) -> str:
    """Apply :data:`LABEL_PROMPT` to ``items`` and ``k``.

    Round-1 sibling of :func:`wrap`. The default empty-items body
    (``"(no examples)"``) yields the SVT public prompt; passing a
    rendered items block yields the matching private prompt for one
    batch member. ``k`` is the requested number of labels and appears
    in the same position in both branches so SVT format-token
    alignment is preserved. No ``label`` argument: round 1 produces
    the labels, it does not consume them.
    """
    return LABEL_PROMPT.format(k=k, items=items)
