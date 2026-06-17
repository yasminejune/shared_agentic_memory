"""WP2 round-1 label parsers shared between the demo harness and the WP2 orchestrator.

The DP output of round 1 (Amin Algorithm 1 with
:func:`agent_memories.agent.privacy.prompts.wrap_label`) is a free
text string whose shape depends on the prompt template currently
held by :data:`~agent_memories.agent.privacy.prompts.LABEL_PROMPT`.
Two parser variants live here so the call sites can pick the one
that matches the prompt template they used; both produce the same
``list[(label_string, parsed_flag)]`` shape so callers can wrap
either one in the same per-slot ``label_<i>`` fallback policy from
WP2-plan §3.2.

* :func:`parse_numbered_labels` matches the historical
  ``Topics:\\n1.`` numbered-list template (the WP2-plan §3.2
  rendering as drafted, retained as the
  ``numbered_wp2plan_3.2`` variant of
  ``scripts/amin_et_al/compare_label_prompts.py``). It is the
  parser ``scripts/amin_et_al/WP2_8.py`` ships with -- this module
  hosts the canonical copy so the demo harness and any other
  caller using the numbered template import the same function.

* :func:`parse_json_labels` matches the current production
  :data:`~agent_memories.agent.privacy.prompts.LABEL_PROMPT`, a
  JSON array of strings (promoted from
  ``compare_label_prompts.parse_json``). The WP2 end-to-end
  orchestrator imports this one because the production prompt is
  the JSON template.

Both parsers pad missing slots with the literal ``label_<i>`` per
WP2-plan §3.2 so the round-2 pipeline always receives exactly ``k``
entries and never silently loses the round-1 privacy spend on
parse failure.
"""

from __future__ import annotations

import json
import re

_NUMBERED_LINE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
_QUOTED_STRING = re.compile(r'"([^"\n]+)"')


def parse_numbered_labels(raw: str, k: int) -> list[tuple[str, bool]]:
    """Parse the numbered-list round-1 output (the WP2-plan §3.2 draft template).

    The numbered template ends literally on ``Topics:\\n1.``, so the
    first sampled token is the continuation of label 1. The parser
    prepends ``"1."`` back to ``raw`` to recover the full numbered
    list, splits on newlines, matches each line against
    ``^\\s*(\\d+)\\.\\s*(.+?)\\s*$``, and pads missing indices in
    the expected ``1..k`` range with ``label_<i>``.

    The boolean second element of each pair is ``True`` for
    LLM-emitted labels and ``False`` for fallback slots so the
    JSONL writer in ``scripts/amin_et_al/WP2_8.py`` can flag them.
    """
    full_text = "1." + raw
    found: dict[int, str] = {}
    for line in full_text.splitlines():
        match = _NUMBERED_LINE.match(line)
        if match is None:
            continue
        idx = int(match.group(1))
        text = match.group(2).strip()
        if 1 <= idx <= k and idx not in found and text:
            found[idx] = text
    return [(found[i], True) if i in found else (f"label_{i}", False) for i in range(1, k + 1)]


def parse_json_labels(raw: str, k: int) -> list[tuple[str, bool]]:
    """Parse the current-production JSON-array round-1 output.

    Matches the
    :data:`~agent_memories.agent.privacy.prompts.LABEL_PROMPT`
    (``"Return only a JSON array of strings"``) used by the WP2
    orchestrator's round 1. Three fallback layers cope with the
    fact that DP sampling can cut the output short before the
    closing bracket or insert spurious tokens:

    1. Strict :func:`json.loads` on the stripped raw output.
    2. Same with a trailing ``"]"`` appended (the DP output may
       have been cut off at ``r`` tokens before the closing
       bracket).
    3. Regex-extract every double-quoted substring as a label
       candidate.

    The first layer that returns at least one non-empty string
    wins; the result is truncated to ``k`` candidates and any
    missing slots are padded with ``label_<i>`` (per WP2-plan
    §3.2 fallback policy, same shape as
    :func:`parse_numbered_labels`).
    """
    raw_stripped = raw.strip()
    candidates: list[str] = []
    for repair in ("", "]"):
        try:
            parsed = json.loads(raw_stripped + repair)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            candidates = [x.strip() for x in parsed if isinstance(x, str) and x.strip()]
            if candidates:
                break
    if not candidates:
        candidates = [m.group(1).strip() for m in _QUOTED_STRING.finditer(raw_stripped)]
        candidates = [c for c in candidates if c]
    candidates = candidates[:k]
    return [
        (candidates[i], True) if i < len(candidates) else (f"label_{i + 1}", False)
        for i in range(k)
    ]
