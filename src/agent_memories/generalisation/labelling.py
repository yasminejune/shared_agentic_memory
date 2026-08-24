"""Parsers for Step 1 InvisibleInk label output.

parse_json_labels matches the production JSON-array prompt.
parse_numbered_labels matches the older numbered-list template.
Both return exactly k (label, parsed_flag) pairs, padding missing
slots with label_<i>.
"""

from __future__ import annotations

import json
import re

_NUMBERED_LINE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
_QUOTED_STRING = re.compile(r'"([^"\n]+)"')


def parse_numbered_labels(raw: str, k: int) -> list[tuple[str, bool]]:
    """Parse numbered-list label output into k (text, parsed) pairs.

    Prepends "1." to raw, matches numbered lines, and pads missing
    indices with label_<i>. The flag is True for model-emitted labels
    and False for fallback slots.
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
    """Parse JSON-array label output into k (text, parsed) pairs.

    Tries json.loads, then the same with a trailing "]", then quoted
    substrings. Truncates to k and pads missing slots with label_<i>.
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
