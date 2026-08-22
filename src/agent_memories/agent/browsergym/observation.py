"""Flatten BrowserGym axtree objects into tree_yaml for Think."""

from __future__ import annotations

from typing import Any

from browsergym.utils.obs import flatten_axtree_to_str


def preprocess_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy obs and set tree_yaml from the accessibility tree (empty if missing)."""
    processed = dict(obs)
    axtree_object = processed.get("axtree_object")
    if axtree_object is not None:
        processed["tree_yaml"] = flatten_axtree_to_str(axtree_object)
    else:
        processed["tree_yaml"] = ""
    return processed
