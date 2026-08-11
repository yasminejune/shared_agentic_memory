"""Factory functions for BrowserGym-backed LangGraph nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_memories.agent.nodes import (
    DEFAULT_STUCK_THRESHOLD,
    MEMORY_INJECTION_INSTRUCTION,
    OBSERVATION_CHAR_BUDGET,
)
from agent_memories.agent.state import AgentState
from agent_memories.types import ChatClient

from .actions import ActionParseError, extract_action_line, parse_action
from .env import THINK_SYSTEM_PROMPT, WebArenaEnvWrapper

NodeFn = Callable[[AgentState], AgentState]

# Covers the native reasoning trace and the action line together when
# the WebArena runners construct OllamaClient with think=True.
THINK_MAX_TOKENS = 1024


def _trace(line: str) -> None:
    print(line, flush=True)


def make_observe(wrapper: WebArenaEnvWrapper) -> NodeFn:
    """Read the latest preprocessed observation from the env wrapper."""

    def observe(state: AgentState) -> AgentState:
        obs = wrapper.last_obs
        url = str(obs.get("url", ""))
        title = ""
        open_titles = obs.get("open_pages_titles", ())
        active_idx = obs.get("active_page_index", [0])
        if len(open_titles) and len(active_idx):
            title = str(open_titles[int(active_idx[0])])
        observation: dict[str, Any] = {
            "url": url,
            "title": title,
            "tree_yaml": obs.get("tree_yaml", ""),
        }
        _trace(f'[Observe]: observing "{url}"')
        return {**state, "observation": observation, "url": url}

    return observe


def make_think(
    client: ChatClient,
    *,
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
) -> NodeFn:
    """LLM Think node emitting one BrowserGym action string per turn."""

    def think(state: AgentState) -> AgentState:
        if _is_stuck(state["history"], stuck_threshold):
            return _record_stuck_abort(state, threshold=stuck_threshold)

        user_prompt = _build_user_prompt(state)
        raw = client.chat(THINK_SYSTEM_PROMPT, user_prompt, max_tokens=THINK_MAX_TOKENS).strip()
        _trace(f"[Think]: {raw}")

        try:
            action_str = parse_action(extract_action_line(raw))
        except ActionParseError as exc:
            return _record_think_failure(state, raw=raw, outcome=f"parse_failure: {exc}")

        return {
            **state,
            "thought": action_str,
            "action": {"type": "browsergym", "string": action_str},
        }

    return think


def make_act(wrapper: WebArenaEnvWrapper) -> NodeFn:
    """Execute a BrowserGym action via ``env.step``."""

    def act(state: AgentState) -> AgentState:
        action_obj = state.get("action", {})
        action_str = action_obj.get("string", "") if isinstance(action_obj, dict) else ""
        wrapper.step(action_str)
        outcome = "ok"
        if wrapper.last_obs.get("last_action_error"):
            outcome = f"error: {wrapper.last_obs['last_action_error']}"

        _trace(f"[Act]: {action_str} -> {outcome}")
        record = {
            "step": state["step"],
            "thought": state.get("thought", ""),
            "action": action_str,
            "outcome": outcome,
        }
        done = wrapper.last_terminated
        return {
            **state,
            "step": state["step"] + 1,
            "done": done or state.get("done", False),
            "history": [*state["history"], record],
            "action": {},
        }

    return act


def _is_stuck(history: list[dict[str, Any]], threshold: int) -> bool:
    if threshold < 2 or len(history) < threshold:
        return False
    tail = history[-threshold:]
    first_thought = tail[0].get("thought", "")
    # Empty thoughts count: repeated empty replies (e.g. a thinking budget
    # that never reaches content) must abort rather than burn max_steps.
    return all(r.get("thought", "") == first_thought for r in tail)


def _record_stuck_abort(state: AgentState, *, threshold: int) -> AgentState:
    _trace(f"[Think]: (stuck-detector aborted run after {threshold} identical replies)")
    record = {
        "step": state["step"],
        "thought": "(stuck-detector aborted run)",
        "action": "",
        "outcome": f"stuck: same thought repeated {threshold} times",
    }
    return {
        **state,
        "thought": "(stuck)",
        "action": {},
        "done": True,
        "history": [*state["history"], record],
    }


def _record_think_failure(state: AgentState, *, raw: str, outcome: str) -> AgentState:
    record = {
        "step": state["step"],
        "thought": raw,
        "action": "",
        "outcome": outcome,
    }
    return {
        **state,
        "thought": raw,
        "action": {},
        "step": state["step"] + 1,
        "history": [*state["history"], record],
    }


def _build_user_prompt(state: AgentState) -> str:
    tree_yaml = state.get("observation", {}).get("tree_yaml", "")
    capped_tree = _truncate_observation(tree_yaml, OBSERVATION_CHAR_BUDGET)
    sections: list[str] = [f"Aim: {state['aim']}"]
    memories = state.get("memories", [])
    if memories:
        sections.extend(["", MEMORY_INJECTION_INSTRUCTION, "", _format_memories(memories)])
    sections.extend(
        [
            "",
            "History (most recent last):",
            _format_history(state.get("history", [])),
            "",
            "Accessibility tree:",
            capped_tree or "(empty)",
            "",
            "Reply with exactly one BrowserGym action line.",
        ]
    )
    return "\n".join(sections)


def _format_memories(memories: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for item in memories:
        title = item.get("title", "").strip()
        content = item.get("content", "").strip()
        rendered.append(f"Title: {title}\nContent: {content}")
    return "\n\n".join(rendered)


def _truncate_observation(tree_yaml: str, budget: int) -> str:
    if budget <= 0 or len(tree_yaml) <= budget:
        return tree_yaml
    marker = "\n# ... <observation truncated to fit context window> ..."
    head = tree_yaml[: max(0, budget - len(marker))]
    return head + marker


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no prior steps)"
    lines = []
    for record in history:
        step = record.get("step", "?")
        thought = record.get("thought", "")
        outcome = record.get("outcome", "")
        lines.append(f"  step {step}: {thought!r} -> {outcome}")
    return "\n".join(lines)


def extract_bot_response(state: AgentState) -> str:
    """Return the last ``send_msg_to_user`` / ``report_infeasible`` answer if any."""
    for record in reversed(state.get("history", [])):
        action = record.get("action", "")
        if not isinstance(action, str):
            continue
        if action.startswith("send_msg_to_user("):
            return _extract_string_arg(action, "send_msg_to_user")
        if action.startswith("report_infeasible("):
            return "N/A"
    return "N/A"


def _extract_string_arg(call: str, func_name: str) -> str:
    prefix = f"{func_name}("
    if not call.startswith(prefix) or not call.endswith(")"):
        return "N/A"
    inner = call[len(prefix) : -1].strip()
    if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "\"'":
        return inner[1:-1]
    return inner or "N/A"
