"""LangGraph nodes bound to a Playwright page.

Factories close over the page (or the LLM client) because LangGraph nodes
only take state. Each step prints one [Observe]/[Think]/[Act] line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

from agent_memories.agent.state import AgentState
from agent_memories.types import ChatClient

from .actions import ActionParseError, dispatch, parse_action
from .observation import format_aria_snapshot

NodeFn = Callable[[AgentState], AgentState]

# 12k chars (~3k tokens). Uncapped ARIA YAML overflows a local 8B
# context window; the server then truncates the front of the prompt.
OBSERVATION_CHAR_BUDGET = 12_000

# Abort after 5 identical thoughts in a row (repeated parse failures
# or the same click on a page that never changes).
DEFAULT_STUCK_THRESHOLD = 5

# ReasoningBank (Ouyang et al. 2025, Appendix A.2). Items shown to Think
# are title and content only.
MEMORY_INJECTION_INSTRUCTION = (
    "Below are some memory items that I accumulated from past interaction "
    "from the environment that may be helpful to solve the task. You can "
    "use it when you feel it's relevant. In each step, please first "
    "explicitly discuss if you want to use each memory item or not, and "
    "then take action."
)


def _trace(line: str) -> None:
    print(line, flush=True)


THINK_SYSTEM_PROMPT = (
    "You are a web agent controlling a browser.\n"
    "Reply with EXACTLY one line, no commentary, no markdown, no code fences.\n"
    "\n"
    "Allowed forms (square brackets and the letter 'e' are MANDATORY where shown):\n"
    "  click [eN]\n"
    '  type [eN] "text"\n'
    "  enter\n"
    "  scroll up\n"
    "  scroll down\n"
    '  goto "url"\n'
    "  stop\n"
    "\n"
    "How to refer to an element: the accessibility tree marks each interactive "
    "element with a token like [ref=e29]. Copy the part after 'ref=' into your "
    "action, keeping the square brackets. So [ref=e29] becomes [e29].\n"
    "\n"
)


def make_observe(page: Page) -> NodeFn:
    """Write the current URL, title, and ARIA YAML into state['observation']."""

    def observe(state: AgentState) -> AgentState:
        observation: dict[str, Any] = {
            "url": page.url,
            "title": page.title(),
            "tree_yaml": format_aria_snapshot(page),
        }
        _trace(f'[Observe]: observing "{page.url}"')
        return {**state, "observation": observation, "url": page.url}

    return observe


def make_think_scripted(actions: list[dict[str, Any]]) -> NodeFn:
    """Walk a fixed action list. Sets done when the list is exhausted."""

    def think(state: AgentState) -> AgentState:
        idx = state["step"]
        if idx >= len(actions):
            thought = "Scripted actions exhausted; stopping."
            _trace(f"[Think]: {thought}")
            return {**state, "thought": thought, "done": True}
        action = actions[idx]
        thought = f"Scripted action {idx}: type={action.get('type')!r}"
        _trace(f"[Think]: {thought}")
        return {**state, "thought": thought, "action": dict(action)}

    return think


def make_think(
    client: ChatClient,
    *,
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
) -> NodeFn:
    """Ask client for one grammar-locked action line.

    Closes over the chat client. Consecutive identical thoughts abort
    after stuck_threshold. Chat-client errors propagate to the runner.
    """

    def think(state: AgentState) -> AgentState:
        if _is_stuck(state["history"], stuck_threshold):
            return _record_stuck_abort(state, threshold=stuck_threshold)

        user_prompt = _build_user_prompt(state)
        raw = client.chat(THINK_SYSTEM_PROMPT, user_prompt).strip()

        _trace(f"[Think]: {raw}")

        try:
            parsed = parse_action(raw)
        except ActionParseError as exc:
            return _record_think_failure(state, raw=raw, outcome=f"parse_failure: {exc}")

        if parsed["type"] == "stop":
            record = {
                "step": state["step"],
                "thought": raw,
                "action": parsed,
                "outcome": "stop",
            }
            return {
                **state,
                "thought": raw,
                "action": {},
                "done": True,
                "history": [*state["history"], record],
            }

        return {**state, "thought": raw, "action": parsed}

    return think


def _is_stuck(history: list[dict[str, Any]], threshold: int) -> bool:
    """True when the last threshold history records share one thought.

    Catches both an unparseable line repeated and the same valid action
    on an unchanging page.
    """
    if threshold < 2 or len(history) < threshold:
        return False
    tail = history[-threshold:]
    first_thought = tail[0].get("thought", "")
    if not first_thought:
        return False
    return all(r.get("thought", "") == first_thought for r in tail)


def _record_stuck_abort(state: AgentState, *, threshold: int) -> AgentState:
    """Flip done and append a stuck record so the run terminates."""
    _trace(f"[Think]: (stuck-detector aborted run after {threshold} identical replies)")
    record = {
        "step": state["step"],
        "thought": "(stuck-detector aborted run)",
        "action": {},
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
    """Append a Think parse failure to history and increment step."""
    record = {
        "step": state["step"],
        "thought": raw,
        "action": {},
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
            "Reply with exactly one grammar line.",
        ]
    )
    return "\n".join(sections)


def _format_memories(memories: list[dict[str, str]]) -> str:
    """Render retrieved items as Title:/Content: blocks (ReasoningBank)."""
    rendered: list[str] = []
    for item in memories:
        title = item.get("title", "").strip()
        content = item.get("content", "").strip()
        rendered.append(f"Title: {title}\nContent: {content}")
    return "\n\n".join(rendered)


def _truncate_observation(tree_yaml: str, budget: int) -> str:
    """Cap tree_yaml to budget characters, trimming from the end.

    The top of the ARIA tree holds landmarks the model needs to navigate.
    The marker line tells the model the view is partial.
    """
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


def make_act(page: Page) -> NodeFn:
    """Dispatch state['action'] on page and append the outcome to history.

    Dispatch exceptions are recorded, not re-raised. step increments on
    both success and failure.
    """

    def act(state: AgentState) -> AgentState:
        action = state["action"]
        try:
            dispatch(page, action)
            outcome = "ok"
        except Exception as exc:
            outcome = f"error: {type(exc).__name__}: {exc}"
        _trace(f"[Act]: {_render_action(action)} -> {outcome}")
        record = {
            "step": state["step"],
            "thought": state.get("thought", ""),
            "action": action,
            "outcome": outcome,
        }
        return {
            **state,
            "step": state["step"] + 1,
            "history": [*state["history"], record],
        }

    return act


def _render_action(action: dict[str, Any]) -> str:
    """Render an action dict as a one-line grammar-shaped string.

    Handles both ARIA-ref and selector-based actions.
    """
    kind = action.get("type", "?")
    target = action.get("ref") or action.get("selector")
    if kind == "click":
        return f"click [{target}]" if target else "click"
    if kind == "fill":
        value = action.get("value", "")
        return f'type [{target}] "{value}"'
    if kind == "press":
        return f"press {action.get('key', '?')}"
    if kind == "scroll":
        return f"scroll {action.get('direction', '?')}"
    if kind == "goto":
        return f'goto "{action.get("url", "")}"'
    if kind == "stop":
        return "stop"
    return f"{kind} {action}"
