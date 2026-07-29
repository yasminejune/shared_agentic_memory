"""Factory functions that produce LangGraph nodes bound to a Playwright page.

LangGraph nodes have signature ``(state) -> state``, so we cannot pass
the browser handle as an argument. Instead, each factory closes over
the ``Page`` (or, for Think, the scripted action list or LLM client)
and returns the node function. This keeps the browser out of the
graph's state and out of module-level globals.

In-trajectory working memory is threaded through ``state['history']``.
Every node that "does something the next Think should know about" --
parse failure, ``stop``, action dispatch, dispatch exception -- appends
one record to it. The LLM-driven Think then serialises history into
its next user prompt, so the agent reasons over its full running trace
rather than a one-shot error string. Chat-client failures (timeouts,
404 for a missing model, etc.) are *not* logged to history; they
propagate so the runner's finally-block closes the browser and the
user sees a real error instead of 30 iterations of identical 404s.

Each node also emits one human-readable trace line per step
(``[Observe]: ...``, ``[Think]: ...``, ``[Act]: ...``). Those three
lines are the entire terminal output of a run; everything richer
(prompt dumps, file logs, run artefacts) is intentionally absent so
the next work package can layer on whatever inspection it needs.
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

# Cap on how much of the ARIA YAML we paste into a single Think prompt.
# Real-world pages like amazon.co.uk produce 60k+ token snapshots that
# blow past an 8B local model's context window (typically 8k-32k); the
# server then silently truncates the *front* of the prompt, throwing
# away the system grammar and aim, and the model stalls. 12,000 chars
# is roughly 3k tokens, which leaves headroom for grammar + running
# history within a 32k context.
OBSERVATION_CHAR_BUDGET = 12_000

# How many consecutive identical ``thought`` records terminate the loop.
# Both pathologies we saw in WP1.5 dev runs -- repeated parse failures
# on a multi-line LLM reply and repeated identical clicks on a page
# whose state never advances -- manifest as the *same* raw LLM string
# being appended to history N turns in a row. Aborting after 5 of
# those frees the runner to fall through to ``MemoryPipeline.create_from_run``
# so the run still produces a memory rather than spinning out
# ``max_steps`` worth of identical wasted turns.
DEFAULT_STUCK_THRESHOLD = 5

# Memory-injection instruction, verbatim from ReasoningBank (Ouyang et al.
# 2025, Appendix A.2 "Memory Retrieval and Response Generation"). This
# string is what the paper expects the agent to see whenever
# ReasoningBank surfaces memories for a task; the description field is
# intentionally omitted from the rendered block because the paper says
# items are "represented by their title and content".
MEMORY_INJECTION_INSTRUCTION = (
    "Below are some memory items that I accumulated from past interaction "
    "from the environment that may be helpful to solve the task. You can "
    "use it when you feel it's relevant. In each step, please first "
    "explicitly discuss if you want to use each memory item or not, and "
    "then take action."
)


def _trace(line: str) -> None:
    """Print a single trace line to stdout, flushed."""
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
    """Read the current page's ARIA snapshot into ``state['observation']``.

    The observation carries page metadata (``url``, ``title``) and the
    YAML accessibility tree (``tree_yaml``) produced by
    ``aria_snapshot(mode="ai")``. The YAML embeds ``[ref=eN]`` markers
    that the LLM-driven Think resolves via ``page.locator("aria-ref=eN")``.
    """

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
    """Walk a fixed list of actions; flip ``done`` when exhausted.

    Uses ``state['step']`` as the index into ``actions``. Once the
    index is past the end of the list, the Think node signals
    termination by setting ``done=True``. The LLM-driven Think
    replaces this in WP1.3 with the same ``(state) -> state`` signature
    so the graph topology stays unchanged.
    """

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
    """Return a Think node that asks ``client`` for one grammar-locked action.

    The factory closes over the chat client so the LangGraph node keeps
    the standard ``(state) -> state`` signature. On every step it:

    * assembles a user prompt from the aim, the running history, and
      the (size-capped) ARIA-tree observation;
    * short-circuits as ``stuck`` when the tail of ``history`` shows
      ``stuck_threshold`` consecutive turns with the same ``thought``
      (no LLM call wasted on that turn);
    * calls ``client.chat`` once and runs the reply through
      :func:`parse_action`;
    * routes the outcome through ``state``: a ``stop`` reply flips
      ``done`` and is logged in ``history``; a valid action populates
      ``action`` and lets Act run (Act logs the dispatch outcome); a
      parse failure clears ``action``, logs the failure to ``history``,
      and increments ``step`` so the loop cannot spin forever.

    Chat-client errors (HTTP 404 for an unpulled Ollama model, a
    timeout, an auth failure, a transport hiccup) are *not* caught
    here: they propagate to the runner, whose ``finally`` block closes
    the browser. The user then sees one real exception instead of the
    loop silently spinning for ``max_steps`` iterations.
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
    """Return True when the last ``threshold`` records share one ``thought``.

    The signature is the raw LLM reply: a parse-failure record stores it
    verbatim, and a successful Act record stores the same string under
    ``thought`` after Act has dispatched. So a thought-equality check
    catches both the "LLM keeps emitting an unparseable line" loop and
    the "LLM keeps emitting the same valid action on an unchanging page"
    loop without false-positives from dynamic-observation noise.
    """
    if threshold < 2 or len(history) < threshold:
        return False
    tail = history[-threshold:]
    first_thought = tail[0].get("thought", "")
    if not first_thought:
        return False
    return all(r.get("thought", "") == first_thought for r in tail)


def _record_stuck_abort(state: AgentState, *, threshold: int) -> AgentState:
    """Flip ``done`` and append a ``stuck`` record so the run terminates.

    Sets ``done=True`` (graph router sends us straight to END) and stamps
    the final history record with ``outcome: "stuck: ..."``. That marker
    is the signal the WP1.5 memory pipeline (and WP1.6 ReasoningBank's
    success/failure split) reads to know the run was unsuccessful.
    """
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
    """Log a Think-side parse failure to history and advance ``step``.

    The LLM produced an unparseable line; we keep its raw output in
    ``thought`` so the next Think turn can see what went wrong, mark
    ``action`` empty (graph router sends us back to Observe), and
    increment ``step`` so ``max_steps`` still bounds the loop in the
    pathological case where the model keeps repeating itself.
    """
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
    """Render retrieved ReasoningBank items as ``Title:``/``Content:`` blocks.

    Matches the paper's "represented by its title and content" rule
    (Appendix A.2). The runner (:mod:`scripts.memories.WP1_5`) is
    responsible for flattening :class:`MemoryEntry.items` into this
    list of ``{"title", "content"}`` dicts before invoking the graph.
    The ``description`` field is intentionally not shown to the Think
    LLM -- it is an audit field, not a reasoning aid.
    """
    rendered: list[str] = []
    for item in memories:
        title = item.get("title", "").strip()
        content = item.get("content", "").strip()
        rendered.append(f"Title: {title}\nContent: {content}")
    return "\n\n".join(rendered)


def _truncate_observation(tree_yaml: str, budget: int) -> str:
    """Cap ``tree_yaml`` to ``budget`` characters with a visible marker.

    Trims from the *end* of the YAML because the top of an accessibility
    tree carries the page's landmark structure (header, nav, main), which
    is what the model needs to navigate; deep descendant lists at the
    bottom of long pages are the parts that explode token counts on a
    site like amazon.co.uk. The marker line lets the model know it is
    looking at a partial view and that scrolling may be needed.
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
    """Execute ``state['action']`` on ``page`` and log the outcome.

    Act no longer decides termination (Think does, via ``done``) and no
    longer crashes the graph on dispatch errors. Any exception raised
    by ``dispatch`` is captured into the history record's ``outcome``
    so the next Think turn can read it and adjust. The step counter is
    incremented here so successful + failed dispatches both count
    against ``max_steps``.
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
    """Render an action dict back to a one-line, grammar-shaped string.

    Used by the ``[Act]:`` trace line. The dict carries enough
    information for both the WP1.3 ref-based grammar and the WP1.1/1.2
    selector-based shape, so this function handles both rather than
    assuming a particular Think factory produced the action.
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
