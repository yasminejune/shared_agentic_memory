"""WP1.5 memory-creation pipeline.

Runs *after* the LangGraph loop completes (called by the WP1.5 runner
on the final :class:`AgentState`, not from inside the graph). The
pipeline:

1. Serialises the run as ``aim`` + the same history shape the Think
   node already uses, so the chat LLM sees a familiar trace.
2. Pulls the top-k most similar existing memories from the store and
   shows them to the LLM as *phrasing context* ("don't restate these")
   rather than as a write-time dedup gate. Small local models proved
   unreliable when asked to abstain via "None"; structural dedup is
   deferred to WP1.6's ReasoningBank distillation.
3. Asks the LLM to write one concise memory sentence and appends it
   to the store. Empty replies and stray ``"None"``-shaped replies
   are filtered defensively in case the model refuses despite the
   prompt, but the prompt no longer offers refusal as an option.

WP1.6 (ReasoningBank) will replace the body of
:meth:`MemoryPipeline.create_from_run` with success/failure
distillation prompts and structured ``{title, description, content}``
output; the public surface (one method taking an ``AgentState`` and
returning ``Memory | None``) stays the same.
"""

from __future__ import annotations

from typing import Any

from agent_memories.agent.state import AgentState
from agent_memories.types import ChatClient

from .store import Memory, MemoryStore

DEFAULT_K_FOR_DEDUP = 5

CREATION_SYSTEM_PROMPT = (
    "You are a memory curator for a web-browsing agent. "
    "Your job is to distil exactly one useful lesson per run -- a "
    "success worth imitating or a failure worth avoiding -- that future "
    "agent runs can consult.\n"
    "\n"
    "Always reply with ONE concise memory sentence stating the lesson. "
    "No markdown, no preamble, no explanations, no refusals."
)

CREATION_USER_PROMPT_TEMPLATE = (
    "Read the following agent run and write the single most useful "
    "lesson for future runs as one concise sentence.\n"
    "\n"
    "A failure trace -- repeated parse failures, infinite click loops, "
    "stuck-detector aborts, navigation mistakes -- is itself the lesson: "
    "state what went wrong and what the agent should do differently next "
    "time. A successful trace contains a transferable pattern worth "
    "memorising.\n"
    "\n"
    "Run:\n"
    "{run_summary}\n"
    "\n"
    "Existing memories (top-{k} most similar; phrase your new sentence "
    "to add value, do not simply restate them):\n"
    "{existing_block}"
)


class MemoryPipeline:
    """Single-shot, post-run memory creator backed by a chat LLM and a store."""

    def __init__(
        self,
        client: ChatClient,
        store: MemoryStore,
        *,
        k_for_dedup: int = DEFAULT_K_FOR_DEDUP,
    ) -> None:
        self.client = client
        self.store = store
        self.k_for_dedup = k_for_dedup

    def create_from_run(self, state: AgentState) -> Memory | None:
        """Write one new memory based on the run captured in ``state``.

        Returns the new :class:`Memory` on success. Returns ``None``
        only as a defensive fallback when the LLM emits an empty reply
        or a stray ``"None"``-shaped reply despite the prompt asking
        for a sentence; this should be rare and is logged.
        """
        run_summary = _serialise_run(state)
        print(f"Run summary: {run_summary}")
        top_memories = self.store.search(run_summary, k=self.k_for_dedup)
        user_prompt = CREATION_USER_PROMPT_TEMPLATE.format(
            run_summary=run_summary,
            k=self.k_for_dedup,
            existing_block=_format_existing_memories(top_memories),
        )
        reply = self.client.chat(CREATION_SYSTEM_PROMPT, user_prompt, temperature=0.3).strip()
        print(f"Curator reply: {reply!r}", flush=True)
        if not reply or _is_none_reply(reply):
            return None
        return self.store.add(reply)


def _is_none_reply(reply: str) -> bool:
    """Treat any case-insensitive ``"None"`` (possibly quoted) as a no-op."""
    cleaned = reply.strip().strip("'\"").strip()
    return cleaned.lower() == "none"


def _serialise_run(state: AgentState) -> str:
    """Render an :class:`AgentState` into a chat-friendly run summary.

    Mirrors the formatting that ``nodes._format_history`` uses inside
    the Think prompt so the curator LLM and the agent LLM are reading
    the same dialect.
    """
    history = state.get("history", [])
    history_lines = _format_history(history)
    return "\n".join(
        [
            f"Aim: {state.get('aim', '')}",
            f"Final URL: {state.get('url', '') or '(unknown)'}",
            f"Steps taken: {state.get('step', 0)}",
            f"Terminated cleanly: {bool(state.get('done', False))}",
            "",
            "Trajectory (most recent last):",
            history_lines,
        ]
    )


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "  (no recorded steps)"
    lines = []
    for record in history:
        step = record.get("step", "?")
        thought = record.get("thought", "")
        outcome = record.get("outcome", "")
        lines.append(f"  step {step}: {thought!r} -> {outcome}")
    return "\n".join(lines)


def _format_existing_memories(memories: list[Memory]) -> str:
    if not memories:
        return "  (the memory bank is empty)"
    return "\n".join(f"  {i + 1}. {m.text}" for i, m in enumerate(memories))
