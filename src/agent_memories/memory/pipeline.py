"""WP1.6 ReasoningBank memory-creation pipeline.

Runs *after* the LangGraph loop completes (called by the WP1.5 runner
on the final :class:`AgentState`, not from inside the graph). The
pipeline implements the closed-loop "memory construction" half of
ReasoningBank (Ouyang et al. 2025, §3.2, Appendix A.1 + A.2):

1. **LLM-as-Judge.** A binary classifier reads the task aim, the
   trajectory, the final webpage state and an (optional) bot response
   and emits ``Status: success`` or ``Status: failure``. Temperature
   is 0.0 for determinism (paper Appendix A.2).
2. **Outcome-routed distillation.** A success-only system prompt or a
   failure-only system prompt is selected; the LLM emits up to three
   ``# Memory Item i`` markdown blocks each carrying ``## Title``,
   ``## Description`` and ``## Content``. Temperature is 1.0 (paper
   Appendix A.2).
3. **Append.** Parsed items are written as a single :class:`MemoryEntry`
   keyed on the task query and tagged with the outcome.

The three system prompts and the two user-prompt templates are quoted
verbatim from the paper (Figures 8 + 9 in Appendix A.1); only
whitespace formatting has been adjusted to match Python string style.

Deviations from the paper (flagged for the methodology chapter):

* Judge default on malformed output is ``"failed"``. The paper does
  not specify a default; choosing ``failed`` keeps the agent learning
  from the unclear case via the failure-distillation prompt rather
  than silently storing strategies derived from an unverified
  trajectory.
* ``response`` is always passed as ``"N/A"``. The paper's own
  user-prompt template contains ``{response if response else "N/A"}``;
  our agent has no equivalent of a final "bot response", so the
  N/A fallback is the prescribed behaviour.
* The raw trajectory is not persisted alongside the items (the paper
  stores ``{query, trajectory, items}``). Storing the trajectory adds
  no downstream value for the WP2/WP4 evaluation and bloats the
  per-user JSONL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent_memories.agent.state import AgentState
from agent_memories.types import ChatClient

from .embedder import Embedder
from .store import MemoryEntry, MemoryItem, MemoryStore, Outcome, _utcnow_iso

JUDGE_MAX_TOKENS = 256
EXTRACTOR_MAX_TOKENS = 768

JUDGE_SYSTEM_PROMPT = (
    "You are an expert in evaluating the performance of a web navigation "
    "agent. The agent is designed to help a human user navigate a website "
    "to complete a task. Given the user's intent, the agent's action "
    "history, the final state of the webpage, and the agent's response to "
    "the user, your goal is to decide whether the agent's execution is "
    "successful or not.\n"
    "\n"
    "There are three types of tasks:\n"
    "1. Information seeking: The user wants to obtain certain information "
    "from the webpage, such as the information of a product, reviews, map "
    "info, comparison of map routes, etc. The bot's response must contain "
    "the information the user wants, or explicitly state that the "
    "information is not available. Otherwise, e.g. the bot encounters an "
    "exception and responds with the error content, the task is considered "
    "a failure. Besides, be careful about the sufficiency of the agent's "
    "actions. For example, when asked to list the top-searched items in a "
    "shop, the agent should order the items by the number of searches, "
    "and then return the top items. If the ordering action is missing, "
    "the task is likely to fail.\n"
    "2. Site navigation: The user wants to navigate to a specific page. "
    "Carefully examine the bot's action history and the final state of "
    "the webpage to determine whether the bot successfully completes the "
    "task. No need to consider the bot's response.\n"
    "3. Content modification: The user wants to modify the content of a "
    "webpage or configuration. Carefully examine the bot's action history "
    "and the final state of the webpage to determine whether the bot "
    "successfully completes the task. No need to consider the bot's "
    "response.\n"
    "\n"
    "IMPORTANT\n"
    "Format your response into two lines as shown below:\n"
    "Thoughts: <your thoughts and reasoning process>\n"
    'Status: "success" or "failure"'
)

JUDGE_USER_PROMPT_TEMPLATE = (
    "User Intent: {intent}\n"
    "\n"
    "Trajectory: {trajectory}\n"
    "\n"
    "The detailed final state of the webpage: ```md\n{final_state}\n```\n"
    "\n"
    "Bot response to the user: {response}"
)

SUCCESS_SYSTEM_PROMPT = (
    "System Instruction\n"
    "You are an expert in web navigation. You will be given a user query, "
    "the corresponding trajectory that represents how an agent "
    "successfully accomplished the task.\n"
    "\n"
    "Guidelines\n"
    "You need to extract and summarize useful insights in the format of "
    "memory items based on the agent's successful trajectory. The goal "
    "of summarized memory items is to be helpful and generalizable for "
    "future similar tasks.\n"
    "\n"
    "Important notes\n"
    "You must first think why the trajectory is successful, and then "
    "summarize the insights.\n"
    "You have learned or seen something new from the successful "
    "trajectory.\n"
    "You can extract at most 3 memory items from the trajectory.\n"
    "You must not repeat similar or overlapping items.\n"
    "Do not mention specific websites, queries, or string contents, but "
    "rather focus on the generalizable insights.\n"
    "\n"
    "Output Format\n"
    "Your output must strictly follow the Markdown format shown below:\n"
    "# Memory Item i\n"
    "## Title <the title of the memory item>\n"
    "## Description <one sentence summary of the memory item>\n"
    "## Content <1-3 sentences describing the insights learned to "
    "successfully accomplishing the task>"
)

FAILURE_SYSTEM_PROMPT = (
    "System Instruction\n"
    "You are an expert in web navigation. You will be given a user query, "
    "the corresponding trajectory that represents how an agent attempted "
    "to resolve the task but failed.\n"
    "\n"
    "Guidelines\n"
    "You need to extract and summarize useful insights in the format of "
    "memory items based on the agent's failed trajectory. The goal of "
    "summarized memory items is to be helpful and generalizable for "
    "future similar tasks.\n"
    "\n"
    "Important notes\n"
    "You must first reflect and think why the trajectory failed, and "
    "then summarize what lessons you have learned or strategies to "
    "prevent the failure in the future.\n"
    "You can extract at most 3 memory items.\n"
    "You must not repeat similar or overlapping items.\n"
    "Do not mention specific websites, queries, or string contents, but "
    "rather focus on the generalizable insights.\n"
    "\n"
    "Output Format\n"
    "Your output must strictly follow the Markdown format shown below:\n"
    "# Memory Item i\n"
    "## Title <the title of the memory item>\n"
    "## Description <one sentence summary of the memory item>\n"
    "## Content <1-3 sentences describing the insights learned to "
    "successfully accomplishing the task>"
)

EXTRACTOR_USER_PROMPT_TEMPLATE = "Query: {query}\nTrajectory: {trajectory}"


@dataclass(frozen=True)
class MemoryBuildResult:
    """In-memory output of :meth:`MemoryPipeline.build_from_run`.

    ``judge_outcome`` is always the LLM-as-Judge signal. ``entry`` is
    ``None`` when the extractor produced zero parseable items (distinct
    from a judge failure).
    """

    judge_outcome: Outcome
    entry: MemoryEntry | None
    memory_extracted: bool


class MemoryPipeline:
    """Two-stage ReasoningBank memory creator backed by a chat LLM and a store."""

    def __init__(
        self,
        client: ChatClient,
        store: MemoryStore,
    ) -> None:
        self.client = client
        self.store = store

    def create_from_run(
        self,
        state: AgentState,
        *,
        final_state: str = "",
    ) -> MemoryEntry | None:
        """Write one new :class:`MemoryEntry` based on the run captured in ``state``.

        Returns the new entry on success. Returns ``None`` when the
        extractor produces zero parseable memory items (the run is
        then dropped: ReasoningBank's "simple addition" consolidation
        treats an empty extraction the same as no new memory at all,
        rather than writing a useless entry).
        """
        result = self.build_from_run(
            state,
            final_state=final_state,
            user_id=self.store.user_id,
            embedder=self.store.embedder,
        )
        if result.entry is None:
            return None
        return self.store.add_entry(
            query=result.entry.query,
            outcome=result.entry.outcome,
            items=result.entry.items,
        )

    def build_from_run(
        self,
        state: AgentState,
        *,
        final_state: str = "",
        user_id: str,
        embedder: Embedder,
        response: str = "N/A",
    ) -> MemoryBuildResult:
        """Build judge outcome and optional :class:`MemoryEntry` without writing to disk."""
        aim = state.get("aim", "")
        trajectory = _serialise_trajectory(state)

        outcome = self._classify(
            intent=aim,
            trajectory=trajectory,
            final_state=final_state,
            response=response,
        )
        print(f"[Memory] Judge outcome: {outcome}", flush=True)

        items = self._extract(aim=aim, trajectory=trajectory, outcome=outcome)
        print(f"[Memory] Extracted {len(items)} item(s) from trajectory.", flush=True)
        if not items:
            return MemoryBuildResult(
                judge_outcome=outcome,
                entry=None,
                memory_extracted=False,
            )

        embedding = embedder.embed(aim)
        entry = MemoryEntry(
            user_id=user_id,
            query=aim,
            outcome=outcome,
            items=items,
            embedding=embedding,
            created_at=_utcnow_iso(),
        )
        return MemoryBuildResult(
            judge_outcome=outcome,
            entry=entry,
            memory_extracted=True,
        )

    def _classify(
        self,
        *,
        intent: str,
        trajectory: str,
        final_state: str,
        response: str = "N/A",
    ) -> Outcome:
        """Run the LLM-as-Judge and return the binary outcome.

        Malformed output (no parseable ``Status:`` line) defaults to
        ``"failed"`` so the agent still learns from the unclear case
        via the failure-distillation path, rather than silently
        storing strategies derived from an unverified trajectory.
        """
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            intent=intent,
            trajectory=trajectory,
            final_state=final_state or "(no final state available)",
            response=response,
        )
        reply = self.client.chat(
            JUDGE_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        return _parse_judge_status(reply)

    def _extract(self, *, aim: str, trajectory: str, outcome: Outcome) -> list[MemoryItem]:
        """Run the success- or failure-specific extractor and parse items.

        The paper allows at most three items per trajectory; we cap
        the parsed list at three to match. Items missing any of the
        three fields are dropped as malformed.
        """
        system_prompt = SUCCESS_SYSTEM_PROMPT if outcome == "successful" else FAILURE_SYSTEM_PROMPT
        user_prompt = EXTRACTOR_USER_PROMPT_TEMPLATE.format(query=aim, trajectory=trajectory)
        reply = self.client.chat(
            system_prompt,
            user_prompt,
            temperature=1.0,
            max_tokens=EXTRACTOR_MAX_TOKENS,
        )
        return _parse_memory_items(reply)[:3]


_STATUS_LINE = re.compile(r"^\s*Status\s*:\s*[\"']?(success|failure)", re.IGNORECASE | re.MULTILINE)

_ITEM_BLOCK = re.compile(r"#\s*Memory\s+Item\s*\d+\b", re.IGNORECASE)
_TITLE_LINE = re.compile(r"^##\s*Title\b[:\s]*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_DESCRIPTION_LINE = re.compile(r"^##\s*Description\b[:\s]*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONTENT_BODY = re.compile(
    r"^##\s*Content\b[:\s]*(.+?)(?=^##\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _parse_judge_status(reply: str) -> Outcome:
    """Extract the ``success`` / ``failure`` token from the judge reply.

    Anything other than a clear ``success`` token (case-insensitive,
    optionally wrapped in quotes) is treated as a failure so the
    pipeline never stores a "validated strategy" derived from an
    unverified trajectory.
    """
    match = _STATUS_LINE.search(reply)
    if match is None:
        return "failed"
    return "successful" if match.group(1).lower() == "success" else "failed"


def _parse_memory_items(reply: str) -> list[MemoryItem]:
    """Parse 0-N :class:`MemoryItem`s from the extractor's markdown reply.

    Items are delimited by ``# Memory Item N`` headers. Each block
    must contain a ``## Title``, ``## Description`` and ``## Content``
    section; blocks missing any one of those are dropped. Surrounding
    chatter before the first item header is ignored, matching the
    paper's "strictly follow the Markdown format" instruction without
    crashing on a verbose model.
    """
    if not reply.strip():
        return []
    chunks = _split_into_item_blocks(reply)
    items: list[MemoryItem] = []
    for chunk in chunks:
        title = _extract_first(_TITLE_LINE, chunk)
        description = _extract_first(_DESCRIPTION_LINE, chunk)
        content = _extract_first(_CONTENT_BODY, chunk)
        if not (title and description and content):
            continue
        items.append(
            MemoryItem(
                title=title.strip(),
                description=description.strip(),
                content=_collapse_whitespace(content),
            )
        )
    return items


def _split_into_item_blocks(reply: str) -> list[str]:
    """Split the extractor reply at each ``# Memory Item N`` header.

    Anything before the first header is dropped so verbose preambles
    do not become a phantom 0th item.
    """
    matches = list(_ITEM_BLOCK.finditer(reply))
    if not matches:
        return []
    blocks: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reply)
        blocks.append(reply[start:end])
    return blocks


def _extract_first(pattern: re.Pattern[str], chunk: str) -> str | None:
    match = pattern.search(chunk)
    if match is None:
        return None
    return match.group(1)


def _collapse_whitespace(text: str) -> str:
    """Normalise a multi-line content section into single-spaced prose."""
    return " ".join(text.split()).strip()


def _serialise_trajectory(state: AgentState) -> str:
    """Render the agent's trajectory in the same dialect Think reads.

    The judge and the extractor both want a compact view of what the
    agent did, not the full ARIA observation at every step. Mirroring
    the Think prompt's history block keeps the curator LLM and the
    agent LLM on the same dialect.
    """
    history = state.get("history", [])
    if not history:
        return "(no recorded steps)"
    lines: list[str] = []
    for record in history:
        step = record.get("step", "?")
        thought = record.get("thought", "")
        outcome = record.get("outcome", "")
        lines.append(f"  step {step}: {thought!r} -> {outcome}")
    return "\n" + "\n".join(lines)
