"""Build a ReasoningBank memory from a finished run.

Judge the trajectory a success or failure, then extract up to three
title/description/content items. Nothing is written if extraction is empty.
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
    """Judge outcome and optional entry from build_from_run.

    entry is None when the extractor produced nothing parseable, which
    is not the same as a failed judge.
    """

    judge_outcome: Outcome
    entry: MemoryEntry | None
    memory_extracted: bool


class MemoryPipeline:
    """Judge a finished run, then extract up to three memory items."""

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
        """Write one MemoryEntry from the finished run in state.

        Returns None if extraction is empty; nothing is written then.
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
        """Judge and extract without writing to disk."""
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
        """LLM-as-judge: successful or failed.

        Malformed output (no parseable Status line) is treated as failed.
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
        """Extract up to three title/description/content items.

        The success or failure prompt is chosen from the judge outcome.
        Items missing any of the three fields are dropped.
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
    """Read success or failure from the judge reply.

    Anything other than a clear success token is treated as failed.
    """
    match = _STATUS_LINE.search(reply)
    if match is None:
        return "failed"
    return "successful" if match.group(1).lower() == "success" else "failed"


def _parse_memory_items(reply: str) -> list[MemoryItem]:
    """Parse title/description/content items from the extractor reply.

    Blocks are split on Memory Item headers. A block missing any of
    the three fields is dropped. Text before the first header is ignored.
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
    """Split the extractor reply on each Memory Item header.

    Text before the first header is dropped.
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
    """Collapse a multi-line content section into single-spaced prose."""
    return " ".join(text.split()).strip()


def _serialise_trajectory(state: AgentState) -> str:
    """Render history as the compact step/thought/outcome lines Think uses."""
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
