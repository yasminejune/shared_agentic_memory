# §4.2 Design choices — review against the new §4.1 structure

Scope: whether the red thread survives the split into 4.1 Tech stack / 4.2 Design choices, whether each argument now belongs where it sits, and edits that are clearly needed. Verified against the codebase and the label list in `New_structure.tex`.

The section holds up well. Most of it was already written to the right boundary — implementation choices rather than tool identity — so the restructure costs you less than it might have. Six things need changing, two of them in §4.1 rather than here.

## Standing exclusion, carried forward

`THINK_MAX_TOKENS = 256` is wrong in the code and will be changed to the client default of 64. **The prose should therefore say 64, and this is excluded from code-correction for the rest of the work** — I will not flag the mismatch again. Edit 6 below.

---

## Edit 1 — Add a bridge from §4.1. **Needed.**

There is currently no sentence connecting the two sections, and "the previous sections" now ambiguously spans both Chapter 4's methodology and §4.1. The red thread between 4.1 and 4.2 has to be written; it does not survive on adjacency.

**Before**

> This section lays out how the system architecture and algorithm from the previous sections were implemented to assess this thesis' research question.

**After**

> This section lays out how the system architecture and algorithm from the previous sections were implemented to assess this thesis' research question. While Section \ref{sec:tech-stack} sets out which tools and models the thesis uses, this section sets out how they were configured.

**Also required:** `\section{Tech stack}` in §4.1 currently has no label. Add `\label{sec:tech-stack}`.

## Edit 2 — The shared-memory roadmap now over-promises. **Needed.**

This is the clearest red-thread break the restructure created. The sentence announces five implementation questions, but the first of them — model choice — is now answered in §4.1 and never returned to here. A reader who takes the roadmap literally waits for an answer that does not come.

**Before**

> The implementation questions around the shared memory creation regarded the model use for both the token sampling, the privacy hyperparameter choice, the privacy budget, parameters in the four steps of the shared memory creation, and the prompt choice.

**After**

> The implementation questions around the shared memory creation regarded the privacy hyperparameter choice, the privacy budget, the parameters in the four steps of the shared memory creation, and the prompt choice.

## Edit 3 — §4.1 says three LLM uses; §4.2 contains five. **Needed, in §4.1.**

§4.1 states "LLMs were used in the thesis in three parts" and enumerates the Think node, the DP mechanism, and title/description generation. But §4.2 then introduces two more LLM calls — the success-or-failure judge and the memory extractor — neither of which appears in that enumeration. The two sections contradict each other on a countable fact.

The fix belongs in §4.1, since §4.2's treatment of the judge and extractor is correctly placed. Add to the end of §4.1's Think-node item:

> The same model and endpoint serve every other non-private call in the system: the success-or-failure judge and the memory extractor described in Section \ref{sec:private memory creation experiment}, and the title and description generation below. Only the differentially private path departs from it, for the reason given next.

That also earns the "once more" in the third item, which currently refers back to a Think node two items away.

## Edit 4 — Two broken cross-references. **Needed.**

- **`\ref{sec:libraries}`** (in Private memory creation, on the sentence about embedding each task) — **no such label exists anywhere in the document.** It should point at the new Embedding model subsection in §4.1, which also has no label yet. Add `\label{sec:embedding-model}` to that subsubsection and change the ref.
- **`\ref{ch:results}`** (in the judge subsection) — the actual label on the Results chapter is `sec:results`, not `ch:results`. Either fix the ref or rename the label. Renaming the label to `ch:results` is tidier, since it is a chapter, but check nothing else points at `sec:results` first.

While you are there: that same sentence says the embedding produces "a 384-dimensional dense vector", which §4.1 now also states. Keep it — here it describes what goes into the stored record, which is a different point — but the cross-reference has to resolve.

## Edit 5 — Heading level inconsistency. **Needed.**

The four steps are marked `\subparagraph{Step 1 - ...}`, `\subparagraph{Step 2 - ...}`, `\subparagraph{Step 4 - ...}` — but Step 3 is `\subsubsection{Step 3 - Number of memories generated per label.}`. A level jump mid-sequence. Change it to `\subparagraph{}` to match its siblings.

## Edit 6 — Think-node token cap. **Needed.**

Per the standing exclusion above.

**Before:** "with the model's reasoning trace disabled and the reply capped at 256 tokens"

**After:** "with the model's reasoning trace disabled and the reply capped at 64 tokens"

The justification that follows already works, and you can strengthen it in one clause: 64 tokens is the length of a single action line, which is what the grammar asks for.

## Edit 7 — The extractor is stochastic, and the text does not say so. **Recommended.**

§4.2 argues carefully that the Think step is deterministic at temperature 0 "so any change in success rate between evaluation runs is attributable to the memories rather than to sampling noise." That argument is sound for the agent. But `decisions.md:59` records that the **judge runs at temperature 0.0 while the extractor runs at temperature 1.0**, reproduced from ReasoningBank Appendix A.2.

So the memories themselves are drawn stochastically, which the section never states. Given how much weight the determinism argument carries, leaving this out is a gap a reader can find. One sentence in Memory creation:

> The extractor call runs at temperature 1.0, reproduced from \citet{ouyang2025reasoningbank}, rather than at the temperature 0 used everywhere else. The memories are therefore drawn stochastically, so a second extraction pass over the same trajectories would not reproduce them exactly. Holding the setting fixed to the published value keeps the extraction comparable to ReasoningBank, at the cost of making the memory set itself a sampled artefact rather than a deterministic function of pass-through A.

## Edit 8 — Duplication with §4.1 on the action space. **Trim §4.1, not here.**

§4.1 currently lists what the WebArena subset adds — hovering, dropdown selection, multi-tab handling, history navigation, the answer and infeasibility channels — as part of justifying BrowserGym over the hand-rolled Playwright path. §4.2 then gives the same deltas more precisely, against the published table.

§4.2's version is the better one and belongs here: it is about what is exposed to the LLM, which is a design choice. Shorten §4.1's list to the general claim ("a substantially larger action vocabulary than the seven actions the hand-rolled implementation supported, set out in Section \ref{sec:experiment_design}") and let §4.2 carry the detail.

## Left alone deliberately

- **The three observation spaces are named in both sections.** §4.1 says what BrowserGym returns and forward-references here; §4.2 restates the three as the premise of your choice. A reader needs them locally to follow the argument, so this repetition earns its place. (Fix the typo "ALthough" while you are in there.)
- **"Step 1 - Prompt"** is a one-line pointer to a table collected at the end under "System prompts", which is mildly redundant. Your structure, and harmless — left as written.
- Configuration ordering A–D, all `itemize`/`enumerate` environments, the tables and verbatim prompts, and your signposting are untouched.

## Still open

- The `% TODO(Yasmine)` on `--k 3` versus the runner's CLI default of 1, in configuration B.
- The `% TODO(Yasmine)` on how many shared memories the cycle produced, in configuration C.
- The reasoning-trace justification — see the separate discussion; no edit made pending your decision.
