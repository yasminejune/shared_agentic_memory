# Chapter 6 — change report

Deliverable: `chapter_6_difficulties_revised.tex`. Drop-in replacement for lines 1659–1850 of `New_structure.tex`.

Mechanical checks passed: environments balanced, braces balanced, no duplicate labels, no label collisions with the rest of the thesis, every `\ref` resolves, every `\cite` key already exists in your bibliography, no first-person pronouns outside quoted data.

---

## 1. Blocking — decide before submission

**No `OPENAI_API_KEY` in the repo, but all 812 baseline runs completed.**
`.env` holds only `GITLAB_API_TOKEN`, `MISTRAL_API_KEY`, `GEMMA_ACCESS_TOKEN`. Yet every row of `data/webarena/trajectories_0.csv` carries `run_status='ok'`. The installed harness (`venv/.../webarena/evaluation_harness/helper_functions.py:161`) calls `gpt-4-1106-preview`, and `openai_utils.py` raises `ValueError` when the key is absent — BrowserGym's `task.py:194` catches only `AssertionError`, so a missing key should have crashed those tasks, not scored them zero. Either the key was exported into the shell on the cluster, or the 118 judge tasks never fired. This changes what your first bullet in §"A low success rate" actually claims. I left a `% TODO(Yasmine)` at that spot and marked the re-run figure `[INTERIM]`.

**Two placeholders still empty**, both marked `[INTERIM]` in bold so they cannot be missed: the success rate after adding the API key, and the success rate after the corrected prompt. Neither exists in `data/webarena/`.

**The parse-failure bug is unfixed.** §`sec:difficulty-parse` states the memory pass-through has to be repeated. If you fix and re-run before submission, that section and the Results chapter both change.

---

## 2. Corrected claims

| Your text | Correction | Evidence |
|---|---|---|
| "Gemma 2 IT 3B" | Gemma 2 **2B** IT (`google/gemma-2-2b-it`) | `scripts/amin_et_al/compare_gemma.py:49`; `.claude/memory/decisions.md` |
| "the MIT-G dataset, which was used to train a classifier to determine the genre of new datasets" | The reproduction ran on **TREC**, not MIT-G. Also MIT-G in the paper is movie-genre *extraction*, evaluated by in-context learning with GPT-3 babbage-002 — no classifier is trained. | `scripts/amin_et_al/amin_experience_copied.ipynb` cell 1 loads `thedevastator/the-trec-question-classification-dataset`, 5,452 rows, coarse labels mapped to Abbreviation/Description/Entity/Person/Number/Location. Amin Table 1a. **The figure caption `fig:amin_reproduction` was changed from MIT-G to TREC.** |
| "a privacy budget of 1, 1/s" (Amin's) | ε = 1, **δ = 1/n** where n is the sensitive dataset size, not 1/s | Amin Appendix C, "Privacy checklist" 3(c): "we report (ε, δ)-DP for ε = 1, 3, 10 and δ = (training_set_size)⁻¹". Your notebook uses `delta = 1.0 / len(train_df)`, i.e. 1/n. **See open question 3 below — `decisions.md` records the project convention as 1/s.** |
| Footnotes: "target epsilon = 200.0, delta = 0.01" | Target was 200.0; `r` hit its `r_max` cap of 80 first, so the **realised** budget was ε = 26.2, δ = 0.01. I state target and realised separately. | `label_prompt_comparison.jsonl`, `label_generation_mode_comparison.jsonl`: `realised_epsilon = 26.2178`, `r = 80` |
| "the amin et al approach, where all three labels are concatenated" / the sequential mode as "an alternative" | **Reversed.** Amin Algorithm 1 line 7 resets `x` to an empty sequence per example. Your `sequential_eos` mode is the paper-faithful one; the concatenated JSON array is your deviation. Reframed as agreed. | Amin Alg. 1 lines 6–20; `compare_label_generation_modes.py` docstring: "``sequential_eos`` — Amin et al. Algorithm 1 lines 4–20" |
| "importantly with a temperature higher than 0" (as what distinguishes the modes) | Cut. Both modes run at τ = 1.0 private / 1.5 public. Temperature is not the difference; the sequence reset is. | `compare_label_generation_modes.py:67–68` |
| "using the parameters from the paper" (on the toy fixture) | Only the reproduction run used the paper's parameters (c = 10, τ = 1.5, θ = 0.3, σ = 0.1, s = 500). The toy-fixture checks used c = 20 or 50. I attached the paper's parameters to the reproduction bullet only. | `compare_gemma.py:45` (c=20); `compare_label_prompts.py:66` (c=50); notebook cell 9 |
| "WebArena only consider a task successful if a send_message_to_user was sent" | False as a general claim. Only the 335 `string_match` tasks need an answer; the other 477 are scored on final URL or page state. Narrowed and footnoted. In your own baseline, 47 of the 59 successes never ended with `send_msg_to_user`. | `evaluation_harness/evaluators.py` `evaluator_router`; counted from `config_files/test.raw.json` |
| "WebArena uses ChatGPT as an evaluator of xx trajectories" | The judge is hard-coded to `gpt-4-1106-preview`, and it fires on **118** of 812 tasks (82 via `llm_fuzzy_match`, 36 via `llm_ua_match`). | `helper_functions.py:146–173` |
| "this thesis is using Qwen 3" | Qwen**3.5** at 4B, `qwen3.5:4b-nvfp4` | `scripts/webarena/common.py:24`; matches your §5 tech stack |
| "the xx result of 24.xx percent from xx" | `\citet{lu2026distillation}`, 24.1%, same Qwen3.5 4B, full 812 tasks | Already stated in your §5; I cross-reference rather than restate the argument |
| "ReasoningBank used Claude and Gemini" | Correct — Gemini-2.5-flash, Gemini-2.5-pro, Claude-3.7-sonnet. Named them. | Ouyang et al. Table 1 |
| Failure-ending table: 620 / 74 / 59 | **Confirmed exactly.** Reproduced from `trajectories_0.csv` by testing whether each action appears anywhere in the trajectory. Percentages of 753 failures: 82.3 / 9.8 / 7.8. | recomputed |
| 7.27% baseline | **Confirmed.** 59/812 = 7.266%. | recomputed |
| "text that was utter gibberish... foreign characters" | **Confirmed and now evidenced.** `interim_amin_memories_copied.csv` holds 26 generated examples full of CJK, Cyrillic, Arabic, Korean and emoji. | recomputed |
| "Amin et al. did not publish their code" | Correct. Appendix C item 4: "At this time, the implementation is close-sourced." | verified |
| Spelling and grammar | Fixed throughout: permuatations, teh, accurancy, algorith, refernece, figrue, appraoch, temperateure, intuiton, hgih, high-probabilyt, prefereences, Althoguh, remarkedly, limitaiton, seperate, reflectino, owuld, WIthout/THe/THis/THe sentence-initial capitals. | — |

---

## 3. Content added (all agreed with you)

**New subsection `sec:difficulty-parse`, "Memory injection breaking the action grammar".** Replaces the thinking-tokens bullet. Every number recomputed from the CSVs:

- 59 vs 57 is noise: 18 success→fail, 16 fail→success, exact McNemar p = 0.86.
- Parse failures 114/7,950 (1.4%) → 3,273/8,534 (38.4%).
- 2,358 rejected replies open with a ```` ```python ```` fence, vs 14 in baseline; 2,135 would have parsed if the fence were stripped.
- 111 tasks had a valid terminal `send_msg_to_user`/`report_infeasible` discarded, vs 4 in baseline. 7 of the 111 are baseline successes.
- Stuck aborts 362 → 525.
- Cause: `env.py:22` ("EXACTLY one line ... no commentary") contradicts the ReasoningBank injection instruction at `playwright/nodes.py:69–71` ("first explicitly discuss if you want to use each memory item"), which is appended *after* the system prompt at `browsergym/nodes.py:161`.

I added one paragraph drawing the methodological lesson — that reproducing a prompt verbatim as an experimental control was right in isolation and wrong in combination with a single-line grammar and a disabled reasoning trace. Cut it if you find it too neat.

**Head-overlap diagnostic**, added to the clipping bullet. At generation step 2 the ten toy memories' top tokens are *negative, music, positive, New, positive, positive, Positive, delicious, Unexpected, positive*; head sizes 1–12; **identical at c = 50, 20 and 10**. Two conclusions: clipping is provably not the cause, and the heads barely overlap, so averaging has little to carry. Source: `outputs/head_overlap_summary_step2.csv`, `assess_head_overlap.py`.

**Five tables built from the raw output you pasted**, plus two you asked for:
`tab:toy-fixture`, `tab:label-prompt-comparison`, `tab:label-mode-comparison`, `tab:clustering-comparison`, `tab:failure-endings`, `tab:hypotheses-revision`, `tab:scenario-revision`. All `booktabs`; the wide ones use `tabularx`, which you already load.

**The five clustering methods described**, from `cluster_comparison_extensive.jsonl` (n = 100, k = 4): mechanism privacy-stripped, mechanism as deployed (realised ε = 10.9), spherical k-means, BERTopic with k-means, BERTopic at defaults (chose k = 3 and dumped 65 of 100 into an outlier bucket). I sharpened your conclusion: the point is not that the mechanism clusters better, it is that k-means and BERTopic cannot *name* their clusters usably — k-means returns the nearest memory, c-TF-IDF returns stopwords.

**The five prompt variants**, with the instruction delta for each. The chosen variant is **`json_discriminative`** — confirmed promoted into production `LABEL_PROMPT` at `src/agent_memories/agent/privacy/prompts.py`, and it matches `tab:label-prompt` in your §5 verbatim.

---

## 4. Structural changes

- Raw `1.` / `*` markers became `\subsection` headings and `itemize` with `\textbf{}` lead-ins. Your ordering is unchanged; `sec:difficulty-parse` is inserted between the old items 3 and 4, per your choice.
- First person removed throughout ("I would get text", "my laptop") to match the rest of the thesis. Quoted review text in `tab:toy-fixture` keeps its "I".
- The broken pseudo-table (`\tableautorefnameid,text ... \end of table`) became `tab:toy-fixture`.
- Section 1's opening was split: the gibberish problem, then why the label step was chosen as the test bed, then the fixture. Previously one paragraph doing all three.
- Removed duplication: the £50-per-run and Qwen-capability arguments are already made in §5 (`sec:tech-stack`), so Chapter 6 states them once and cross-refers.

---

## 5. Left untouched

- Both figure environments (`gemma_comparison.png`, `all_token_logit_heights_with_zoom.png`) and their captions, unchanged. I only edited the `amin_reproduction` caption's dataset name, which was factually wrong.
- Your "critical reflections" opening paragraph — it was already doing what it needed to.
- The £50 / £100 credit / `t3a.xlarge` / 1 TB / 2 TB disk figures. These are your own records and I could not check them against anything in the repo.
- Your colloquial grammar ("The reason for the switch is two-fold", the uneven "Firstly... Second...").

---

## 6. Open questions for you

1. **Figure `fig:amin_reproduction`** — caption says "first two memories", your prose said "three illustrative examples". I removed the count from the prose and kept your caption. Confirm the figure shows two.

2. **§5 says the Think node has a "64-token budget"** (`New_structure.tex`, ~line 1234). The WebArena Think node uses `THINK_MAX_TOKENS = 256` (`browsergym/nodes.py:21`); 64 is only the `OllamaClient` constructor default. Outside this chapter's scope, but it undercuts the "a reasoning block would exhaust the budget" argument, which is load-bearing for your §5 justification.

3. **δ convention.** `decisions.md` records "The WP2 differential-privacy `delta` defaults to `1/s` (the expected batch size) per Amin et al. (2024) Appendix C convention". Appendix C says 1/n, the dataset size, and `check_delta` correctly validates `delta <= 1/n`. Two different conventions are recorded in your own notes. In the toy runs they coincide (s = n = 100), so nothing here is wrong numerically, but §"Differential privacy mechanism" may need a look.

4. **Scenario B in `tab:scenario-revision`.** I took the four revised scenarios from §`sec:experiment_design`, including your own note that B measures the ceiling of memory injection rather than transfer. There is still an unresolved `% TODO(Yasmine): confirm --k 3` there — the memory run did use `--k 3` (all 811 rows have three `retrieved_task_ids`), so that TODO can be closed.

5. **Should Chapter 6 name the 684-task caveat?** `\citet{ouyang2025reasoningbank}` excludes the Map domain, so its 40.5–56.3% figures are on 684 tasks, not 812, which flatters them against your 7.27%. I left it out to avoid duplicating §5, but it strengthens your point if you want it.
