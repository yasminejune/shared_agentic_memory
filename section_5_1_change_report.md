# §5.1 Implementation — change report (v2)

Verification basis: the repository at `src/`, `scripts/`, `.claude/thesis/`; your Overleaf `bibliography.bib`; and the WebArena, BrowserGym, Gemma 2, Sentence-BERT, MTEB, Go-Browse, WebAgent-R1 and VisualAgentBench papers. LaTeX verified mechanically: environments balanced, braces balanced, no duplicate labels, every `\cite` key resolves against `bibliography.bib` plus the new entries file.

## Corrections to my own v1 report

I was wrong on three counts, because I only had the two `.bib` files inside the repo and not your Overleaf one.

- **`ouyang2022training` does exist** in `bibliography.bib`. It was never broken.
- **Both BrowserGym keys resolve.** `chezelles2025browsergym` (TMLR) and `dechezelles2024browsergym` (arXiv) are both present. Nothing breaks. They are two entries for the same paper, so consolidating is tidy rather than urgent — the TMLR entry is the better one.
- **`\addbibresource{bibliography.bib}`** is fine, since the file lives in Overleaf.

## Still blocking — six duplicate keys will error

`bibliography.bib` defines these keys **twice each**. BibLaTeX will throw on every one:

`brown2020language` · `langchain2025langmem` · `ouyang2025reasoningbank` · `shinn2023reflexion` · `yao2023tot` · `zhou2024webarena`

In each case one is an `@article`/`@misc` preprint entry and the other an `@inproceedings` published entry. Keep the published version, delete the preprint. `zhou2024webarena` and `ouyang2025reasoningbank` matter most — both are cited heavily in §5.1.

## New bibliography entries

In `section_5_1_new_bib_entries.bib`. I checked all of them against your file: **none already exist.** One note — you asked for a `.tex` file, but these are BibTeX entries and pasting them into a `.tex` would break the build, so they are in a `.bib`. Append to `bibliography.bib`.

Twelve entries: `harris2020array`, `paszke2019pytorch`, `reimers2019sentencebert`, `wang2020minilm`, `muennighoff2023mteb`, `yang2024qwen25`, `yang2025qwen3`, `gemmateam2024gemma2`, `wei2022finetuned`, `lu2026distillation`, `gandhi2026gobrowse`, `zhang2025symbiotic`, plus an optional `wei2025webagentr1`.

One is marked `[VERIFY]`: `zhang2025symbiotic`, where I could not confirm the author list past the first author.

## Your question 1 — is there a benchmark showing Qwen beats Gemma?

**No, and there cannot be a clean one.** I looked at the Open LLM Leaderboard, LMArena, MMLU-Pro, IFEval and BFCL.

- The only independent harness that ever scored `gemma-2-2b-it` on IFEval was **Open LLM Leaderboard v2, retired in March 2025** — a year before Qwen3.5-4B existed. The two models were never run through the same harness and now cannot be.
- Every Qwen3.5-4B number is vendor-reported. Its model card does not include Gemma in any comparison table.
- The Gemma 2 technical report never reports IFEval or MMLU-Pro for IT models at all.
- The closest pair: Qwen3.5-4B **IFEval 89.8** (vendor) against gemma-2-2b-it **IFEval 56.68** (archived OLLM v2, different harness). The gap is probably too large for harness differences to reverse, but it is not a controlled comparison and I would not put it in a thesis as one.
- Your deployed artefact is the **NVFP4 quantised build**; every published score is for bf16.

The comparison is confounded on three axes at once: 4B vs 2.6B parameters, March 2026 vs July 2024, and multimodal 262k-context vs text-only 8k-context. A 2026 4B model beating a 2024 2B model on instruction following is the expected result of scale plus twenty months of progress, and it tells a reader nothing.

**So I rewrote the justification to be role-driven rather than capability-driven**, which is the stronger argument anyway: Gemma is there because the DP mechanism needs token-level logit access, not because of a judgement about quality; Qwen is there because it is already resident behind the same `ChatClient` interface and the step is short-form instruction following. The text now says explicitly that no head-to-head is reported and why. If you do want an evidence-backed claim, the cheap route is your own eval: ~50 paragraphs, both models, identical prompts, score format-compliance. A few hours, directly on-task, and worth more to an examiner than incommensurable leaderboard numbers.

## Your question 2 — does the WebArena-Lite evidence transfer?

**No. Drop it — and there is a much better number.**

**WebArena-Lite** comes from VisualAgentBench: **165 tasks**, not 812. Cross-website tasks — the hardest category — were removed, and 30 of the 165 had broken or over-strict graders repaired. So it is **easier and noisier** than full WebArena, not equivalent. The 6.1% is also the better of two prompting variants; the same model scores 3.2% without the thinking-format prompt. And Qwen2.5-3B is two generations behind Qwen3.5-4B.

**The finding that matters:** Lù and Reddy (arXiv:2604.07776, April 2026) report **Qwen3.5-4B at 24.1% on the full 812 tasks under the BrowserGym protocol, with no fine-tuning.** That is your exact model on your exact benchmark through a closely related harness. It is the most directly relevant published number that exists for your setup, and not citing it would be a real omission.

I have rewritten the paragraph around the honest framing: published rates for ≤8B open models on the full 812 span roughly 2% to 24%, and the spread is driven as much by harness, observation format and prompting as by the model. Supporting numbers, all full-812 and independent of the model vendors: Qwen2.5-7B-Instruct 8.3% (Go-Browse, ICLR 2026), Llama-3.1-8B 5.6% and Llama-3.2-1B 2.4% (AgentSymbiotic), Llama-3.1-70B 18.4% (BrowserGym paper).

**This cuts both ways, so be ready for it.** 24.1% is now a visible target. If your system lands near it you are on par with published work. If it lands far below, an examiner will ask why, and the answer will be about your harness — which is a conversation better had on your own terms in the Discussion than under cross-examination.

## Other changes this round

- **Hardware split rewritten** per your answer: A30 cluster runs the agent and Think node across the 812 tasks; the laptop runs private and shared memory generation under MLX/MPS, using both Gemma and Qwen. The `% OPEN QUESTION` comment is gone. The split is now justified by what each part is bound by — the agent loop is throughput-bound and repeats per condition, the DP path is bound by token-by-token logit prediction.
- **The ~5s figure now carries a second limit.** You measured it on the Mac, but the evaluation Think node ran on the cluster, so it is the basis on which you chose the 4bn model and *not* the per-step cost of the reported runs. Worth stating — otherwise a reader assumes it describes the experiments.
- **Instruction-tuning benefits** (truthfulness, toxicity) stay out, per your answer.

## Carried forward from v1 — still open

- `\ref{intermediate results}` has a space in the label and I found no matching `\label`. Check the target exists.
- For §5.2, if you write up the Think node: the evaluated BrowserGym path uses `THINK_MAX_TOKENS = 256`, not the 64 that the Playwright path uses. And the action set is restricted on two axes — `subsets=["webarena"]` and `multiaction=False`, both narrower than BrowserGym's defaults.
