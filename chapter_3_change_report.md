# Chapter 3 (Project management) — change report

Deliverables:

- `chapter_3_project_management_revised.tex` — the revised chapter, ready to replace lines 491–598 of `New_structure.tex`
- `chapter_3_new_bib_entries.bib` — six new entries to append to `bibliography.bib`

## 1. Blocks submission — decide these first

| # | Issue | Action needed from you |
|---|---|---|
| B1 | `\ref{sec:interim results}` in your draft points at a label that does not exist (and contains a space, which would break regardless). | Add `\label{ch:interim-results}` to `\chapter{Interim results}` at line 1447 of `New_structure.tex`. The revised chapter references `ch:interim-results` twice. Not done by me, per your instruction to leave everything outside Chapter 3 alone. |

## 2. Corrected claims

Every verdict below is against the repository, not the draft. File:line references are to the repo.

| Your claim | Verdict | Evidence | Correction made |
|---|---|---|---|
| "run with a simple `make ci` command, which ran make lint/ruff/black/ci/test all at once" | Wrong | `Makefile:40–44`; comment at `:39` reads "without auto-fixing" | `make ci` runs Black in check mode, Ruff, mypy and pytest in sequence. It does not call `format` or `lint`, does not auto-fix, and does not call itself. Ruff *is* the linter, so "lint/ruff" double-counts one tool. |
| "a makefile for **four** simple commands" | Wrong | `Makefile:1` — `.PHONY: help venv install format lint typecheck test ci` | Eight. |
| "`make venv` creates a virtual environment in the directory that the command is run from" | Partly wrong | `Makefile:7,13–15` | Creates it at `venv/` in the repository root, guarded by testing whether `venv/bin/python` is executable. |
| "`make format` … in the /evaluation, /src, and /tests directory" | Partly wrong | `Makefile:24–25` | Four directories — `scripts/` is included too. |
| `make typecheck`, `make test`, `make ci` left as `{claude: add explanation}` | Missing | `Makefile:32–44` | Written, with the exact targets each covers. |
| ruff / black / mypy / pytest left as `{claude: add brief concise description}` | Missing | `pyproject.toml:85–142`, `tests/` | Written, with the actual configured rule sets, line length, strictness flags and suite size (97 test functions, 13 files, 12 unit + 1 integration). |
| "pyproject … differentiate between which packages were needed to run the entire script, versus subparts of it" | Wrong | `pyproject.toml:14–35` vs `:37–46`; `docs/CODING_STANDARDS.md:211–213` | The split is runtime dependencies vs a single `dev` tooling group. There are no per-feature extras. Corrected, and your `{claude: check if there is another glaring benefit}` answered with the two real ones: all four tools' configuration lives in the same file, and the editable install is what puts `src/agent_memories` on the import path — there is no `conftest.py` and no `sys.path` hack anywhere in `tests/`, so the tests genuinely depend on it. |
| "The Github repository was a simple copy of the Gitlab repository" | Partly wrong | `git remote -v`; `docs/CODING_STANDARDS.md:35,49,55` | GitHub is a push mirror configured on `origin` itself — one fetch URL, two push URLs — so a single `git push origin` updates both. Not a manual copy. |
| "merge it with the **main** branch" | Wrong | `git symbolic-ref refs/remotes/origin/HEAD` → `origin/master`; all merge commits read `into 'master'`; `CODING_STANDARDS.md:97,262` | `master`. Corrected in three places, including the LLM subsection. |
| "a pre-structured **merge message** … copy paste the **commit message** … to the merge tracker" | Partly wrong | `.gitlab/merge_request_templates/Default.md`; `.github/pull_request_template.md` | These are merge/pull *request* templates, not commit-message templates; there is no `.gitmessage` in the repo. Rewritten to describe what the template actually contains (summary, four-part checklist, testing-evidence block). Your point about pasting the passing test suite is correct and kept. |
| "as shown in figure xx below" | Wrong | — | `Figure~\ref{fig:Gitlab_merge_template}`. |
| "The pre-push hooks linked to the two LLM calls" + `{claude: should anything else be added here?}` | Correct but thin | `.githooks/pre-push:1–44`; `.claude/commands/thesis-tasks-curator.md:14–15` | Kept, and added the four facts an examiner would ask for: the hook is backgrounded and always exits 0 so it never blocks a push; it exits silently if `claude` is absent; the branch restriction lives in the curator command, not the hook; and the hook is tracked in `.githooks/` so it is version-controlled and needs activating on a fresh clone. |
| Coding standards document | Correct | `docs/CODING_STANDARDS.md` | Kept verbatim, with one sentence added naming which sections are generic and which are project-specific. |

**Not corrected, per your instruction**: everything in `\subsection{LLMs}` about how you used Claude. I only fixed language and the `main`→`master` slip there.

**One thing I did not touch but you should know about**: `docs/CODING_STANDARDS.md:296–298, 316–325` describes the curators as being fired by a `PostToolUse` hook in `.claude/settings.json`, and says they "only fire when push is run from inside a Claude Code session". That file does not exist, and the real trigger is the git-level `.githooks/pre-push`, which fires on any push from any terminal. Your standards document contradicts your repository. The chapter now describes the repository.

## 3. Structure

Per your answer, `\section{Frameworks}` was collapsed — `Agile` is now section 3.1 directly. No `\ref` anywhere in the thesis pointed at `sec:frameworks`, or at any other Chapter 3 label, so nothing else breaks.

`\section{Tool use}` subsections reordered into planning → writing → version control → code quality → LLMs. `Tuleap`, `Overleaf`, `GitLab and GitHub`, `Code quality tooling`, `LLMs`.

`\subsection{Coding tools}` renamed to `Code quality tooling`, answering your `{claude: check if there is a more appropriate subsection heading name}`. It stays in Chapter 3 per your answer, since Chapter 5's tech stack describes the research artefact while this describes how you worked.

Your `\subparagraph` pile (and the `\subparagrph` typo) became an `itemize` with `\textbf{Label.}` lead-ins, matching the environment you already use in the Agile section.

Two structural bugs fixed: the "daily scrum" paragraph was sitting inside the `Feedback` `\item` and before `\end{itemize}`, so it rendered as part of that bullet; and the inner `itemize` in the LLM subsection never closed, so `Refining text`, `Code checks` and `Source finding` were nested inside `Logging`.

## 4. Content added

Kept deliberately short — you asked for simplicity over completeness.

**SMART corrected to Doran's original criteria.** Your draft read "specific, measurable, **accessible**, realistic and time-bound"; that expansion came from the Anifa et al. review, not from Doran. It now reads "specific, measurable, **assignable**, realistic and time-related", matching Doran (1981), to whom it is cited. Note that "assignable" — specifying who does the work — is trivially satisfied by sole authorship, so it is the weakest of the five in your case. If that reads oddly to you under examination, the alternative is to keep "achievable" and cite the modern gloss instead; I have not assumed which you prefer.

**Two sentences in the opening paragraph** giving the Agile section a source, since it previously asserted a methodology with none. `\citep{beck2001manifesto}` for change-tolerance as a founding principle, `\citet{dyba2008empirical}` for the empirical backing.

**One sentence in "Product backlog"** noting that the Scrum Guide defines the backlog as emergent, so your divergence from the proposal's work packages is the framework working as intended rather than a failure to follow it. This turns a slightly apologetic passage into a defended one.

**Two sentences in the "what was not adhered to" paragraph.** Per your instruction, the text now states plainly that this thesis implements parts of the agile methodology rather than the framework in full, and backs that with `\citet{fitzgerald2006customising}` — a three-year study at Intel Shannon in which the agile methods in use were tailored to the team's practices rather than adopted wholesale. Partial adoption is therefore normal practice, not a shortfall, which is the point an examiner would otherwise press you on. `pagotto2016scrumsolo` has been dropped from the chapter and from the bib file.

The Fitzgerald record is fully verified: *European Journal of Information Systems* 15(2), 200–213, 2006, DOI `10.1057/palgrave.ejis.3000605`. Peer-reviewed, longitudinal, and heavily cited — a stronger source than the one it replaces, which was paywalled and unread.

**One paragraph on risk**, per your answer — no subsection, no table. It states that risk was handled at the weekly refinement rather than in a separate register, and points at the Interim results chapter for the pivots.

## 5. Language

Fixed throughout: `develoment`, `THis`, `requriements`, `nextw eek`, `change,d`, `auhtor` (×3), `adn` (×2), `receiveing`, `teh` (×3), `exerpt`, `otehrewise`, `coudl`, `Refing`, `orgiinated`, `Univeristy`, `methdology`, `differenciate`, `documnet`, `\subparagrph`.

Product names capitalised: GitLab, GitHub, LaTeX, Makefile, Black, Ruff.

First person removed. Your draft slipped into "I"/"my" in the Agile closing paragraph and throughout the LLM subsection, against your own academic register. Now impersonal, using "the author" as you do elsewhere.

Code identifiers moved from `\textsc{}` to `\texttt{}` with escaped underscores, matching the convention used in your later chapters. Flagging it rather than assuming: if `\textsc` was deliberate, it is a one-line revert.

Straight quotes in the user stories replaced with LaTeX directional quotes and wrapped in `\textit{}`, matching how you quote verbatim strings in Chapter 5.

Two sentences were rebuilt because their grammar collapsed rather than because I preferred different wording: *"decided how to proceed on what new items to add and what items were finalised"* and *"This message is again part of the author's general tech stack was created as part of the group project module"*.

Left untouched: your list environments, your ordering within each list, your signposting, "This author", "part of her every work cycle since", and the concession-then-justification shapes.

## 6. Mechanical verification

Run on the revised file:

- `itemize` 6/6, `figure` 2/2, nesting stack clean
- Braces balanced (132/132)
- No duplicate labels
- All `{claude: ...}` markers and `<...command>` placeholders removed
- Cross-references resolving inside the file: `fig:Gitlab_merge_template`, `sec:llms`
- Cross-reference resolving outside the file: `ch:interim-results` — **needs the label added, see B1**

## 7. Bonus finding outside Chapter 3

A whole-thesis check found nine `\ref` targets with no matching `\label`:

`ch:discussion`, `intermediate results`, `sec:dp-mechanism`, `sec:experiment design`, `sec:experiment-design`, `sec:interim results`, `sec:llm models`, `sec:memories`, `sec:memory-architecture`

Each will compile to a bold `??`. Note that `sec:experiment design` and `sec:experiment-design` are both used where the actual label is `sec:experiment_design`. Outside the scope you gave me, so untouched.

Your document also mixes `\cite{}` (Background) with `\citep{}`/`\citet{}` (Implementation). Chapter 3 uses `\citep`/`\citet` per your answer.
