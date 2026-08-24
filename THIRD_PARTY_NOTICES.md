# Third-party notices

This repository is distributed as source. Dependencies are declared in
`pyproject.toml` and installed from PyPI by `make install`; no third-party code
is vendored here. The permissive entries below are therefore informational
rather than an attribution obligation this project carries. That changes if a
wheel, container image, or vendored copy is ever shipped, in which case each
package's notices have to travel with it.

Two entries are not informational. `invink` sets the licence of this project,
and Gemma 2 carries use terms that apply whether or not anything is
redistributed.

## invink (GPL-3.0-only)

`invink` 0.1.0, copyright Vishnu Vinod. Source:
<https://github.com/cerai-iitm/invisibleink>.

The published wheel declares the `GPLv3` PyPI classifier, not `GPLv3+`, and its
per-file headers read `License: GPLv3` with no "or later" wording. This project
therefore treats it as GPL-3.0 **only** and is licensed GPL-3.0-only to match.
See `LICENSE`.

Used for the differential privacy primitives:

- `src/agent_memories/agent/invisible_ink/accounting.py` imports `cdp_eps`,
  `cdp_rho`, `compute_rho`, `get_clip`, `get_epsilon`.
- `src/agent_memories/agent/invisible_ink/mechanism.py` imports
  `difference_clip`, `get_topk`.
- `scripts/labels/compare_clustering_methods_extensive.py` imports `cdp_eps`.

`invink.generate` is not called. The generation loop in
`src/agent_memories/agent/invisible_ink/generation.py`, the prompting, the OOM
fallback and the memory-store integration are original work; the accounting and
masking primitives above are upstream.

`scripts/amin_et_al/modified_amin_et_al.py` reimplements several of the same
accounting functions locally, written by reference to the invink
implementations it names in its docstrings.

Two questions are open with the upstream authors: whether GPLv3 is intended as
only or or-later, and how they read import-level use of `invink.utils`.

## google/gemma-2-2b-it

Weights are not redistributed here. They are downloaded from HuggingFace at run
time using `GEMMA_ACCESS_TOKEN`. Use is governed by the Gemma Terms of Use and
the Gemma Prohibited Use Policy, which apply to use and to derivatives, not only
to redistribution.

- <https://ai.google.dev/gemma/terms>
- <https://ai.google.dev/gemma/prohibited_use_policy>

## Other dependencies

Versions are those resolved in the development environment. All are permissive.

| Package | Version | Licence |
|---|---|---|
| bertopic | 0.17.4 | MIT |
| browsergym | 0.13.3 | Apache-2.0 |
| browsergym-webarena | 0.13.3 | Apache-2.0 |
| greenlet | 3.5.1 | MIT AND PSF-2.0 |
| gymnasium | 1.3.0 | MIT |
| hdbscan | 0.8.44 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| huggingface_hub | 0.36.2 | Apache-2.0 |
| kagglehub | 1.0.2 | Apache-2.0 |
| langgraph | 1.0.10 | MIT |
| matplotlib | 3.11.0 | PSF-based (matplotlib licence) |
| mistralai | 2.4.7 | Apache-2.0 |
| nltk | 3.9.4 | Apache-2.0 |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pandas | 3.0.3 | BSD-3-Clause |
| playwright | 1.60.0 | Apache-2.0 |
| python-dotenv | 1.2.2 | BSD-3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| sentence-transformers | 5.5.1 | Apache-2.0 |
| torch | 2.12.0 | BSD-3-Clause |
| transformers | 4.57.6 | Apache-2.0 |
| umap-learn | 0.5.12 | BSD-3-Clause |

Transitive dependencies are not listed. The only non-permissive ones present are
`certifi`, `pathspec` and `tqdm`, all MPL-2.0, which is file-level copyleft and
imposes nothing on an unmodified consumer.
