"""HuggingFace Gemma 2 2B IT wrapper for Amin et al. Algorithm 1.

Exposes the minimal token-level surface that
:mod:`agent_memories.agent.privacy.privatisation` needs: encode a prompt
into ids (raw via :func:`encode` or via the model's chat template via
:func:`encode_chat`), decode ids back to text, fetch the stacked
next-token logit matrix for a list of prompts either from raw strings
(:func:`get_next_token_logits`) or from pre-tokenised id lists
(:func:`get_next_token_logits_from_ids` — saves the per-iteration
re-tokenisation cost when the same wrapped prompt is reused across
many sampling steps), and read the relevant stop-token ids
(:func:`eos_id`, :func:`stop_ids`). For the production WP2 sampling
loop the cached pair :func:`prefill_padded` plus :func:`continue_batched`
fold the ``s + 1`` per-token forwards (one per sensitive prompt plus
one public prompt) into one padded batched prefill followed by one
``(B, 1)`` continuation per token, reusing the prompt-prefix
``past_key_values`` rather than re-attending to the prefix every step.

The model is loaded lazily on the first call so that simply importing
the package (e.g. from a test or another script) does not download or
load Gemma. The default model name is ``google/gemma-2-2b-it`` per
``.claude/thesis/WP2-plan.md`` Section 5 — this is a deviation from
Amin et al. 2024 (who used Gemma 1.1 2B IT, paper §5); we pick the
current Gemma 2 generation because the algorithm and privacy proof
are model-agnostic and the 2B-IT architecture is unchanged. The
model is loaded in ``torch.bfloat16`` — its training dtype — which
preserves the fp32 dynamic range that Gemma 2's attention logit
soft-capping needs (fp16 produces silent NaN / Inf there) and roughly
halves the per-forward-pass cost on MPS versus fp32. All logits are
cast back to ``float32`` before they leave this module so downstream
sampling code in :mod:`.privatisation` is insulated from the inference
dtype. The project standard is IT + chat-template prompting via
:func:`encode_chat` and is used by the production
:mod:`agent_memories.agent.privacy.privatisation` path and every
sibling script. The default can be overridden via :func:`set_model`
for the IT-vs-base ablation in
:mod:`scripts.amin_et_al.compare_gemma`, which is the only entry
point that loads the raw-completion base ``google/gemma-2-2b``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b-it"

_tokenizer: Any = None
_model: Any = None


def _load() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    load_dotenv()
    access_token = os.getenv("GEMMA_ACCESS_TOKEN")
    if access_token:
        login(access_token)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    model_obj: Any = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    _model = model_obj.to("mps")
    _model.eval()


def set_model(name: str) -> None:
    """Configure which HuggingFace model the next :func:`_load` will pull.

    Rebinds :data:`MODEL_NAME` and clears the cached tokeniser and
    model so the next call into any function in this module rebuilds
    from the new ``name``. Call before any other function if you want
    to deviate from the default. The project standard is the IT
    default ``google/gemma-2-2b-it``; the only caller that swaps to
    the raw-completion base ``google/gemma-2-2b`` is the IT-vs-base
    comparison harness in :mod:`scripts.amin_et_al.compare_gemma`.
    """
    global MODEL_NAME, _tokenizer, _model
    MODEL_NAME = name
    _tokenizer = None
    _model = None


def encode(text: str) -> list[int]:
    """Tokenise ``text`` into a flat list of token ids."""
    _load()
    ids: list[int] = _tokenizer(text, return_tensors=None, add_special_tokens=True)["input_ids"]
    return ids


def encode_chat(prompt: str) -> list[int]:
    """Tokenise ``prompt`` as a single-turn user message via the chat template.

    Wraps the prompt in the model's chat template
    (``<start_of_turn>user ... <end_of_turn><start_of_turn>model``)
    and appends the assistant generation prompt suffix
    (``add_generation_prompt=True``) so the next sampled token
    continues the assistant's reply. Use with instruction-tuned
    models like ``google/gemma-2-2b-it``; raw-completion models like
    ``google/gemma-2-2b`` should use :func:`encode` instead so the
    WP2-plan §3.3 ``## Title`` anchor at the end of the wrapped
    prompt actually pre-fills the model's continuation rather than
    being buried inside the user turn.
    """
    _load()
    ids: list[int] = _tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    return ids


def decode(ids: list[int]) -> str:
    """Detokenise a list of token ids back into a string."""
    _load()
    text: str = _tokenizer.decode(ids, skip_special_tokens=True)
    return text


def eos_id() -> int:
    """Return the end-of-sequence token id for the model."""
    _load()
    return int(_tokenizer.eos_token_id)


def stop_ids() -> set[int]:
    """Return the set of token ids that should terminate generation.

    Always includes the tokeniser's ``eos_token_id``. For Gemma 2 IT
    (model id ending in ``-it``) the chat-template turn-end marker
    ``<end_of_turn>`` is also added so callers sampling under the
    chat template stop at the natural end of the assistant's turn
    rather than running on to a hard token cap. The raw-completion
    base Gemma 2 is never trained to emit ``<end_of_turn>``, so it is
    omitted from the stop set in that regime to keep the set
    structurally honest: the base model's only natural terminator is
    ``<eos>``. Discrimination is on the model id suffix to match the
    Gemma family naming convention; rebinding via :func:`set_model`
    therefore swaps the stop set automatically.
    """
    _load()
    ids: set[int] = set()
    if _tokenizer.eos_token_id is not None:
        ids.add(int(_tokenizer.eos_token_id))
    if MODEL_NAME.split("/")[-1].endswith("-it"):
        end_of_turn = _tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn, int) and end_of_turn != _tokenizer.unk_token_id:
            ids.add(end_of_turn)
    return ids


def get_next_token_logits(prompts: list[str]) -> torch.Tensor:
    """Stacked next-token logits, one row per prompt.

    Calls the model once per prompt (``len(prompts)`` forward passes)
    and stacks the last-position logit vectors into a single tensor of
    shape ``(len(prompts), vocab_size)``. This matches the prose form
    of Amin et al. Algorithm 1 line 9 ("run the model on each p in S")
    and the WP2-plan Section 3.1 step 2.i; it is mathematically
    equivalent to a single left-padded batched forward pass (causal
    self-attention does not cross padded inputs) but is easier to read
    and avoids the padding-mask plumbing.

    Used for both the sensitive batch ``Z`` and the single public
    prompt ``z_public`` in :mod:`.privatisation` (the public call wraps
    its prompt in a one-element list and indexes ``[0]``).
    """
    _load()
    rows: list[torch.Tensor] = []
    for prompt in prompts:
        inputs = _tokenizer(prompt, return_tensors="pt").to(
            "mps"
        )  # Ids make the lookup easier for the model rather than using the string
        # It also breaks up the prompt into tokens, which the model would have to do seperately otherwise.
        with torch.no_grad():
            outputs = _model(**inputs, logits_to_keep=1)
        rows.append(outputs.logits[0, -1, :])
        # Picks the the first (and only) item in the batch.
        # Then picks the last position in the sequence, since we are interested in the next token prediction.
        # We can ignore the tokenization of all other tokens in the prompt.
        # : because we are interested in the logit vector for all words in the vocabulary.

    return torch.stack(rows, dim=0)


def _pad_left(prompt_ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length id lists to one ``(B, T)`` tensor with mask.

    Left-padding is used so the last position of every padded row is
    that row's last real token; the next-token logit can then be read
    as ``logits[:, -1, :]`` without per-row indexing, and a single
    new token can be appended on the right during a cache continuation
    without disturbing the per-row alignment. ``pad_id`` is the id
    written into the padded slots; the returned ``attention_mask``
    (1 for real tokens, 0 for pads) is the source of truth the model
    uses to ignore them, so the literal id value is structurally
    irrelevant as long as the mask is honoured.
    """
    max_len = max(len(ids) for ids in prompt_ids)
    batch_size = len(prompt_ids)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(prompt_ids):
        n = len(ids)
        input_ids[i, max_len - n :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - n :] = 1
    return input_ids, attention_mask


def get_next_token_logits_from_ids(prompt_ids: list[list[int]]) -> torch.Tensor:
    """Stacked next-token logits given pre-tokenised prompt id lists.

    Functionally equivalent to :func:`get_next_token_logits` but
    accepts already-tokenised inputs. Lets the caller tokenise each
    wrapped prompt once and only concatenate the freshly sampled
    token ids per sampling step, avoiding the per-iteration
    re-tokenisation that the string-based path pays. Shape of the
    returned tensor is ``(len(prompt_ids), vocab_size)``, matching
    :func:`get_next_token_logits`.

    Internally the rows are left-padded to a common length via
    :func:`_pad_left` and run through the model in a single batched
    forward pass with an explicit attention mask: causal self-attention
    does not cross between rows and the mask blocks attention to the
    pads, so the result is mathematically identical to the prior
    one-forward-pass-per-prompt loop but pays only one kernel-launch
    overhead per call. Returned logits are cast to ``float32`` so
    downstream sampling code is insulated from the model's bf16
    inference dtype.
    """
    _load()
    pad_id = int(_tokenizer.pad_token_id)
    input_ids, attention_mask = _pad_left(prompt_ids, pad_id)
    input_ids = input_ids.to("mps")
    attention_mask = attention_mask.to("mps")
    with torch.no_grad():
        outputs = _model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
    logits: torch.Tensor = outputs.logits[:, -1, :].float()
    return logits


@dataclass
class PrefillState:
    """Opaque KV-cache state returned by :func:`prefill_padded`.

    Holds the HuggingFace ``past_key_values`` cache and the running
    ``attention_mask`` so :func:`continue_batched` can extend both by
    one position per call. Treat as opaque from outside the module —
    the only valid operation is to thread the returned instance back
    into :func:`continue_batched`.
    """

    past_key_values: Any
    attention_mask: torch.Tensor


def prefill_padded(
    prompt_ids: list[list[int]],
) -> tuple[torch.Tensor, PrefillState]:
    """Padded batched prefill: one forward pass over all prompts, cache K/V.

    Pads ``prompt_ids`` to a common length via :func:`_pad_left`, runs
    the full model forward once with ``use_cache=True``, and returns
    the per-row next-token logits together with a :class:`PrefillState`
    that the caller can thread into :func:`continue_batched` so the
    prompt prefix is never re-attended-to in subsequent sampling
    steps. Used by the production WP2 path
    (:func:`agent_memories.agent.privacy.privatisation.generate`),
    which stacks the ``s`` sensitive prompts together with the
    single public prompt into one ``s + 1`` batch so the whole
    sampling loop runs as one prefill plus one tiny per-token
    continuation rather than ``s + 1`` independent full-prompt
    forwards per token. Passes ``logits_to_keep=1`` so Gemma's
    ``lm_head`` materialises only the last-position logits; the
    default ``logits_to_keep=0`` keeps every position and on MPS
    with ``B ~ 100`` long dialogues the ``(batch, seq, vocab)``
    tensor exceeds Metal's buffer limit (~36 GiB at this project's
    batch and sequence lengths).
    """
    _load()
    pad_id = int(_tokenizer.pad_token_id)
    input_ids, attention_mask = _pad_left(prompt_ids, pad_id)
    input_ids = input_ids.to("mps")
    attention_mask = attention_mask.to("mps")
    with torch.no_grad():
        outputs = _model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1, :].float()
    return logits, PrefillState(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
    )


def continue_batched(
    state: PrefillState,
    new_token_ids: list[int],
) -> tuple[torch.Tensor, PrefillState]:
    """One-token continuation using cached K/V; returns updated logits and state.

    ``new_token_ids[i]`` is the next token to append to row ``i``.
    The model sees only ``(B, 1)`` of fresh input ids per call;
    every position from the prefill (and from previous continuations)
    is re-used from ``state.past_key_values`` without re-attention.
    The running ``attention_mask`` is extended by one column of ones
    so HuggingFace's combined past + new mask shape
    ``(B, past_len + 1)`` stays consistent. Returns the per-row
    next-token logits (cast to ``float32``) and a fresh
    :class:`PrefillState` carrying the now-longer cache.
    """
    _load()
    batch_size = len(new_token_ids)
    input_ids = torch.tensor(new_token_ids, dtype=torch.long).unsqueeze(-1).to("mps")
    new_mask_col = torch.ones(
        (batch_size, 1),
        dtype=state.attention_mask.dtype,
        device=state.attention_mask.device,
    )
    attention_mask = torch.cat([state.attention_mask, new_mask_col], dim=1)
    with torch.no_grad():
        outputs = _model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1, :].float()
    return logits, PrefillState(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
    )


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the last dimension."""
    return torch.softmax(logits, dim=-1)
