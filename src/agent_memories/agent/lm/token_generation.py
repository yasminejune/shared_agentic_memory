"""Gemma 2 2B IT logit wrapper shared by Amin et al and InvisibleInk.

Loads ``google/gemma-2-2b-it`` in bfloat16 on MPS, applies the
chat template, and returns last-position logits as float32. Prefill
plus one-token continuation reuses the KV cache so the sampling loop
does not re-calculate the prompt prefix each step. set_model() swaps the
HuggingFace id (IT-vs-base comparison).
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
    """Bind a HuggingFace model id and clear the cached tokeniser and weights.

    The name is the model name. Default is
    ``google/gemma-2-2b-it``
    """
    global MODEL_NAME, _tokenizer, _model
    MODEL_NAME = name
    _tokenizer = None
    _model = None


def encode(text: str) -> list[int]:
    """Tokenise the text into a flat list of token ids."""
    _load()  # Loads the gemma model and tokenizer
    ids: list[int] = _tokenizer(text, return_tensors=None, add_special_tokens=True)["input_ids"]
    return ids


def encode_chat(prompt: str) -> list[int]:
    """Tokenise ``prompt`` as a single-turn user message via the chat template.

    Applies Gemma's ``<start_of_turn>user ... <end_of_turn><start_of_turn>model``
    wrapper with ``add_generation_prompt=True`` so the next token continues
    the assistant reply. Only instruction-tuned models use this, whereas raw-completion
    models should use encode() instead.
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
    """Token ids that should terminate generation.

    Includes eos, and for the instruction-tuned Gemma, it also includes
    ``<end_of_turn>``.
    """
    _load()
    ids: set[int] = set()
    if _tokenizer.eos_token_id is not None:
        ids.add(int(_tokenizer.eos_token_id))
    if MODEL_NAME.split("/")[-1].endswith("-it"):  # I.e. is the Gemma IT model
        end_of_turn = _tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn, int) and end_of_turn != _tokenizer.unk_token_id:
            ids.add(end_of_turn)
    return ids


def get_next_token_logits(prompts: list[str]) -> torch.Tensor:
    """Get the stacked next-token logits from a raw text.
    Each row is for a new prompt/memory, thus the shape is (n, vocab size).
    """
    _load()
    rows: list[torch.Tensor] = []
    for prompt in prompts:
        inputs = _tokenizer(prompt, return_tensors="pt").to("mps")
        with torch.no_grad():
            outputs = _model(**inputs, logits_to_keep=1)
        rows.append(outputs.logits[0, -1, :])

    return torch.stack(rows, dim=0)


def _pad_left(prompt_ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad id lists to size (B, T) so the last column is each row's last real token.

    Next-token logits are then logits[:, -1, :]. Pads are ignored via
    the attention mask (1 = real, 0 = pad).
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
    """Stacked next-token logits from a list of already tokenised ids.

    Left-pads and runs one batched forward. Logits are returned as
    float32.
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
    """KV-cache and running attention mask from prefill_padded."""

    past_key_values: Any
    attention_mask: torch.Tensor


def prefill_padded(
    prompt_ids: list[list[int]],
) -> tuple[torch.Tensor, PrefillState]:
    """Batched prefill: last-position logits and a KV-cache PrefillState.

    Left-pads the prompt_ids and runs one forward pass.
    Thread the state into continue_batched so later tokens do not
    re-attend the prefix. The logits_to_keep=1 means only the last
    position is materialised.
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
    """One-token continuation from cached K/V; return new logits and state."""
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
    """Softmax over vocabulary size"""
    return torch.softmax(logits, dim=-1)
