"""Pull the next-token logit vector from Gemma 2 IT.

Amin Algorithm 1 needs the raw vocabulary logits, not decoded text.
Smallest script that loads the instruct model and prints the vector
shape, so I know the HF forward pass is what I think it is.
"""

import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

access_token = os.getenv("GEMMA_ACCESS_TOKEN")
login(access_token)

model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
model.eval()

prompt = "Generate text similar to: The approval threshold is £400."
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(input_ids=input_ids)

# Amin Algorithm 1 consumes this: raw logit vector over the vocabulary.
logits = outputs.logits  # [batch, seq_len, vocab_size]
next_token_logits = logits[0, -1, :]

print(f"Logit vector shape: {next_token_logits.shape}")
print(f"Vocabulary size: {next_token_logits.shape[0]}")
print(f"Sample logits (first 10): {next_token_logits[:10]}")
