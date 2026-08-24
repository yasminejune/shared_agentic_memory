"""Same logit dump as tokens_gemma.py, plus the argmax token printed.

Runs on MPS. Added the decode step so I could see whether the greedy
next token looked like English before trusting the vector.
"""

import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

device = torch.device("mps")

access_token = os.getenv("GEMMA_ACCESS_TOKEN")
login(access_token)

model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
).to(device)
model.eval()

prompt = "Generate text similar to: The approval threshold is £400."
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids)

# Next-token logits over the vocab; this is the vector Amin averages.
logits = outputs.logits  # [batch, seq_len, vocab_size]
next_token_logits = logits[0, -1, :]

print(f"Logit vector shape: {next_token_logits.shape}")
print(f"Vocabulary size: {next_token_logits.shape[0]}")
print(f"Sample logits (first 10): {next_token_logits[:10]}")

predicted_token_id = next_token_logits.argmax().item()
print(f"Predicted next token: '{tokenizer.decode([predicted_token_id])}'")
