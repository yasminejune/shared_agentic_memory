from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os

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
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# This is what Algorithm 1 needs — raw logit vector over vocabulary
logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
next_token_logits = logits[0, -1, :]  # logits for next token prediction

print(f"Logit vector shape: {next_token_logits.shape}")
print(f"Vocabulary size: {next_token_logits.shape[0]}")
print(f"Sample logits (first 10): {next_token_logits[:10]}")

predicted_token_id = next_token_logits.argmax().item()
print(f"Predicted next token: '{tokenizer.decode([predicted_token_id])}'")