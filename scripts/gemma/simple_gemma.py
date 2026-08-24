"""HuggingFace pipeline smoke test for google/gemma-2-2b-it.

One pirate-speak prompt on MPS. Checking that the token, the weights,
and device="mps" all work before wiring Gemma into Amin.
"""

import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline

load_dotenv()

access_token = os.getenv("GEMMA_ACCESS_TOKEN")
login(access_token)

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"dtype": torch.bfloat16},
    device="mps",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
