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
    device="mps",  # replace with "mps" to run on a Mac device
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
# Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜
