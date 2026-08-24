"""Can we reach qwen3.5:4b-nvfp4 through OllamaClient?

Same client the WP1.3 Think node uses, so a clean run here means
Ollama is up, the model is pulled, the MLX runner is alive, and
/api/chat honours think: false. One hardcoded prompt, no flags,
no browser.

Needs ``ollama serve`` on localhost:11434 and
``ollama pull qwen3.5:4b-nvfp4``.
"""

from __future__ import annotations

from dotenv import load_dotenv

from agent_memories.config import load_random_seed
from agent_memories.services.ollama_client import OllamaClient


def main() -> None:
    load_dotenv()
    client = OllamaClient(model="qwen3.5:4b-nvfp4", seed=load_random_seed())
    reply = client.chat(
        system="You answer in one short sentence.",
        user="Say hello and name yourself.",
    )
    print(reply)


if __name__ == "__main__":
    main()
