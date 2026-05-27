"""Smallest possible sanity check that ``qwen3.5:4b-nvfp4`` is reachable via Ollama.

Uses the project's own :class:`OllamaClient` (the same client the WP1.3
agent goes through) so a clean run here proves the full path -- Ollama
up, model pulled, MLX runner subprocess engaged, native ``/api/chat``
honouring ``think: false`` -- that the agent depends on. Hardcoded
prompt, no CLI flags, no browser: one chat call and a print.

Prerequisites:

* ``ollama serve`` running on http://localhost:11434
* ``ollama pull qwen3.5:4b-nvfp4``
"""

from __future__ import annotations

from agent_memories.services.ollama_client import OllamaClient


def main() -> None:
    client = OllamaClient(model="qwen3.5:4b-nvfp4")
    reply = client.chat(
        system="You answer in one short sentence.",
        user="Say hello and name yourself.",
    )
    print(reply)


if __name__ == "__main__":
    main()
