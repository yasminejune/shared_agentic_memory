"""Think-Act-Observe agent loop (WP1.1 scaffold).

The graph is intentionally small: one observe-think-act cycle on a
controlled Playwright page, with a deterministic stub Think node.
The real LLM is wired in later via the Mistral API.
"""

from .graph import build_graph
from .state import AgentState, new_state

__all__ = ["AgentState", "build_graph", "new_state"]
