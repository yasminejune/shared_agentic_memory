# 12k chars (~3k tokens)
OBSERVATION_CHAR_BUDGET = 12_000

# Abort after 5 identical thoughts in a row
DEFAULT_STUCK_THRESHOLD = 5

# ReasoningBank (Ouyang et al. 2025, Appendix A.2) prompt
MEMORY_INJECTION_INSTRUCTION = (
    "Below are some memory items that I accumulated from past interaction "
    "from the environment that may be helpful to solve the task. You can "
    "use it when you feel it's relevant. In each step, please first "
    "explicitly discuss if you want to use each memory item or not, and "
    "then take action."
)
