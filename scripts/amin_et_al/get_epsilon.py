from __future__ import annotations

from agent_memories.agent.privacy.privacy_accounting import get_epsilon


def main():
    """This function gets the approximate epsilon using the closed form equation from the Amin et al. paper.
    The infimum operation could be used to get a tighter epsilon, but is not implemented here for simplicity.
    """
    epsilon = get_epsilon(r=25, s=10, c=10.0, tau=1.0, sigma=0.1)
    # Delta is always set to 1 / s
    print(f"Epsilon: {epsilon[0]}, Delta: {epsilon[1]}")


if __name__ == "__main__":
    main()
