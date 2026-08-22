"""Print Amin's closed-form epsilon for a fixed (r, s, c, tau, sigma)."""

from __future__ import annotations

from agent_memories.agent.privacy.privacy_accounting import get_epsilon


def main():
    """Closed-form Amin epsilon; skips the tighter infimum from the paper."""
    epsilon = get_epsilon(r=25, s=10, c=10.0, tau=1.0, sigma=0.1)
    # Delta is always set to 1 / s
    print(f"Epsilon: {epsilon[0]}, Delta: {epsilon[1]}")


if __name__ == "__main__":
    main()
