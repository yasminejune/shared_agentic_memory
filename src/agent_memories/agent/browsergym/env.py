"""BrowserGym-backed WebArena environment wrapper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from browsergym.core.action.highlevel import HighLevelActionSet

from .observation import preprocess_obs

WEBARENA_ACTION_SET = HighLevelActionSet(subsets=["webarena"], multiaction=False)


def build_think_system_prompt(action_set: HighLevelActionSet) -> str:
    grammar = action_set.describe(
        with_long_description=False,
        with_examples=True,
    )
    return (
        "You are a web agent controlling a browser via BrowserGym WebArena actions.\n"
        "Reply with EXACTLY one line: a single Python function call, no commentary.\n"
        "\n"
        "Use bid as the element id from the accessibility tree.\n"
        "\n"
        f"{grammar.strip()}\n"
    )


THINK_SYSTEM_PROMPT = build_think_system_prompt(WEBARENA_ACTION_SET)


def make_webarena_env(task_id: int, *, headless: bool = True) -> gym.Env:
    """Create a BrowserGym WebArena environment for ``task_id``."""
    return gym.make(
        f"browsergym/webarena.{task_id}",
        headless=headless,
        slow_mo=0,
        action_mapping=WEBARENA_ACTION_SET.to_python_code,
    )


class WebArenaEnvWrapper:
    """Thin holder around a BrowserGym env with preprocessed observations."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.last_obs: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        self.last_reward: float = 0.0
        self.last_terminated: bool = False

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset()
        self.last_obs = preprocess_obs(obs)
        self.last_info = info
        self.last_reward = 0.0
        self.last_terminated = False
        return self.last_obs, info

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = preprocess_obs(obs)
        self.last_info = info
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated or truncated)
        return self.last_obs, self.last_reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()
