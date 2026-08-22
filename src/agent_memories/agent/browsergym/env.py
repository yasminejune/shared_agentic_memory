"""Gym wrapper around BrowserGym WebArena plus our action mapping."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import FunctionType
from typing import Any, cast

import gymnasium as gym
from browsergym.core.action import utils as action_utils
from browsergym.core.action.functions import (
    click,
    fill,
    go_back,
    hover,
    keyboard_press,
    new_tab,
    noop,
    report_infeasible,
    scroll,
    select_option,
    send_msg_to_user,
    tab_close,
    tab_focus,
)
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.action.parsers import highlevel_action_parser

from . import deferred_judge
from .navigation import go_home, goto
from .observation import preprocess_obs

# Stock webarena set minus goto/go_forward; our sandboxed goto and go_home instead.
# noop is always allowed by BrowserGym; listed so it is in the exec mapping.
_WEBARENA_ACTIONS: list[Callable[..., Any]] = [
    noop,
    scroll,
    keyboard_press,
    click,
    fill,
    hover,
    tab_focus,
    new_tab,
    go_back,
    tab_close,
    select_option,
    send_msg_to_user,
    report_infeasible,
    goto,
    go_home,
]

WEBARENA_ACTION_SET = HighLevelActionSet(
    subsets=["custom"],
    custom_actions=[fn for fn in _WEBARENA_ACTIONS if fn is not noop],
    multiaction=False,
    demo_mode="off",
)


def _build_python_includes() -> str:
    """Assemble action source without BrowserGym's broken python_includes indent."""
    parts = [
        "import playwright.sync_api\n",
        "from typing import Literal\n\n\n",
        "demo_mode='off'\n",
        "retry_with_force=False\n\n",
    ]
    for _, func in inspect.getmembers(action_utils, inspect.isfunction):
        parts.append(inspect.getsource(func))
        parts.append("\n\n")
    for action_fn in cast(list[FunctionType], _WEBARENA_ACTIONS):
        parts.append(inspect.getsource(action_fn))
        parts.append("\n\n")
    return "".join(parts)


_PYTHON_INCLUDES = _build_python_includes()


def webarena_action_to_python(action: str) -> str:
    """Map a high-level action string to executable Python for BrowserGym.

    Reimplements HighLevelActionSet.to_python_code because browsergym-core
    0.13.3 leaves leading whitespace in python_includes; exec then raises
    IndentationError on every action.
    """
    function_calls = highlevel_action_parser.search_string(action)
    function_calls = sum(function_calls.as_list(), [])
    if not function_calls:
        raise ValueError("Received an empty action.")
    if len(function_calls) > 1:
        raise ValueError("Received a multi-action, only single-actions are allowed.")

    function_name, function_args = function_calls[0]
    if function_name not in WEBARENA_ACTION_SET.action_set:
        raise NameError(f"Invalid action type '{function_name}'.")

    call = str(function_name) + "(" + ", ".join([repr(arg) for arg in function_args]) + ")\n"
    return _PYTHON_INCLUDES + call


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
    """Open a BrowserGym WebArena env for one of the 812 tasks."""
    return gym.make(
        f"browsergym/webarena.{task_id}",
        headless=headless,
        slow_mo=0,
        action_mapping=webarena_action_to_python,
    )


class WebArenaEnvWrapper:
    """Holds a BrowserGym env and the last preprocessed obs, reward, and judge stubs."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.last_obs: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        self.last_reward: float = 0.0
        self.last_terminated: bool = False
        self.last_judge_calls: list[dict[str, Any]] = []

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset()
        self.last_obs = preprocess_obs(obs)
        self.last_info = info
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_judge_calls = []
        return self.last_obs, info

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        deferred_judge.reset()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = preprocess_obs(obs)
        self.last_info = info
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated or truncated)
        self.last_judge_calls = deferred_judge.calls()
        return self.last_obs, self.last_reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()
