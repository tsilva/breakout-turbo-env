from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import gymnasium as gym

from .action_tables import ACTION_SETS, ACTION_TABLES, BUTTONS, ActionTable
from .env import (
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RAW_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    BreakoutVecEnv,
)

GYMNASIUM_ENV_ID = "Breakout-Turbo-v0"
_GYMNASIUM_VECTOR_ENTRY_POINT = "breakout_turbo_env:_make_gymnasium_vec_env"
_COMPATIBILITY_ENV_SPECS = {
    "Breakout-Atari2600-v0": {
        "game": "Breakout-Atari2600-v0",
        "state": "Start",
        "scenario": "scenario",
        "info": "data",
        "use_restricted_actions": "filtered",
    }
}

try:
    __version__ = version("env-breakoutatari2600-turbo-native")
except PackageNotFoundError:  # Source tree imported without an installed distribution.
    __version__ = "0+unknown"


def _make_gymnasium_vec_env(
    *, game: str, num_envs: int = 1, **kwargs: Any
) -> BreakoutVecEnv:
    return BreakoutVecEnv(game=game, num_envs=num_envs, **kwargs)


def _register_gymnasium_envs() -> None:
    existing = gym.registry.get(GYMNASIUM_ENV_ID)
    if existing is None:
        gym.register(
            id=GYMNASIUM_ENV_ID,
            entry_point=None,
            vector_entry_point=_GYMNASIUM_VECTOR_ENTRY_POINT,
        )
    elif not (
        existing.entry_point is None
        and existing.vector_entry_point == _GYMNASIUM_VECTOR_ENTRY_POINT
        and existing.kwargs == {}
        and existing.max_episode_steps is None
        and existing.additional_wrappers == ()
    ):
        raise gym.error.Error(
            f"Gymnasium environment ID {GYMNASIUM_ENV_ID!r} is already "
            "registered with a conflicting specification"
        )

    for env_id, kwargs in _COMPATIBILITY_ENV_SPECS.items():
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point=None,
                vector_entry_point="breakout_turbo_env:BreakoutVecEnv",
                kwargs=kwargs,
            )


_register_gymnasium_envs()

__all__ = [
    "__version__",
    "BreakoutVecEnv",
    "GYMNASIUM_ENV_ID",
    "ACTION_SETS",
    "ACTION_TABLES",
    "ActionTable",
    "BUTTONS",
    "FIXED_POINT_ONE",
    "RAW_HEIGHT",
    "RAW_WIDTH",
    "RENDER_HEIGHT",
    "RENDER_WIDTH",
]
