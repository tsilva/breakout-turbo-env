from __future__ import annotations

from .action_tables import ACTION_SETS, ACTION_TABLES, BUTTONS, ActionTable
from .env import (
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RAW_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    BreakoutVecEnv,
)

__all__ = [
    "BreakoutVecEnv",
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

try:
    import gymnasium as gym

    gym.register(
        id="BreakoutTurbo-v0",
        entry_point=None,
        vector_entry_point="breakout_turbo_env:BreakoutVecEnv",
    )
    gym.register(
        id="Breakout-Atari2600-v0",
        entry_point=None,
        vector_entry_point="breakout_turbo_env:BreakoutVecEnv",
        kwargs={
            "game": "Breakout-Atari2600-v0",
            "state": "Start",
            "scenario": "scenario",
            "info": "data",
            "use_restricted_actions": "filtered",
        },
    )
except Exception:
    # Registration is best-effort so importing the native class stays usable in
    # minimal environments and duplicate imports remain harmless.
    pass
