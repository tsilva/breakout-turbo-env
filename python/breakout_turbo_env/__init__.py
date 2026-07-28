from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .action_tables import ACTION_SETS, ACTION_TABLES, BUTTONS, ActionTable
from .env import (
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RAW_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    BreakoutVecEnv,
)

try:
    __version__ = version("breakout-turbo-env")
except PackageNotFoundError:  # Source tree imported without an installed distribution.
    __version__ = "0+unknown"

__all__ = [
    "__version__",
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
