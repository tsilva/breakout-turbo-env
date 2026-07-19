from __future__ import annotations

from .env import (
    BreakoutVecEnv,
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RAW_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
)

__all__ = [
    "BreakoutVecEnv",
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
except Exception:
    # Registration is best-effort so importing the native class stays usable in
    # minimal environments and duplicate imports remain harmless.
    pass
