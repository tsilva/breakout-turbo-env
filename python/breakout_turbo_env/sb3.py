"""Optional Stable-Baselines3 adapter for :class:`BreakoutVecEnv`.

Stable-Baselines3 vector environments expose an auto-reset API.  The native
Breakout environment deliberately does not.  This adapter is the sole place
where that translation occurs: terminal observations are preserved in
``info`` and only completed lanes are reset.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _indices(indices: None | int | Iterable[int], num_envs: int) -> list[int]:
    if indices is None:
        return list(range(num_envs))
    if isinstance(indices, int):
        return [indices]
    return list(indices)


def _lane_infos(infos: dict[str, np.ndarray], num_envs: int) -> list[dict[str, Any]]:
    """Convert Gymnasium's masked dict-of-arrays infos to SB3 lane dictionaries."""

    result: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    for key, values in infos.items():
        if key.startswith("_"):
            continue
        present = infos.get(f"_{key}")
        for lane in range(num_envs):
            if present is None or bool(present[lane]):
                result[lane][key] = values[lane]
    return result


class _SB3ManualResetMixin:
    """Implementation shared with a dynamically imported SB3 ``VecEnv`` base."""

    def __init__(self, env: Any):
        self.env = env
        self._actions: np.ndarray | None = None
        super().__init__(
            num_envs=env.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
        )

    def reset(self) -> np.ndarray:
        seed = None if all(value is None for value in self._seeds) else self._seeds
        if any(self._options):
            if not all(value == self._options[0] for value in self._options):
                raise ValueError("BreakoutVecEnv requires common SB3 reset options")
            options = self._options[0]
        else:
            options = None
        observations, infos = self.env.reset(seed=seed, options=options)
        self.reset_infos = _lane_infos(infos, self.num_envs)
        self._reset_seeds()
        self._reset_options()
        self._actions = None
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        if self._actions is not None:
            raise RuntimeError("step_async called while another step is pending")
        self._actions = np.asarray(actions)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called without step_async")
        actions, self._actions = self._actions, None
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        done = np.asarray(terminated | truncated, dtype=np.bool_)
        lane_infos = _lane_infos(infos, self.num_envs)

        for lane in np.flatnonzero(done):
            lane_infos[lane]["terminal_observation"] = observations[lane].copy()
            lane_infos[lane]["TimeLimit.truncated"] = bool(
                truncated[lane] and not terminated[lane]
            )

        if done.any():
            observations, reset_infos = self.env.reset(options={"reset_mask": done})
            reset_lane_infos = _lane_infos(reset_infos, self.num_envs)
            for lane in np.flatnonzero(done):
                self.reset_infos[lane] = reset_lane_infos[lane]

        return observations, rewards, done, lane_infos

    def close(self) -> None:
        self.env.close()

    def get_images(self) -> list[np.ndarray]:
        return [self.env.render_lane(lane) for lane in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        value = getattr(self.env, attr_name)
        return [value for _ in _indices(indices, self.num_envs)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        selected = _indices(indices, self.num_envs)
        if selected != list(range(self.num_envs)):
            raise ValueError("the wrapped vector environment only supports global attributes")
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        selected = _indices(indices, self.num_envs)
        if selected != list(range(self.num_envs)):
            raise ValueError("the wrapped vector environment only supports global methods")
        value = getattr(self.env, method_name)(*args, **kwargs)
        return [value for _ in selected]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False for _ in _indices(indices, self.num_envs)]


def make_sb3_vec_env(env: Any):
    """Wrap a ``BreakoutVecEnv`` for Stable-Baselines3.

    Stable-Baselines3 is intentionally not a package dependency. Install it
    separately before calling this function.
    """

    try:
        from stable_baselines3.common.vec_env import VecEnv
    except ImportError as error:  # pragma: no cover - exercised by users without SB3
        raise ImportError(
            "Stable-Baselines3 is optional; install `stable-baselines3` to use "
            "make_sb3_vec_env"
        ) from error

    class SB3ManualResetVecEnv(_SB3ManualResetMixin, VecEnv):
        pass

    return SB3ManualResetVecEnv(env)
