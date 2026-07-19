from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from ._breakout_turbo import (
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RAW_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    NativeBreakoutVecEnv,
)

_SIGNAL_NAMES = (
    "paddle_x",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "brick_mask",
    "score",
    "lives",
    "tick",
    "bricks_remaining",
    "layout_id",
    "collision_events",
    "pending_reset",
)
_START_IDS = ("full", "checker", "tunnel", "sparse")
_ATARI_2600_NTSC_PALETTE = np.array(
    [
        [0, 0, 0],
        [136, 136, 136],
        [200, 72, 72],
        [192, 104, 56],
        [176, 120, 48],
        [160, 160, 40],
        [72, 160, 72],
        [64, 72, 200],
        [64, 152, 128],
    ],
    dtype=np.uint8,
)


class BreakoutVecEnv(VectorEnv):
    """Native deterministic Breakout vector environment.

    The only lifecycle is manual/disabled autoreset. Any selected lane may be
    reset at any time with ``reset(options={"reset_mask": mask})``. A terminal
    lane must be reset before the next call to ``step``.
    """

    metadata = {
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        obs_resize: tuple[int, int] = (84, 84),
        obs_crop: tuple[int, int, int, int] | None = None,
        obs_crop_mode: str = "remove",
        obs_crop_fill: int = 0,
        obs_resize_algorithm: str = "area",
        frame_skip: int = 4,
        frame_stack: int = 4,
        maxpool_last_two: bool = False,
        obs_layout: str = "chw",
        obs_grayscale: bool = True,
        obs_copy: str = "safe_view",
        info_filter: str | Mapping[str, Any] = "all",
        render_mode: str = "rgb_array",
        **unsupported: Any,
    ):
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported option(s): {names}")
        if maxpool_last_two:
            raise ValueError("maxpool_last_two is not implemented and must be False")
        if str(obs_layout).lower() != "chw":
            raise ValueError("obs_layout is fixed to 'chw' for the rlab policy contract")
        if not obs_grayscale:
            raise ValueError("obs_grayscale is fixed to True for the rlab policy contract")
        if obs_resize_algorithm != "area":
            raise ValueError("obs_resize_algorithm is fixed to 'area' for the rlab policy contract")
        if obs_crop_mode not in {"remove", "mask"}:
            raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
        if not isinstance(obs_crop_fill, int) or isinstance(obs_crop_fill, bool) or not 0 <= obs_crop_fill <= 255:
            raise ValueError("obs_crop_fill must be an integer in [0, 255]")
        if render_mode != "rgb_array":
            raise ValueError("render_mode must be 'rgb_array'")
        num_envs = int(num_envs)
        frame_skip = int(frame_skip)
        frame_stack = int(frame_stack)
        if num_envs <= 0 or frame_skip <= 0 or frame_stack <= 0:
            raise ValueError("num_envs, frame_skip, and frame_stack must be positive")
        if len(obs_resize) != 2 or min(int(v) for v in obs_resize) <= 0:
            raise ValueError("obs_resize must contain positive (height, width)")
        obs_h, obs_w = (int(obs_resize[0]), int(obs_resize[1]))
        crop = (0, 0, 0, 0) if obs_crop is None else tuple(int(value) for value in obs_crop)
        if len(crop) != 4 or min(crop) < 0:
            raise ValueError("obs_crop must contain non-negative (top, bottom, left, right)")
        if crop[0] + crop[1] >= RAW_HEIGHT or crop[2] + crop[3] >= RAW_WIDTH:
            raise ValueError("obs_crop removes the entire source image")
        if obs_copy not in {"copy", "safe_view", "unsafe_view"}:
            raise ValueError("obs_copy must be 'copy', 'safe_view', or 'unsafe_view'")

        if isinstance(info_filter, Mapping):
            self._info_mode = str(info_filter.get("mode", "all"))
            keys = info_filter.get("keys")
            self._info_keys = tuple(_SIGNAL_NAMES if keys is None else map(str, keys))
        else:
            self._info_mode = str(info_filter)
            self._info_keys = _SIGNAL_NAMES
        if self._info_mode not in {"all", "terminal", "none"}:
            raise ValueError("info_filter mode must be 'all', 'terminal', or 'none'")
        unknown = set(self._info_keys) - set(_SIGNAL_NAMES)
        if unknown:
            raise ValueError(f"unknown info keys: {sorted(unknown)}")

        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.obs_layout = "chw"
        self.obs_copy = obs_copy
        self.autoreset_mode = AutoresetMode.DISABLED
        self.render_mode = render_mode
        self.initial_state_names = _START_IDS
        self.single_action_space = gym.spaces.Discrete(3)
        self.action_space = gym.spaces.MultiDiscrete(np.full(num_envs, 3, dtype=np.int64))
        self.single_observation_space = gym.spaces.Box(
            0, 255, shape=(frame_stack, obs_h, obs_w), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(num_envs, frame_stack, obs_h, obs_w), dtype=np.uint8
        )
        threads = num_envs if num_threads is None else int(num_threads)
        if threads <= 0:
            raise ValueError("num_threads must be positive")
        self.native = NativeBreakoutVecEnv(
            num_envs,
            obs_h,
            obs_w,
            frame_skip,
            frame_stack,
            threads,
            list(crop),
            obs_crop_mode == "mask",
            obs_crop_fill,
        )
        count = 1 if obs_copy == "unsafe_view" else 2
        self._obs_buffers = [
            np.empty((num_envs, frame_stack, obs_h, obs_w), dtype=np.uint8)
            for _ in range(count)
        ]
        self._reward_buffers = [np.empty(num_envs, dtype=np.float32) for _ in range(count)]
        self._terminated_buffers = [np.empty(num_envs, dtype=np.bool_) for _ in range(count)]
        self._truncated_buffers = [np.empty(num_envs, dtype=np.bool_) for _ in range(count)]
        self._signal_buffers = [np.empty((num_envs, len(_SIGNAL_NAMES)), dtype=np.int64) for _ in range(count)]
        self._buffer_index = 0
        self._active_state_indices = np.zeros(num_envs, dtype=np.int32)
        self._active_state_indices.setflags(write=False)
        self._last_signals = self._signal_buffers[0]
        self._closed = False

    def _next_buffers(self):
        index = self._buffer_index
        self._buffer_index = (self._buffer_index + 1) % len(self._obs_buffers)
        return (
            self._obs_buffers[index],
            self._reward_buffers[index],
            self._terminated_buffers[index],
            self._truncated_buffers[index],
            self._signal_buffers[index],
        )

    def _obs(self, observations: np.ndarray) -> np.ndarray:
        return observations.copy() if self.obs_copy == "copy" else observations

    def _infos(self, signals: np.ndarray, present: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if self._info_mode == "none":
            return {}
        if present is None:
            present = np.ones(self.num_envs, dtype=np.bool_)
        if self._info_mode == "terminal":
            present = present & signals[:, 12].astype(bool)
        result: dict[str, np.ndarray] = {}
        for key in self._info_keys:
            index = _SIGNAL_NAMES.index(key)
            result[key] = signals[:, index]
            result[f"_{key}"] = present
        return result

    def reset(self, *, seed: int | Sequence[int | None] | None = None, options=None):
        del seed  # Reset selection is explicit; no hidden random reset distribution exists.
        options = {} if options is None else dict(options)
        mask = options.pop("reset_mask", None)
        if mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (self.num_envs,):
            raise TypeError(f"options['reset_mask'] must be a bool NumPy array with shape ({self.num_envs},)")
        if not np.any(mask):
            raise ValueError("options['reset_mask'] must select at least one lane")
        starts = options.pop("start_indices", None)
        start_ids = options.pop("start_ids", None)
        if starts is not None and start_ids is not None:
            raise ValueError("pass either start_indices or start_ids, not both")
        if start_ids is not None:
            values = np.asarray(start_ids, dtype=object)
            if values.shape != (self.num_envs,):
                raise ValueError(f"start_ids must have shape ({self.num_envs},)")
            lookup = {name: index for index, name in enumerate(_START_IDS)}
            starts = np.full(self.num_envs, -1, dtype=np.int32)
            for lane in np.flatnonzero(mask):
                value = values[lane]
                if value is not None:
                    try:
                        starts[lane] = lookup[str(value)]
                    except KeyError as exc:
                        raise ValueError(f"unknown start_id {value!r}") from exc
        elif starts is None:
            starts = np.full(self.num_envs, -1, dtype=np.int32)
        if not isinstance(starts, np.ndarray) or starts.dtype != np.int32 or starts.shape != (self.num_envs,):
            raise TypeError(f"start_indices must be an int32 NumPy array with shape ({self.num_envs},)")
        if options:
            raise ValueError(f"unsupported reset options: {sorted(options)}")
        observations, _, _, _, signals = self._next_buffers()
        self.native.reset_into(mask, starts, observations, signals)
        writable = self._active_state_indices.flags.writeable
        self._active_state_indices.setflags(write=True)
        self._active_state_indices[mask] = np.where(starts[mask] < 0, 0, starts[mask])
        self._active_state_indices.setflags(write=writable)
        self._last_signals = signals
        infos = self._infos(signals, mask.copy())
        start_names = np.asarray([_START_IDS[index] for index in self._active_state_indices], dtype=object)
        infos["start_id"] = start_names
        infos["_start_id"] = mask.copy()
        return self._obs(observations), infos

    def step(self, actions):
        values = np.asarray(actions, dtype=np.uint8)
        if values.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        observations, rewards, terminated, truncated, signals = self._next_buffers()
        self.native.step_into(
            values,
            observations,
            rewards,
            terminated,
            truncated,
            signals,
            self._info_mode != "none",
        )
        self._last_signals = signals
        return self._obs(observations), rewards, terminated, truncated, self._infos(signals)

    def active_state_indices(self) -> np.ndarray:
        return self._active_state_indices

    def active_states(self) -> tuple[str, ...]:
        return tuple(_START_IDS[index] for index in self._active_state_indices)

    def get_state(self) -> list[bytes]:
        return [bytes(value) for value in self.native.get_states()]

    def set_state(self, states: Sequence[bytes], reset_mask: np.ndarray | None = None) -> None:
        if reset_mask is None:
            reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        if not isinstance(reset_mask, np.ndarray) or reset_mask.dtype != np.bool_ or reset_mask.shape != (self.num_envs,):
            raise TypeError(f"reset_mask must be a bool NumPy array with shape ({self.num_envs},)")
        self.native.set_states(list(states), reset_mask)
        layout_ids = np.asarray(self.native.layout_ids(), dtype=np.int32)
        self._active_state_indices.setflags(write=True)
        self._active_state_indices[reset_mask] = layout_ids[reset_mask]
        self._active_state_indices.setflags(write=False)

    def configure_lane(self, lane: int, **state: int) -> None:
        ordered = ("paddle_x", "ball_x", "ball_y", "ball_vx", "ball_vy", "bricks", "lives")
        required = set(ordered)
        missing = required - state.keys()
        extra = state.keys() - required
        if missing or extra:
            raise ValueError(f"configure_lane requires {sorted(required)}")
        self.native.configure_lane(int(lane), *(int(state[name]) for name in ordered))

    def branch(self, states: Sequence[bytes], actions: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
        action_values = np.asarray(actions, dtype=np.uint8)
        next_states, flat_obs, rewards, terminated, flat_signals = self.native.branch(
            list(states), action_values.tolist()
        )
        count = len(states) * len(action_values)
        shape = self.single_observation_space.shape
        observations = np.frombuffer(flat_obs, dtype=np.uint8).copy().reshape((count, *shape))
        signals = np.asarray(flat_signals, dtype=np.int64).reshape((count, len(_SIGNAL_NAMES)))
        return {
            "next_states": [bytes(value) for value in next_states],
            "observations": observations,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.bool_),
            "signals": {name: signals[:, index] for index, name in enumerate(_SIGNAL_NAMES)},
            "source_index": np.repeat(np.arange(len(states)), len(action_values)),
            "actions": np.tile(action_values, len(states)),
        }

    def render(self):
        indexed = np.frombuffer(self.native.render_indexed(0), dtype=np.uint8).reshape(
            RENDER_HEIGHT, RENDER_WIDTH
        )
        return _ATARI_2600_NTSC_PALETTE[indexed]

    def close(self):
        self._closed = True


__all__ = [
    "BreakoutVecEnv",
    "FIXED_POINT_ONE",
    "RAW_HEIGHT",
    "RAW_WIDTH",
    "RENDER_HEIGHT",
    "RENDER_WIDTH",
]
