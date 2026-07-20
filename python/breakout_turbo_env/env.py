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
    "brick_mask_high",
    "score",
    "lives",
    "tick",
    "bricks_remaining",
    "walls_cleared",
    "layout_id",
    "collision_events",
    "pending_reset",
)
# The native kernel retains its FIRE-wait flag as private simulation state.
# Public infos intentionally mirror the Atari cartridge contract, where
# ``ball_y == 0`` represents that state.
_NATIVE_SIGNAL_NAMES = (*_SIGNAL_NAMES, "_awaiting_fire")
_CANONICAL_GAME = "Breakout-Atari2600-v0"
_LEGACY_GAME = "BreakoutTurbo-v0"
_START_IDS = ("Start", "checker", "tunnel", "sparse")
_START_ALIASES = {"full": "Start"}
_RETRO_BUTTON_COUNT = 8
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


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).strip().lower()


def _normalize_game(game: str | None) -> str:
    value = _CANONICAL_GAME if game is None else str(game)
    if value not in {_CANONICAL_GAME, _LEGACY_GAME}:
        raise ValueError(
            f"game must be {_CANONICAL_GAME!r}; {_LEGACY_GAME!r} is retained as a legacy alias"
        )
    return _CANONICAL_GAME


def _canonical_start_id(value: Any) -> str:
    text = str(value)
    return _START_ALIASES.get(text, text)


def _uses_retro_actions(value: Any) -> bool:
    if value is None or _enum_name(value) in {"none", "native"}:
        return False
    if _enum_name(value) == "filtered" or value == 1:
        return True
    raise ValueError(
        "use_restricted_actions must be 'filtered' or omitted for native actions"
    )


def _require_fixed_option(name: str, value: Any, expected: Any) -> None:
    if value != expected:
        raise ValueError(
            f"{name} must be {expected!r} for Atari Breakout compatibility"
        )


def _validate_retro_compatibility_options(
    *,
    state: str | None,
    scenario: str | None,
    info: str | None,
    record: bool,
    players: int,
    inttype: Any,
    obs_type: Any,
    rom_path: str | None,
    noop_reset_max: int,
    use_fire_reset: bool,
    sticky_action_prob: float,
    reward_clip: bool,
) -> None:
    if scenario not in {None, "scenario"}:
        raise ValueError("scenario must be 'scenario' or None")
    if info not in {None, "data"}:
        raise ValueError("info must be 'data' or None; Atari signals are built in")
    _require_fixed_option("record", record, False)
    _require_fixed_option("players", players, 1)
    if _enum_name(inttype) not in {"stable", "1"}:
        raise ValueError("inttype must select the Stable integration")
    if _enum_name(obs_type) not in {"image", "0"}:
        raise ValueError("obs_type must be 'image'")
    _require_fixed_option("rom_path", rom_path, None)
    _require_fixed_option("noop_reset_max", noop_reset_max, 0)
    _require_fixed_option("use_fire_reset", use_fire_reset, False)
    _require_fixed_option("sticky_action_prob", float(sticky_action_prob), 0.0)
    _require_fixed_option("reward_clip", reward_clip, False)
    if state is not None and _canonical_start_id(state) not in _START_IDS:
        raise ValueError(f"unknown state {state!r}; expected one of {_START_IDS}")


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
    supports_live_snapshots = True

    def __init__(
        self,
        game: str | None = None,
        state: str | None = None,
        scenario: str | None = None,
        info: str | None = None,
        use_restricted_actions: Any = None,
        record: bool = False,
        players: int = 1,
        inttype: Any = "stable",
        obs_type: Any = "image",
        render_mode: str = "rgb_array",
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | None = None,
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
        noop_reset_max: int = 0,
        use_fire_reset: bool = False,
        sticky_action_prob: float = 0.0,
        reward_clip: bool = False,
        state_catalog: Sequence[str] | None = None,
        **unsupported: Any,
    ):
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported option(s): {names}")
        self.game = _normalize_game(game)
        self._retro_actions = _uses_retro_actions(use_restricted_actions)
        _validate_retro_compatibility_options(
            state=state,
            scenario=scenario,
            info=info,
            record=record,
            players=players,
            inttype=inttype,
            obs_type=obs_type,
            rom_path=rom_path,
            noop_reset_max=noop_reset_max,
            use_fire_reset=use_fire_reset,
            sticky_action_prob=sticky_action_prob,
            reward_clip=reward_clip,
        )
        requested_state = _canonical_start_id(state or "Start")
        if requested_state not in _START_IDS:
            raise ValueError(f"unknown state {state!r}; expected one of {_START_IDS}")
        configured_catalog = (
            _START_IDS
            if state_catalog is None
            else tuple(_canonical_start_id(value) for value in state_catalog)
        )
        unknown_states = sorted(set(configured_catalog) - set(_START_IDS))
        if unknown_states:
            raise ValueError(f"state_catalog contains unknown states: {unknown_states}")
        if not configured_catalog:
            raise ValueError("state_catalog must not be empty")
        if len(set(configured_catalog)) != len(configured_catalog):
            raise ValueError("state_catalog must contain unique states")
        if requested_state not in configured_catalog:
            raise ValueError("state must be present in state_catalog")
        self._default_start_index = configured_catalog.index(requested_state)
        self._catalog_to_engine = np.asarray(
            [_START_IDS.index(value) for value in configured_catalog],
            dtype=np.int32,
        )
        if maxpool_last_two:
            raise ValueError("maxpool_last_two is not implemented and must be False")
        if str(obs_layout).lower() != "chw":
            raise ValueError(
                "obs_layout is fixed to 'chw' for the rlab policy contract"
            )
        if not obs_grayscale:
            raise ValueError(
                "obs_grayscale is fixed to True for the rlab policy contract"
            )
        if obs_resize_algorithm != "area":
            raise ValueError(
                "obs_resize_algorithm is fixed to 'area' for the rlab policy contract"
            )
        if obs_crop_mode not in {"remove", "mask"}:
            raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
        if (
            not isinstance(obs_crop_fill, int)
            or isinstance(obs_crop_fill, bool)
            or not 0 <= obs_crop_fill <= 255
        ):
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
        crop = (
            (0, 0, 0, 0)
            if obs_crop is None
            else tuple(int(value) for value in obs_crop)
        )
        if len(crop) != 4 or min(crop) < 0:
            raise ValueError(
                "obs_crop must contain non-negative (top, bottom, left, right)"
            )
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
        self.state_catalog = configured_catalog
        self.initial_state_names = configured_catalog
        if self._retro_actions:
            self.single_action_space = gym.spaces.MultiBinary(_RETRO_BUTTON_COUNT)
            self.action_space = gym.spaces.Box(
                0,
                1,
                shape=(num_envs, _RETRO_BUTTON_COUNT),
                dtype=np.int8,
            )
        else:
            self.single_action_space = gym.spaces.Discrete(4)
            self.action_space = gym.spaces.MultiDiscrete(
                np.full(num_envs, 4, dtype=np.int64)
            )
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
        self._reward_buffers = [
            np.empty(num_envs, dtype=np.float32) for _ in range(count)
        ]
        self._terminated_buffers = [
            np.empty(num_envs, dtype=np.bool_) for _ in range(count)
        ]
        self._truncated_buffers = [
            np.empty(num_envs, dtype=np.bool_) for _ in range(count)
        ]
        self._signal_buffers = [
            np.empty((num_envs, len(_NATIVE_SIGNAL_NAMES)), dtype=np.int64)
            for _ in range(count)
        ]
        self._buffer_index = 0
        self._active_state_indices = np.zeros(num_envs, dtype=np.int32)
        self._active_state_indices.setflags(write=False)
        self._last_signals = self._signal_buffers[0]
        self._initialized = np.zeros(num_envs, dtype=np.bool_)
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

    def _infos(
        self, signals: np.ndarray, present: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        if self._info_mode == "none":
            return {}
        if present is None:
            present = np.ones(self.num_envs, dtype=np.bool_)
        if self._info_mode == "terminal":
            present = present & signals[
                :, _NATIVE_SIGNAL_NAMES.index("pending_reset")
            ].astype(bool)
        result: dict[str, np.ndarray] = {}
        for key in self._info_keys:
            index = _NATIVE_SIGNAL_NAMES.index(key)
            result[key] = signals[:, index]
            result[f"_{key}"] = present
        return result

    def reset(self, *, seed: int | Sequence[int | None] | None = None, options=None):
        if self._closed:
            raise RuntimeError("cannot reset a closed environment")
        options = {} if options is None else dict(options)
        mask = options.pop("reset_mask", None)
        if mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        if not isinstance(mask, np.ndarray):
            raise TypeError("options['reset_mask'] must be a NumPy array")
        if mask.shape != (self.num_envs,):
            raise ValueError(
                f"options['reset_mask'] must have shape ({self.num_envs},)"
            )
        if mask.dtype != np.bool_:
            raise TypeError("options['reset_mask'] must have dtype np.bool_")
        if not np.any(mask):
            raise ValueError("options['reset_mask'] must select at least one lane")
        snapshots = options.pop("snapshots", None)
        if snapshots is None:
            snapshot_values: list[Any | None] = [None] * self.num_envs
        else:
            if isinstance(snapshots, (str, bytes, bytearray)) or not isinstance(
                snapshots, Sequence
            ):
                raise TypeError("options['snapshots'] must be a lane-aligned sequence")
            if len(snapshots) != self.num_envs:
                raise ValueError(
                    f"options['snapshots'] must have length {self.num_envs}"
                )
            snapshot_values = list(snapshots)
        snapshot_mask = np.asarray(
            [value is not None for value in snapshot_values], dtype=np.bool_
        )
        if np.any(snapshot_mask & ~mask):
            raise ValueError("snapshots may only be supplied for selected reset lanes")
        if np.any(snapshot_mask):
            if seed is not None and np.isscalar(seed):
                raise ValueError("snapshot reset lanes cannot also specify a seed")
            if seed is not None:
                if isinstance(seed, (str, bytes, bytearray)) or not isinstance(
                    seed, Sequence
                ):
                    raise TypeError("seed must be an integer or a lane-aligned sequence")
                if len(seed) != self.num_envs:
                    raise ValueError(f"seed must have length {self.num_envs}")
                if any(seed[lane] is not None for lane in np.flatnonzero(snapshot_mask)):
                    raise ValueError("snapshot reset lanes cannot also specify a seed")
        # Static reset seeds are accepted for Gymnasium compatibility. This
        # deterministic provider has no random reset distribution.
        starts = options.pop("start_indices", None)
        state_indices = options.pop("state_indices", None)
        if starts is not None and state_indices is not None:
            raise ValueError("pass either start_indices or state_indices, not both")
        if state_indices is not None:
            starts = state_indices
        start_ids = options.pop("start_ids", None)
        if starts is not None and start_ids is not None:
            raise ValueError("pass either start_indices or start_ids, not both")
        if start_ids is not None:
            values = np.asarray(start_ids, dtype=object)
            if values.shape != (self.num_envs,):
                raise ValueError(f"start_ids must have shape ({self.num_envs},)")
            if any(values[lane] is not None for lane in np.flatnonzero(snapshot_mask)):
                raise ValueError(
                    "snapshot reset lanes must use None for the static start selector"
                )
            lookup = {name: index for index, name in enumerate(self.state_catalog)}
            lookup.update(
                {
                    alias: lookup[canonical]
                    for alias, canonical in _START_ALIASES.items()
                    if canonical in lookup
                }
            )
            starts = np.full(self.num_envs, -1, dtype=np.int32)
            for lane in np.flatnonzero(mask & ~snapshot_mask):
                value = values[lane]
                if value is not None:
                    try:
                        starts[lane] = lookup[str(value)]
                    except KeyError as exc:
                        raise ValueError(f"unknown start_id {value!r}") from exc
        elif starts is None:
            starts = np.full(self.num_envs, self._default_start_index, dtype=np.int32)
            starts[snapshot_mask] = -1
        if not isinstance(starts, np.ndarray):
            raise TypeError("start_indices must be a NumPy array")
        if starts.shape != (self.num_envs,):
            raise ValueError(f"start_indices must have shape ({self.num_envs},)")
        if starts.dtype != np.int32:
            raise TypeError("start_indices must have dtype np.int32")
        if np.any(starts[snapshot_mask] != -1):
            raise ValueError(
                "snapshot reset lanes must use -1 for the static start selector"
            )
        static_mask = mask & ~snapshot_mask
        selected_starts = starts[static_mask]
        if np.any((selected_starts < 0) | (selected_starts >= len(self.state_catalog))):
            raise ValueError(
                f"selected start indices must be in [0, {len(self.state_catalog) - 1}]"
            )
        if options:
            raise ValueError(f"unsupported reset options: {sorted(options)}")
        observations, rewards, terminated, truncated, signals = self._next_buffers()
        engine_starts = np.full(self.num_envs, -1, dtype=np.int32)
        engine_starts[static_mask] = self._catalog_to_engine[starts[static_mask]]
        if snapshots is None:
            self.native.reset_into(mask, engine_starts, observations, signals)
        else:
            self.native.reset_mixed_into(
                mask, engine_starts, snapshot_values, observations, signals
            )
        rewards[mask] = 0.0
        terminated[mask] = False
        truncated[mask] = False
        writable = self._active_state_indices.flags.writeable
        self._active_state_indices.setflags(write=True)
        self._active_state_indices[static_mask] = starts[static_mask]
        if np.any(snapshot_mask):
            engine_to_catalog = {
                int(engine_index): catalog_index
                for catalog_index, engine_index in enumerate(self._catalog_to_engine)
            }
            layout_ids = self.native.layout_ids()
            restored_indices = np.asarray(
                [engine_to_catalog.get(int(layout_id), -1) for layout_id in layout_ids],
                dtype=np.int32,
            )
            if np.any(restored_indices[snapshot_mask] < 0):
                raise ValueError("snapshot layout is absent from state_catalog")
            self._active_state_indices[snapshot_mask] = restored_indices[snapshot_mask]
        self._active_state_indices.setflags(write=writable)
        self._initialized[mask] = True
        self._last_signals = signals
        infos = self._infos(signals, mask.copy())
        start_names = np.asarray(
            [self.state_catalog[index] for index in self._active_state_indices],
            dtype=object,
        )
        infos["start_id"] = start_names
        infos["_start_id"] = mask.copy()
        infos["state"] = start_names
        infos["_state"] = mask.copy()
        infos["start_state"] = start_names
        infos["_start_state"] = mask.copy()
        infos["state_index"] = self._active_state_indices.copy()
        infos["_state_index"] = mask.copy()
        start_source = np.full(self.num_envs, "environment", dtype=object)
        start_source[snapshot_mask] = "snapshot"
        infos["start_source"] = start_source
        infos["_start_source"] = mask.copy()
        return self._obs(observations), infos

    def step(self, actions):
        if self._closed:
            raise RuntimeError("cannot step a closed environment")
        if not np.all(self._initialized):
            raise RuntimeError("all lanes must be reset before the first step")
        values = self._native_actions(actions)
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
        return (
            self._obs(observations),
            rewards,
            terminated,
            truncated,
            self._infos(signals),
        )

    def _native_actions(self, actions: Any) -> np.ndarray:
        if not self._retro_actions:
            return np.asarray(actions, dtype=np.uint8)
        buttons = np.asarray(actions, dtype=np.int8)
        expected_shape = (self.num_envs, _RETRO_BUTTON_COUNT)
        if buttons.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}")
        if np.any((buttons != 0) & (buttons != 1)):
            raise ValueError("Stable Retro-compatible actions must contain only 0 or 1")
        native = np.zeros(self.num_envs, dtype=np.uint8)
        fire = buttons[:, 0] != 0
        left = buttons[:, 6] != 0
        right = buttons[:, 7] != 0
        native[right & ~left] = 2
        native[left & ~right] = 3
        native[fire] = 1
        return native

    def active_state_indices(self) -> np.ndarray:
        return self._active_state_indices

    def active_states(self) -> tuple[str, ...]:
        return tuple(self.state_catalog[index] for index in self._active_state_indices)

    def get_state(self) -> list[bytes]:
        if self._closed:
            raise RuntimeError("cannot read state from a closed environment")
        return [bytes(value) for value in self.native.get_states()]

    def capture_snapshots(self, mask: np.ndarray) -> tuple[Any | None, ...]:
        if self._closed:
            raise RuntimeError("cannot capture snapshots from a closed environment")
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array")
        if mask.shape != (self.num_envs,):
            raise ValueError(f"mask must have shape ({self.num_envs},)")
        if mask.dtype != np.bool_:
            raise TypeError("mask must have dtype np.bool_")
        if not np.any(mask):
            raise ValueError("mask must select at least one lane")
        if not np.all(self._initialized[mask]):
            raise RuntimeError("cannot capture a lane before its initial reset")
        return tuple(self.native.capture_snapshots(mask))

    def set_state(
        self, states: Sequence[bytes], reset_mask: np.ndarray | None = None
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot restore state into a closed environment")
        if reset_mask is None:
            reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        if (
            not isinstance(reset_mask, np.ndarray)
            or reset_mask.dtype != np.bool_
            or reset_mask.shape != (self.num_envs,)
        ):
            raise TypeError(
                f"reset_mask must be a bool NumPy array with shape ({self.num_envs},)"
            )
        observations, rewards, terminated, truncated, signals = self._next_buffers()
        self.native.set_states_into(
            list(states), reset_mask, observations, signals
        )
        rewards[reset_mask] = 0.0
        terminated[reset_mask] = False
        truncated[reset_mask] = False
        self._last_signals = signals
        self._initialized[reset_mask] = True
        layout_ids = np.asarray(self.native.layout_ids(), dtype=np.int32)
        engine_to_catalog = {
            int(engine_index): catalog_index
            for catalog_index, engine_index in enumerate(self._catalog_to_engine)
        }
        restored_indices = np.asarray(
            [engine_to_catalog.get(int(layout_id), -1) for layout_id in layout_ids],
            dtype=np.int32,
        )
        if np.any(restored_indices[reset_mask] < 0):
            raise ValueError("restored state layout is absent from state_catalog")
        self._active_state_indices.setflags(write=True)
        self._active_state_indices[reset_mask] = restored_indices[reset_mask]
        self._active_state_indices.setflags(write=False)

    def configure_lane(self, lane: int, **state: int) -> None:
        ordered = (
            "paddle_x",
            "ball_x",
            "ball_y",
            "ball_vx",
            "ball_vy",
            "bricks",
            "lives",
        )
        required = set(ordered)
        missing = required - state.keys()
        extra = state.keys() - required
        if missing or extra:
            raise ValueError(f"configure_lane requires {sorted(required)}")
        self.native.configure_lane(int(lane), *(int(state[name]) for name in ordered))

    def branch(
        self, states: Sequence[bytes], actions: Sequence[int] = (0, 1, 2, 3)
    ) -> dict[str, Any]:
        action_values = np.asarray(actions, dtype=np.uint8)
        next_states, flat_obs, rewards, terminated, flat_signals = self.native.branch(
            list(states), action_values.tolist()
        )
        count = len(states) * len(action_values)
        shape = self.single_observation_space.shape
        observations = (
            np.frombuffer(flat_obs, dtype=np.uint8).copy().reshape((count, *shape))
        )
        signals = np.asarray(flat_signals, dtype=np.int64).reshape(
            (count, len(_NATIVE_SIGNAL_NAMES))
        )
        return {
            "next_states": [bytes(value) for value in next_states],
            "observations": observations,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.bool_),
            "signals": {
                name: signals[:, index] for index, name in enumerate(_SIGNAL_NAMES)
            },
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
