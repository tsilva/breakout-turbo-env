from __future__ import annotations

import hashlib

import numpy as np
import pytest
from breakout_turbo_env import BreakoutVecEnv


def _action_tape(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tape = rng.integers(0, 4, size=(length, 4), dtype=np.uint8)
    # Exercise FIRE and both paddle directions in every lane before relying on
    # the random suffix.
    tape[:4] = np.asarray(
        [
            [1, 1, 1, 1],
            [0, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 2, 1],
        ],
        dtype=np.uint8,
    )
    return tape


def _update_array(digest: hashlib._Hash, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(contiguous.dtype.str.encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    if contiguous.dtype.hasobject:
        digest.update(repr(contiguous.tolist()).encode())
    else:
        digest.update(contiguous.tobytes())


def _rollout_digest(env: BreakoutVecEnv, tape: np.ndarray) -> str:
    digest = hashlib.sha256()
    for step, actions in enumerate(tape):
        observations, rewards, terminated, truncated, infos = env.step(actions)
        for value in (observations, rewards, terminated, truncated):
            _update_array(digest, value)
        for key in sorted(infos):
            digest.update(key.encode())
            _update_array(digest, infos[key])
        for state in env.get_state():
            digest.update(state)
        if step % 17 == 0:
            for lane in range(env.num_envs):
                _update_array(digest, env.render_lane(lane))
        assert not truncated.any()
        if terminated.any():
            starts = np.asarray(env.active_state_indices(), dtype=np.int32).copy()
            reset_observations, reset_infos = env.reset(
                options={"reset_mask": terminated.copy(), "state_indices": starts}
            )
            _update_array(digest, reset_observations)
            for key in sorted(reset_infos):
                digest.update(key.encode())
                _update_array(digest, reset_infos[key])
            for state in env.get_state():
                digest.update(state)
    return digest.hexdigest()


@pytest.mark.parametrize(
    "configuration",
    [
        {
            "frame_skip": 1,
            "frame_stack": 1,
            "obs_resize": (37, 43),
            "obs_crop": (8, 4, 3, 5),
        },
        {
            "frame_skip": 4,
            "frame_stack": 4,
        },
        {
            "frame_skip": 3,
            "frame_stack": 5,
            "obs_resize": (47, 53),
            "obs_crop": (8, 4, 3, 5),
            "obs_crop_mode": "mask",
            "obs_crop_fill": 17,
        },
    ],
)
def test_serialized_and_live_snapshots_replay_the_complete_observable_trace(
    configuration,
):
    env = BreakoutVecEnv(num_envs=4, num_threads=2, **configuration)
    restored = BreakoutVecEnv(num_envs=4, num_threads=1, **configuration)
    try:
        starts = np.arange(4, dtype=np.int32)
        env.reset(options={"state_indices": starts})
        env.step(np.ones(4, dtype=np.uint8))
        for actions in _action_tape(53, 20260720):
            transition = env.step(actions)
            assert not transition[2].any()

        states = env.get_state()
        capture = env.capture_snapshots(np.ones(4, dtype=np.bool_))
        frames = [env.render_lane(lane) for lane in range(4)]
        future = _action_tape(80, 20260721)

        expected_digest = _rollout_digest(env, future)

        env.set_state(states)
        assert env.get_state() == states
        for lane, expected in enumerate(frames):
            np.testing.assert_array_equal(env.render_lane(lane), expected)
        assert _rollout_digest(env, future) == expected_digest

        env.reset(
            options={
                "reset_mask": np.ones(4, dtype=np.bool_),
                "state_indices": np.full(4, -1, dtype=np.int32),
                "snapshots": capture,
            }
        )
        assert env.get_state() == states
        for lane, expected in enumerate(frames):
            np.testing.assert_array_equal(env.render_lane(lane), expected)
        assert _rollout_digest(env, future) == expected_digest

        restored.reset()
        restored.set_state(states)
        assert restored.get_state() == states
        assert restored.active_states() == env.state_catalog
        for lane, expected in enumerate(frames):
            np.testing.assert_array_equal(restored.render_lane(lane), expected)
        assert _rollout_digest(restored, future) == expected_digest
    finally:
        env.close()
        restored.close()


def test_branch_results_are_exactly_the_same_as_real_environment_steps():
    source = BreakoutVecEnv(num_envs=4, num_threads=2, frame_skip=3, frame_stack=4)
    actual = BreakoutVecEnv(num_envs=8, num_threads=1, frame_skip=3, frame_stack=4)
    try:
        source.reset(options={"state_indices": np.arange(4, dtype=np.int32)})
        source.step(np.ones(4, dtype=np.uint8))
        for actions in _action_tape(37, 20260722):
            source.step(actions)

        states = source.get_state()[:2]
        actions = np.arange(4, dtype=np.uint8)
        branches = source.branch(states, actions)

        expanded_states = [
            state for state in states for _action in range(len(actions))
        ]
        actual.reset()
        actual.set_state(expanded_states)
        observations, rewards, terminated, _, infos = actual.step(
            np.tile(actions, len(states))
        )

        assert actual.get_state() == branches["next_states"]
        np.testing.assert_array_equal(observations, branches["observations"])
        np.testing.assert_array_equal(rewards, branches["rewards"])
        np.testing.assert_array_equal(terminated, branches["terminated"])
        for key, expected in branches["signals"].items():
            np.testing.assert_array_equal(infos[key], expected)
        assert source.get_state()[:2] == states
    finally:
        source.close()
        actual.close()


def test_rejected_cross_catalog_restore_is_atomic():
    source = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        state="checker",
        state_catalog=("checker",),
    )
    target = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        state="Start",
        state_catalog=("Start",),
    )
    try:
        source.reset()
        incompatible = source.get_state()
        target.reset()
        before = target.get_state()

        with pytest.raises(ValueError, match="absent from state_catalog"):
            target.set_state(incompatible)

        assert target.get_state() == before
    finally:
        source.close()
        target.close()
