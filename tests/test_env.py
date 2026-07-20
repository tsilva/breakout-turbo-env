from __future__ import annotations

import hashlib
import inspect
import pickle

import gymnasium as gym
import numpy as np
import pytest
from breakout_turbo_env import (
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    BreakoutVecEnv,
)
from gymnasium.vector import AutoresetMode


def make_env(**kwargs):
    return BreakoutVecEnv(num_envs=4, num_threads=2, **kwargs)


def test_registered_vector_entry_point_matches_declared_spaces():
    env = gym.make_vec("BreakoutTurbo-v0", num_envs=4, num_threads=2)
    try:
        observations, infos = env.reset(seed=7)
        assert env.observation_space.contains(observations)
        actions = np.array([0, 1, 2, 3], dtype=np.uint8)
        transition = env.step(actions)
        assert env.observation_space.contains(transition[0])
        assert transition[1].shape == (4,)
        assert transition[2].shape == (4,)
        assert transition[3].shape == (4,)
        assert isinstance(infos, dict)
        assert isinstance(transition[4], dict)
    finally:
        env.close()


def test_stable_retro_breakout_contract_is_a_drop_in_provider_surface():
    env = BreakoutVecEnv(
        "Breakout-Atari2600-v0",
        state="Start",
        scenario="scenario",
        info="data",
        use_restricted_actions="filtered",
        record=False,
        players=1,
        inttype="stable",
        obs_type="image",
        render_mode="rgb_array",
        num_envs=4,
        num_threads=2,
        rom_path=None,
        obs_resize=(84, 84),
        obs_crop=(17, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=False,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="all",
        obs_copy="safe_view",
    )
    try:
        assert env.game == "Breakout-Atari2600-v0"
        assert env.state_catalog[0] == "Start"
        assert env.single_action_space == gym.spaces.MultiBinary(8)
        observations, infos = env.reset()
        assert observations.shape == (4, 4, 84, 84)
        assert infos["start_id"].tolist() == ["Start"] * 4
        actions = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.int8,
        )
        transition = env.step(actions)
        assert env.observation_space.contains(transition[0])
    finally:
        env.close()


def test_stable_retro_button_rows_match_native_actions():
    native = BreakoutVecEnv(num_envs=4, num_threads=1, frame_skip=1)
    compatible = BreakoutVecEnv(
        "Breakout-Atari2600-v0",
        use_restricted_actions="filtered",
        num_envs=4,
        num_threads=1,
        frame_skip=1,
    )
    try:
        native.reset()
        compatible.reset()
        button_actions = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.int8,
        )
        expected = native.step(np.asarray([0, 1, 2, 3], dtype=np.uint8))
        actual = compatible.step(button_actions)
        for expected_value, actual_value in zip(expected[:4], actual[:4], strict=True):
            np.testing.assert_array_equal(actual_value, expected_value)
    finally:
        native.close()
        compatible.close()


def test_canonical_registered_id_uses_stable_retro_action_space():
    env = gym.make_vec("Breakout-Atari2600-v0", num_envs=2, num_threads=1)
    try:
        observations, infos = env.reset()
        assert observations.shape == (2, 4, 84, 84)
        assert infos["start_id"].tolist() == ["Start", "Start"]
        assert env.single_action_space == gym.spaces.MultiBinary(8)
        env.step(np.zeros((2, 8), dtype=np.int8))
    finally:
        env.close()


def test_state_catalog_uses_stable_retro_catalog_indices():
    env = BreakoutVecEnv(
        "Breakout-Atari2600-v0",
        state="checker",
        state_catalog=("checker", "Start"),
        num_envs=2,
        num_threads=1,
    )
    try:
        _, infos = env.reset(
            options={"state_indices": np.asarray([0, 1], dtype=np.int32)}
        )
        assert env.state_catalog == ("checker", "Start")
        assert infos["state"].tolist() == ["checker", "Start"]
        assert infos["state_index"].tolist() == [0, 1]
    finally:
        env.close()


def test_reset_seed_is_accepted_but_does_not_change_deterministic_start():
    env = make_env(frame_skip=1)
    try:
        first_obs, _ = env.reset(seed=1)
        first_state = env.get_state()
        second_obs, _ = env.reset(seed=999)
        assert env.get_state() == first_state
        np.testing.assert_array_equal(second_obs, first_obs)
    finally:
        env.close()


def test_full_wall_info_preserves_all_108_brick_bits_and_wall_progress():
    env = make_env(frame_skip=1)
    try:
        _, info = env.reset()
        low = int(info["brick_mask"][0]) & ((1 << 64) - 1)
        high = int(info["brick_mask_high"][0])
        assert low | (high << 64) == (1 << 108) - 1
        assert info["walls_cleared"][0] == 0
    finally:
        env.close()


def test_contract_is_chw_manual_and_no_maxpool():
    env = make_env()
    obs, infos = env.reset()
    assert obs.shape == (4, 4, 84, 84)
    assert obs.dtype == np.uint8
    assert env.autoreset_mode is AutoresetMode.DISABLED
    assert infos["_start_id"].all()
    assert BreakoutVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    assert "autoreset_mode" not in inspect.signature(BreakoutVecEnv).parameters
    with pytest.raises(TypeError, match="unsupported option.*autoreset_mode"):
        make_env(autoreset_mode=AutoresetMode.DISABLED)
    with pytest.raises(ValueError, match="maxpool"):
        make_env(maxpool_last_two=True)
    with pytest.raises(ValueError, match="chw"):
        make_env(obs_layout="hwc")


def test_masked_reset_preserves_unselected_lane_exactly():
    env = make_env()
    env.reset()
    env.step(np.array([1, 2, 1, 2], dtype=np.uint8))
    before = env.get_state()
    mask = np.array([True, False, True, False], dtype=np.bool_)
    env.reset(options={"reset_mask": mask})
    after = env.get_state()
    assert before[1] == after[1]
    assert before[3] == after[3]
    assert before[0] != after[0]
    assert before[2] != after[2]


def test_info_presence_masks_follow_the_configured_filter():
    all_info = make_env(info_filter="all")
    all_info.reset()
    _, _, terminated, _, infos = all_info.step(np.zeros(4, dtype=np.uint8))
    assert not terminated.any()
    assert infos["_bricks_remaining"].all()
    assert infos["_lives"].all()

    terminal_info = make_env(info_filter="terminal")
    terminal_info.reset()
    _, _, terminated, _, infos = terminal_info.step(np.zeros(4, dtype=np.uint8))
    assert not terminated.any()
    assert not infos["_bricks_remaining"].any()
    assert not infos["_lives"].any()


def test_snapshot_replay_is_byte_exact():
    env = make_env(frame_skip=1)
    env.reset()
    snapshot = env.get_state()
    actions = np.array([0, 1, 2, 0], dtype=np.uint8)
    first = env.step(actions)
    first_states = env.get_state()
    env.set_state(snapshot)
    second = env.step(actions)
    assert env.get_state() == first_states
    for left, right in zip(first[:4], second[:4], strict=True):
        np.testing.assert_array_equal(left, right)


def test_live_snapshots_support_masked_capture_cross_lane_fanout_and_replay():
    env = make_env(frame_skip=1)
    try:
        env.reset(
            options={"state_indices": np.asarray([0, 1, 2, 3], dtype=np.int32)}
        )
        env.step(np.asarray([1, 2, 3, 0], dtype=np.uint8))
        captured_states = env.get_state()
        handles = env.capture_snapshots(
            np.asarray([True, False, False, False], dtype=np.bool_)
        )
        assert handles[0] is not None
        assert handles[0].nbytes > 0
        assert handles[1:] == (None, None, None)
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(handles[0])

        env.step(np.asarray([2, 2, 2, 2], dtype=np.uint8))
        unselected_before = env.get_state()[3]
        mask = np.asarray([True, True, True, False], dtype=np.bool_)
        starts = np.asarray([-1, 3, -1, -1], dtype=np.int32)
        restored_obs, infos = env.reset(
            options={
                "reset_mask": mask,
                "state_indices": starts,
                "snapshots": [handles[0], None, handles[0], None],
            }
        )
        restored_states = env.get_state()
        assert restored_states[0] == captured_states[0]
        assert restored_states[2] == captured_states[0]
        assert restored_states[3] == unselected_before
        np.testing.assert_array_equal(restored_obs[0], restored_obs[2])
        assert infos["start_source"].tolist() == [
            "snapshot",
            "environment",
            "snapshot",
            "environment",
        ]
        np.testing.assert_array_equal(infos["_start_source"], mask)

        actions = np.asarray([3, 0, 3, 0], dtype=np.uint8)
        first = env.step(actions)
        env.reset(
            options={
                "reset_mask": mask,
                "state_indices": starts,
                "snapshots": [handles[0], None, handles[0], None],
            }
        )
        second = env.step(actions)
        for first_value, second_value in zip(first[:4], second[:4], strict=True):
            np.testing.assert_array_equal(first_value[mask], second_value[mask])
    finally:
        env.close()


def test_live_snapshot_lifecycle_owner_and_selector_validation_are_atomic():
    env = make_env(frame_skip=1)
    mask = np.asarray([True, False, False, False], dtype=np.bool_)
    with pytest.raises(RuntimeError, match="initial reset"):
        env.capture_snapshots(mask)
    env.reset()
    handles = env.capture_snapshots(mask)
    before = env.get_state()

    with pytest.raises(ValueError, match="static start selector"):
        env.reset(
            options={
                "reset_mask": mask,
                "state_indices": np.asarray([0, -1, -1, -1], dtype=np.int32),
                "snapshots": handles,
            }
        )
    assert env.get_state() == before

    other = make_env(frame_skip=1)
    try:
        other.reset()
        other_before = other.get_state()
        with pytest.raises(ValueError, match="different environment"):
            other.reset(
                options={
                    "reset_mask": mask,
                    "state_indices": np.full(4, -1, dtype=np.int32),
                    "snapshots": handles,
                }
            )
        assert other.get_state() == other_before
    finally:
        other.close()

    env.close()
    with pytest.raises(RuntimeError, match="closed environment"):
        env.capture_snapshots(mask)


def test_live_snapshot_mask_validation_uses_consistent_error_categories():
    env = BreakoutVecEnv(num_envs=2)
    try:
        env.reset()
        with pytest.raises(TypeError, match="NumPy array"):
            env.capture_snapshots([True, False])
        with pytest.raises(ValueError, match="shape"):
            env.capture_snapshots(np.array([True], dtype=np.bool_))
        with pytest.raises(TypeError, match="dtype"):
            env.capture_snapshots(np.array([1, 0], dtype=np.uint8))
    finally:
        env.close()


def test_branches_cover_all_actions_without_mutating_source():
    env = make_env(frame_skip=1)
    env.reset()
    states = env.get_state()[:2]
    before = env.get_state()
    result = env.branch(states)
    assert result["observations"].shape == (8, 4, 84, 84)
    np.testing.assert_array_equal(result["actions"], [0, 1, 2, 3, 0, 1, 2, 3])
    assert env.get_state() == before


def test_start_catalog_and_atomic_validation():
    env = make_env()
    env.reset()
    before = env.get_state()
    mask = np.array([True, False, False, False], dtype=np.bool_)
    starts = np.array([99, -1, -1, -1], dtype=np.int32)
    with pytest.raises(ValueError):
        env.reset(options={"reset_mask": mask, "start_indices": starts})
    assert env.get_state() == before


def test_crop_modes_preserve_chw_shape_and_change_pixels():
    removed = make_env(obs_crop=(8, 0, 0, 0), obs_crop_mode="remove")
    masked = make_env(obs_crop=(8, 0, 0, 0), obs_crop_mode="mask", obs_crop_fill=17)
    removed_obs, _ = removed.reset()
    masked_obs, _ = masked.reset()
    assert removed_obs.shape == masked_obs.shape == (4, 4, 84, 84)
    assert not np.array_equal(removed_obs, masked_obs)


def test_all_layouts_start_hidden_and_fire_uses_the_atari_serve():
    env = make_env(frame_skip=1)
    starts = np.arange(4, dtype=np.int32)
    _, info = env.reset(options={"start_indices": starts})
    assert RAW_HEIGHT == 210
    assert np.all(info["ball_y"] == 0)
    assert np.all(info["lives"] == 5)
    assert "awaiting_fire" not in info
    assert len(set(info["ball_vx"].tolist())) == 1
    assert len(set(info["ball_vy"].tolist())) == 1

    _, _, _, _, info = env.step(np.ones(4, dtype=np.uint8))
    assert np.all(info["ball_x"] == 80 * FIXED_POINT_ONE)
    assert np.all(info["ball_y"] == 113)
    assert np.all(info["ball_vx"] == FIXED_POINT_ONE)
    assert np.all(info["ball_vy"] == FIXED_POINT_ONE)


def test_render_matches_atari_2600_geometry_and_palette():
    env = make_env(frame_skip=1)
    env.reset()
    frame = env.render()
    assert frame.shape == (RENDER_HEIGHT, RENDER_WIDTH, 3) == (210, 160, 3)
    assert frame.dtype == np.uint8

    black = np.array([0, 0, 0], dtype=np.uint8)
    gray = np.array([136, 136, 136], dtype=np.uint8)
    row_colors = np.array(
        [
            [200, 72, 72],
            [192, 104, 56],
            [176, 120, 48],
            [160, 160, 40],
            [72, 160, 72],
            [64, 72, 200],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(
        frame[17:32], np.broadcast_to(gray, frame[17:32].shape)
    )
    np.testing.assert_array_equal(
        frame[32:57, :8], np.broadcast_to(gray, frame[32:57, :8].shape)
    )
    np.testing.assert_array_equal(
        frame[32:57, 8:152], np.broadcast_to(black, frame[32:57, 8:152].shape)
    )
    # Stable Retro's first rendered frame has already cleared the first brick
    # as part of the ROM's startup animation. Every other brick is intact.
    np.testing.assert_array_equal(frame[57:63, 8:16], np.broadcast_to(black, (6, 8, 3)))
    for row, color in enumerate(row_colors):
        band = frame[57 + row * 6 : 63 + row * 6, 8:152]
        start = 8 if row == 0 else 0
        np.testing.assert_array_equal(
            band[:, start:], np.broadcast_to(color, band[:, start:].shape)
        )
    five = np.zeros((10, 12, 3), dtype=np.uint8)
    five[0:2, :] = gray
    five[2:4, 0:4] = gray
    five[4:6, :] = gray
    five[6:8, 8:12] = gray
    five[8:10, :] = gray
    np.testing.assert_array_equal(frame[5:15, 100:112], five)
    teal = np.array([64, 152, 128], dtype=np.uint8)
    red = np.array([200, 72, 72], dtype=np.uint8)
    np.testing.assert_array_equal(frame[189:196, :8], np.broadcast_to(teal, (7, 8, 3)))
    np.testing.assert_array_equal(frame[189:195, 152:], np.broadcast_to(red, (6, 8, 3)))


def test_render_uses_exact_atari_ball_and_paddle_footprints_at_motion_limits():
    env = make_env(frame_skip=1)
    env.reset()
    red = np.array([200, 72, 72], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)

    common = {
        "ball_y": 120 * FIXED_POINT_ONE,
        "ball_vx": FIXED_POINT_ONE,
        "ball_vy": FIXED_POINT_ONE,
        "bricks": 1,
        "lives": 5,
    }
    cases = (
        (8, 8),
        (76, 80),
        (144, 150),
    )
    for paddle_x, ball_x in cases:
        env.configure_lane(
            0,
            paddle_x=paddle_x * FIXED_POINT_ONE,
            ball_x=ball_x * FIXED_POINT_ONE,
            **common,
        )
        frame = env.render()

        # The active Atari sprite is 16x4. Its right half deliberately merges
        # with the permanent red cap at the rightmost travel limit.
        np.testing.assert_array_equal(
            frame[189:193, paddle_x : paddle_x + 16],
            np.broadcast_to(red, (4, 16, 3)),
        )

        # The native Atari ball is exactly 2x4 (eight pixels), including both
        # horizontal motion limits.
        np.testing.assert_array_equal(
            frame[120:124, ball_x : ball_x + 2],
            np.broadcast_to(red, (4, 2, 3)),
        )
        if ball_x > 8:
            np.testing.assert_array_equal(
                frame[120:124, ball_x - 1],
                np.broadcast_to(black, (4, 3)),
            )
        if ball_x + 2 < 152:
            np.testing.assert_array_equal(
                frame[120:124, ball_x + 2],
                np.broadcast_to(black, (4, 3)),
            )


def test_render_reflects_missing_bricks_and_lane_status():
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=80 * FIXED_POINT_ONE,
        ball_y=120 * FIXED_POINT_ONE,
        ball_vx=FIXED_POINT_ONE,
        ball_vy=FIXED_POINT_ONE,
        bricks=((1 << 108) - 1) ^ (1 << 2),
        lives=5,
    )
    frame = env.render()
    black = np.array([0, 0, 0], dtype=np.uint8)
    red = np.array([200, 72, 72], dtype=np.uint8)
    np.testing.assert_array_equal(
        frame[57:63, 8:24], np.broadcast_to(red, frame[57:63, 8:24].shape)
    )
    np.testing.assert_array_equal(
        frame[57:63, 24:32], np.broadcast_to(black, frame[57:63, 24:32].shape)
    )
    assert np.any(frame[5:15, 36:80])
    assert np.any(frame[189:193, 8:152] == red)


def test_terminal_lane_requires_explicit_reset_then_can_continue():
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=10 * FIXED_POINT_ONE,
        ball_y=217 * FIXED_POINT_ONE,
        ball_vx=FIXED_POINT_ONE,
        ball_vy=FIXED_POINT_ONE,
        bricks=1 | (1 << 47),
        lives=1,
    )
    _, _, terminated, _, _ = env.step(np.zeros(4, dtype=np.uint8))
    assert terminated.tolist() == [True, False, False, False]
    with pytest.raises(RuntimeError, match="pending reset"):
        env.step(np.zeros(4, dtype=np.uint8))
    env.reset(options={"reset_mask": terminated})
    env.step(np.zeros(4, dtype=np.uint8))


def test_frame_skip_matches_repeated_native_physics():
    skipped = make_env(frame_skip=4)
    repeated = make_env(frame_skip=1)
    skipped.reset()
    repeated.reset()
    actions = np.array([0, 1, 2, 0], dtype=np.uint8)
    _, skipped_reward, _, _, skipped_info = skipped.step(actions)
    repeated_reward = np.zeros(4, dtype=np.float32)
    for _ in range(4):
        _, reward, _, _, repeated_info = repeated.step(actions)
        repeated_reward += reward
    np.testing.assert_array_equal(skipped_reward, repeated_reward)
    for key in (
        "paddle_x",
        "ball_x",
        "ball_y",
        "ball_vx",
        "ball_vy",
        "brick_mask",
        "tick",
    ):
        np.testing.assert_array_equal(skipped_info[key], repeated_info[key])


def test_thread_count_does_not_change_trace():
    serial = BreakoutVecEnv(num_envs=16, num_threads=1)
    parallel = BreakoutVecEnv(num_envs=16, num_threads=8)
    serial.reset()
    parallel.reset()
    rng = np.random.default_rng(1234)
    for _ in range(20):
        actions = rng.integers(0, 4, size=16, dtype=np.uint8)
        serial.step(actions)
        parallel.step(actions)
    assert serial.get_state() == parallel.get_state()


@pytest.mark.parametrize(
    "preprocessing",
    [
        {},
        {"obs_resize": (80, 96), "obs_crop": (8, 4, 3, 5)},
        {
            "obs_resize": (47, 53),
            "obs_crop": (8, 4, 3, 5),
            "obs_crop_mode": "mask",
            "obs_crop_fill": 17,
        },
    ],
)
def test_incremental_observations_match_forced_full_rebuild(preprocessing):
    incremental = make_env(info_filter="none", **preprocessing)
    rebuilt = make_env(info_filter="none", **preprocessing)
    starts = np.arange(4, dtype=np.int32)
    first, _ = incremental.reset(options={"start_indices": starts})
    second, _ = rebuilt.reset(options={"start_indices": starts})
    np.testing.assert_array_equal(first, second)

    rng = np.random.default_rng(20260719)
    for _ in range(100):
        actions = rng.integers(0, 4, size=4, dtype=np.uint8)
        rebuilt.set_state(rebuilt.get_state())  # Invalidate only the visual cache.
        incremental_step = incremental.step(actions)
        rebuilt_step = rebuilt.step(actions)
        for actual, expected in zip(incremental_step[:4], rebuilt_step[:4]):
            np.testing.assert_array_equal(actual, expected)
        assert incremental.get_state() == rebuilt.get_state()
        terminated = incremental_step[2]
        if terminated.any():
            options = {"reset_mask": terminated, "start_indices": starts}
            incremental_reset, _ = incremental.reset(options=options)
            rebuilt_reset, _ = rebuilt.reset(options=options)
            np.testing.assert_array_equal(incremental_reset, rebuilt_reset)


def test_optimized_hot_path_preserves_golden_observation_trace():
    env = BreakoutVecEnv(num_envs=4, num_threads=1, frame_skip=4, frame_stack=4)
    observation, _ = env.reset(options={"start_indices": np.arange(4, dtype=np.int32)})
    digest = hashlib.sha256(observation.tobytes())
    for step in range(100):
        actions = np.array([(step + lane) % 4 for lane in range(4)], dtype=np.uint8)
        observation, reward, terminated, truncated, _ = env.step(actions)
        digest.update(observation.tobytes())
        digest.update(reward.tobytes())
        digest.update(terminated.tobytes())
        digest.update(truncated.tobytes())
        if terminated.any():
            observation, _ = env.reset(
                options={
                    "reset_mask": terminated,
                    "start_indices": np.arange(4, dtype=np.int32),
                }
            )
            digest.update(observation.tobytes())
    assert (
        digest.hexdigest()
        == "36720063423e4cc9ae644861da5d08454512f6a9d222a06fdb013e308b4722cb"
    )


def test_delayed_collision_latches_reproduce_the_top_left_corner_trace():
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=10 * FIXED_POINT_ONE,
        ball_y=34 * FIXED_POINT_ONE + FIXED_POINT_ONE // 2,
        ball_vx=-FIXED_POINT_ONE,
        ball_vy=-(3 * FIXED_POINT_ONE // 2),
        bricks=(1 << 108) - 1,
        lives=5,
    )

    trace = []
    for _ in range(6):
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.uint8))
        trace.append(
            (
                int(info["ball_x"][0]),
                int(info["ball_y"][0]),
                int(info["ball_vx"][0]),
                int(info["ball_vy"][0]),
                int(info["collision_events"][0]),
            )
        )
    assert trace == [
        (9 * FIXED_POINT_ONE, 24, -FIXED_POINT_ONE, -(3 * FIXED_POINT_ONE // 2), 0),
        (8 * FIXED_POINT_ONE, 25, -FIXED_POINT_ONE, 3 * FIXED_POINT_ONE // 2, 1),
        (7 * FIXED_POINT_ONE, 27, -FIXED_POINT_ONE, 3 * FIXED_POINT_ONE // 2, 0),
        (8 * FIXED_POINT_ONE, 28, FIXED_POINT_ONE, 3 * FIXED_POINT_ONE // 2, 1),
        (9 * FIXED_POINT_ONE, 30, FIXED_POINT_ONE, 3 * FIXED_POINT_ONE // 2, 0),
        (10 * FIXED_POINT_ONE, 31, FIXED_POINT_ONE, 3 * FIXED_POINT_ONE // 2, 0),
    ]


def test_atari_digital_paddle_inertia_trace():
    env = make_env(frame_skip=1)
    env.reset()
    env.step(np.array([1, 0, 0, 0], dtype=np.uint8))
    for _ in range(12):
        env.step(np.zeros(4, dtype=np.uint8))

    positions = []
    for _ in range(20):
        _, _, _, _, info = env.step(np.array([2, 0, 0, 0], dtype=np.uint8))
        positions.append(int(info["paddle_x"][0] // FIXED_POINT_ONE))
    assert positions == [
        26,
        26,
        26,
        26,
        26,
        27,
        27,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        45,
        47,
        49,
        51,
        53,
    ]


@pytest.mark.parametrize(
    "row,expected_reward", tuple(enumerate((7.0, 7.0, 4.0, 4.0, 1.0, 1.0)))
)
def test_reward_matches_stable_retro_score_delta_by_brick_row(row, expected_reward):
    env = make_env(frame_skip=1)
    env.reset()
    target = 1 << (row * 18 + 9)
    survivor = 1 << (107 if row < 5 else 0)
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=80 * FIXED_POINT_ONE,
        ball_y=(63 + row * 6) * FIXED_POINT_ONE,
        ball_vx=0,
        ball_vy=-FIXED_POINT_ONE,
        bricks=target | survivor,
        lives=5,
    )

    env.step(np.zeros(4, dtype=np.uint8))
    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.uint8))

    assert reward[0] == expected_reward
    assert info["score"][0] == expected_reward
    assert not terminated[0]


@pytest.mark.parametrize("lives,expected_terminated", ((2, False), (1, True)))
def test_life_loss_has_no_reward_shaping(lives, expected_terminated):
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=10 * FIXED_POINT_ONE,
        ball_y=217 * FIXED_POINT_ONE,
        ball_vx=FIXED_POINT_ONE,
        ball_vy=FIXED_POINT_ONE,
        bricks=1 | (1 << 47),
        lives=lives,
    )

    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.uint8))

    assert reward[0] == 0.0
    assert info["score"][0] == 0
    assert terminated[0] == expected_terminated


def test_board_clear_returns_only_the_score_delta_without_bonus():
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=80 * FIXED_POINT_ONE,
        ball_y=63 * FIXED_POINT_ONE,
        ball_vx=0,
        ball_vy=-FIXED_POINT_ONE,
        bricks=1 << 9,
        lives=5,
    )

    env.step(np.zeros(4, dtype=np.uint8))
    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.uint8))

    assert reward[0] == 7.0
    assert info["score"][0] == 7
    assert not terminated[0]
    assert info["bricks_remaining"][0] == 0
    assert info["walls_cleared"][0] == 1
