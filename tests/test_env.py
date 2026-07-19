from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from breakout_turbo_env import (
    BreakoutVecEnv,
    FIXED_POINT_ONE,
    RAW_HEIGHT,
    RENDER_HEIGHT,
    RENDER_WIDTH,
)


def make_env(**kwargs):
    return BreakoutVecEnv(num_envs=4, num_threads=2, **kwargs)


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


def test_branches_cover_all_actions_without_mutating_source():
    env = make_env(frame_skip=1)
    env.reset()
    states = env.get_state()[:2]
    before = env.get_state()
    result = env.branch(states)
    assert result["observations"].shape == (6, 4, 84, 84)
    np.testing.assert_array_equal(result["actions"], [0, 1, 2, 0, 1, 2])
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


def test_all_layouts_use_the_same_predictable_launch_in_taller_arena():
    env = make_env(frame_skip=1)
    starts = np.arange(4, dtype=np.int32)
    _, info = env.reset(options={"start_indices": starts})
    assert RAW_HEIGHT == 96
    assert np.all(info["ball_y"] == 82 * FIXED_POINT_ONE)
    assert len(set(info["ball_vx"].tolist())) == 1
    assert len(set(info["ball_vy"].tolist())) == 1


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
    np.testing.assert_array_equal(frame[17:32], np.broadcast_to(gray, frame[17:32].shape))
    np.testing.assert_array_equal(frame[32:57, :8], np.broadcast_to(gray, frame[32:57, :8].shape))
    np.testing.assert_array_equal(frame[32:57, 8:152], np.broadcast_to(black, frame[32:57, 8:152].shape))
    for row, color in enumerate(row_colors):
        band = frame[57 + row * 6 : 63 + row * 6, 8:152]
        np.testing.assert_array_equal(band, np.broadcast_to(color, band.shape))
    np.testing.assert_array_equal(frame[5:15, 100:104], np.broadcast_to(gray, (10, 4, 3)))
    np.testing.assert_array_equal(frame[5:15, 104:112], np.broadcast_to(black, (10, 8, 3)))
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
        "ball_y": 50 * FIXED_POINT_ONE,
        "ball_vx": FIXED_POINT_ONE,
        "ball_vy": FIXED_POINT_ONE,
        "bricks": 1,
        "lives": 3,
    }
    cases = (
        (0, 1, 8, 8),
        (39, 48, 76, 80),
        (78, 95, 144, 150),
    )
    for paddle_source_x, ball_source_x, paddle_x, ball_x in cases:
        env.configure_lane(
            0,
            paddle_x=paddle_source_x * FIXED_POINT_ONE,
            ball_x=ball_source_x * FIXED_POINT_ONE,
            **common,
        )
        frame = env.render()

        # The active Atari sprite is 16x4. Its right half deliberately merges
        # with the permanent red cap at the rightmost travel limit.
        np.testing.assert_array_equal(
            frame[189:193, paddle_x : paddle_x + 16],
            np.broadcast_to(red, (4, 16, 3)),
        )

        # At this source height the isolated ball lands at y=117 and must be
        # exactly 2x4 (eight pixels), including both horizontal limits.
        np.testing.assert_array_equal(
            frame[117:121, ball_x : ball_x + 2],
            np.broadcast_to(red, (4, 2, 3)),
        )
        if ball_x > 8:
            np.testing.assert_array_equal(
                frame[117:121, ball_x - 1],
                np.broadcast_to(black, (4, 3)),
            )
        if ball_x + 2 < 152:
            np.testing.assert_array_equal(
                frame[117:121, ball_x + 2],
                np.broadcast_to(black, (4, 3)),
            )


def test_render_reflects_missing_bricks_and_lane_status():
    env = make_env(frame_skip=1)
    env.reset(options={"start_indices": np.array([1, 0, 0, 0], dtype=np.int32)})
    frame = env.render()
    black = np.array([0, 0, 0], dtype=np.uint8)
    red = np.array([200, 72, 72], dtype=np.uint8)
    np.testing.assert_array_equal(frame[57:63, 8:26], np.broadcast_to(red, frame[57:63, 8:26].shape))
    np.testing.assert_array_equal(frame[57:63, 26:44], np.broadcast_to(black, frame[57:63, 26:44].shape))
    assert np.any(frame[5:15, 36:80])
    assert np.any(frame[189:193, 8:152] == red)


def test_terminal_lane_requires_explicit_reset_then_can_continue():
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=10 * FIXED_POINT_ONE,
        ball_y=(RAW_HEIGHT + 2) * FIXED_POINT_ONE,
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
    for key in ("paddle_x", "ball_x", "ball_y", "ball_vx", "ball_vy", "brick_mask", "tick"):
        np.testing.assert_array_equal(skipped_info[key], repeated_info[key])


def test_thread_count_does_not_change_trace():
    serial = BreakoutVecEnv(num_envs=16, num_threads=1)
    parallel = BreakoutVecEnv(num_envs=16, num_threads=8)
    serial.reset()
    parallel.reset()
    rng = np.random.default_rng(1234)
    for _ in range(20):
        actions = rng.integers(0, 3, size=16, dtype=np.uint8)
        serial.step(actions)
        parallel.step(actions)
    assert serial.get_state() == parallel.get_state()


def test_optimized_hot_path_preserves_golden_observation_trace():
    env = BreakoutVecEnv(num_envs=4, num_threads=1, frame_skip=4, frame_stack=4)
    observation, _ = env.reset(options={"start_indices": np.arange(4, dtype=np.int32)})
    digest = hashlib.sha256(observation.tobytes())
    for step in range(100):
        actions = np.array([(step + lane) % 3 for lane in range(4)], dtype=np.uint8)
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
    assert digest.hexdigest() == "6f4c1474aad16c0320094e38aa862c80188e0cbbbb137710a0ed48cd94673148"


@pytest.mark.parametrize(
    ("ball_x", "ball_y", "ball_vx", "ball_vy", "velocity_key", "expected_sign"),
    [
        (9, 6, 0, 1, "ball_vy", -1),
        (9, 12, 0, -1, "ball_vy", 1),
        (3, 9, 1, 0, "ball_vx", -1),
        (15, 9, -1, 0, "ball_vx", 1),
    ],
)
def test_ball_bounces_on_every_brick_face(
    ball_x, ball_y, ball_vx, ball_vy, velocity_key, expected_sign
):
    env = make_env(frame_skip=1)
    env.reset()
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=ball_x * FIXED_POINT_ONE,
        ball_y=ball_y * FIXED_POINT_ONE,
        ball_vx=ball_vx * FIXED_POINT_ONE,
        ball_vy=ball_vy * FIXED_POINT_ONE,
        bricks=1 | (1 << 47),
        lives=3,
    )
    _, reward, _, _, info = env.step(np.zeros(4, dtype=np.uint8))
    assert reward[0] == 7.0
    assert info["brick_mask"][0] == 1 << 47
    assert int(info[velocity_key][0]) * expected_sign > 0
    assert info["collision_events"][0] & 4


@pytest.mark.parametrize(
    "row,expected_reward", enumerate((7.0, 7.0, 4.0, 4.0, 1.0, 1.0))
)
def test_reward_matches_stable_retro_score_delta_by_brick_row(row, expected_reward):
    env = make_env(frame_skip=1)
    env.reset()
    target = 1 << (row * 8)
    survivor = 1 << (47 if row < 5 else 0)
    env.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=9 * FIXED_POINT_ONE,
        ball_y=(6 + row * 5) * FIXED_POINT_ONE,
        ball_vx=0,
        ball_vy=FIXED_POINT_ONE,
        bricks=target | survivor,
        lives=3,
    )

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
        ball_y=(RAW_HEIGHT + 2) * FIXED_POINT_ONE,
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
        ball_x=9 * FIXED_POINT_ONE,
        ball_y=6 * FIXED_POINT_ONE,
        ball_vx=0,
        ball_vy=FIXED_POINT_ONE,
        bricks=1,
        lives=3,
    )

    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.uint8))

    assert reward[0] == 7.0
    assert info["score"][0] == 7
    assert terminated[0]
