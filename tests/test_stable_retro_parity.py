"""Live differential tests against the Stable Retro Breakout cartridge.

These tests contain no reference trace. Each case generates actions at runtime
and applies the same action to Stable Retro and breakout-turbo-env before
comparing the resulting native frame and transition values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STABLE_REPO = Path(
    os.environ.get(
        "BREAKOUT_STABLE_RETRO_REPO", REPO_ROOT.parent / "stable-retro-turbo"
    )
).resolve()
DATA_DIR = STABLE_REPO / "stable_retro/data/stable/Breakout-Atari2600-v0"
REQUIRED_REFERENCE_FILES = (
    DATA_DIR / "rom.a26",
    DATA_DIR / "Start.state",
    DATA_DIR / "data.json",
    DATA_DIR / "scenario.json",
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(STABLE_REPO))

pytestmark = pytest.mark.stable_retro


def _missing_reference_reason() -> str | None:
    missing = [str(path) for path in REQUIRED_REFERENCE_FILES if not path.is_file()]
    if missing:
        return "missing Stable Retro Breakout reference files: " + ", ".join(missing)
    try:
        import stable_retro  # noqa: F401
    except (ImportError, OSError) as error:
        return f"Stable Retro cannot be imported from {STABLE_REPO}: {error}"
    return None


@pytest.fixture(scope="session")
def stable_reference():
    reason = _missing_reference_reason()
    if reason is not None:
        if os.environ.get("BREAKOUT_REQUIRE_STABLE_RETRO") == "1":
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)

    from compare_stable_retro import StableReference

    reference = StableReference(DATA_DIR)
    try:
        yield reference
    finally:
        reference.close()


@pytest.mark.parametrize("corner", ("top-left", "top-right"))
def test_forced_corner_dynamics_match_live_cartridge(stable_reference, corner):
    from compare_stable_retro import compare_corner

    transitions = compare_corner(stable_reference, corner, frames=20)
    assert transitions
    for transition in transitions:
        assert transition.stable == transition.turbo, transition
        assert transition.stable_delta == transition.turbo_delta, transition
        assert transition.stable_reward == transition.turbo_reward, transition


@pytest.mark.parametrize(
    ("policy", "aim", "seed", "max_frames"),
    (
        pytest.param("tracking", 8, None, 8_000, id="tracking"),
        pytest.param("predictive", 4, None, 8_000, id="predictive-aim4"),
        pytest.param("predictive", 6, None, 12_000, id="predictive-aim6"),
        pytest.param("predictive", 8, None, 10_000, id="predictive-aim8"),
        pytest.param("predictive", 10, None, 8_000, id="predictive-aim10"),
        pytest.param("predictive", 12, None, 8_000, id="predictive-aim12"),
        pytest.param("random", 8, 0, 2_000, id="random-seed0"),
        pytest.param("random", 8, 1, 2_000, id="random-seed1"),
        pytest.param("random", 8, 2, 2_000, id="random-seed2"),
    ),
)
def test_native_frames_rewards_and_lifecycle_match_live_cartridge(
    stable_reference,
    policy,
    aim,
    seed,
    max_frames,
):
    from compare_stable_retro import compare_episode

    result = compare_episode(
        stable_reference,
        policy=policy,
        aim=aim,
        seed=seed,
        max_frames=max_frames,
    )
    assert result.exact, result.mismatch


def test_policy_observations_match_live_cartridge(stable_reference):
    from breakout_turbo_env import BreakoutVecEnv

    def policy_frame(frame):
        rgb565 = np.asarray(frame, dtype=np.uint16).copy()
        rgb565[..., 0] &= 0xF8
        rgb565[..., 1] &= 0xFC
        rgb565[..., 2] &= 0xF8
        grayscale = (
            rgb565[..., 0] * 77
            + rgb565[..., 1] * 150
            + rgb565[..., 2] * 29
            + 128
        ) >> 8
        grayscale[:17] = 0
        resized = np.empty((84, 84), dtype=np.uint8)
        for output_y in range(84):
            source_y0 = output_y * 210 // 84
            source_y1 = max((output_y + 1) * 210 // 84, source_y0 + 1)
            for output_x in range(84):
                source_x0 = output_x * 160 // 84
                source_x1 = max((output_x + 1) * 160 // 84, source_x0 + 1)
                region = grayscale[source_y0:source_y1, source_x0:source_x1]
                resized[output_y, output_x] = (
                    region.sum(dtype=np.uint32) // region.size
                )
        return resized

    stable_reference.reopen()
    stable_frame, stable_info = stable_reference.env.reset()
    turbo = BreakoutVecEnv(
        "Breakout-Atari2600-v0",
        state="Start",
        scenario="scenario",
        info="data",
        use_restricted_actions="filtered",
        record=False,
        players=1,
        inttype="stable",
        render_mode="rgb_array",
        num_envs=1,
        num_threads=1,
        obs_resize=(84, 84),
        obs_crop=(17, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=False,
        sticky_action_prob=0.0,
        obs_copy="copy",
        obs_layout="chw",
        obs_grayscale=True,
        info_filter="all",
    )

    try:
        turbo_observation, turbo_info = turbo.reset(seed=[123])
        stable_stack = [policy_frame(stable_frame)] * 4
        np.testing.assert_array_equal(turbo_observation[0], stable_stack)

        for step in range(128):
            if stable_reference.awaiting_fire():
                direction = 0
                fire = True
            elif (step // 8) % 4 == 1:
                direction = 1
                fire = False
            elif (step // 8) % 4 == 3:
                direction = -1
                fire = False
            else:
                direction = 0
                fire = False
            stable_action = stable_reference.action(direction, fire=fire)
            stable_reward = 0.0
            stable_terminated = False
            stable_truncated = False
            for _ in range(4):
                (
                    stable_frame,
                    frame_reward,
                    stable_terminated,
                    stable_truncated,
                    stable_info,
                ) = stable_reference.env.step(stable_action)
                stable_reward += float(frame_reward)
                if stable_terminated or stable_truncated:
                    break
            stable_stack = [*stable_stack[1:], policy_frame(stable_frame)]

            turbo_action = stable_action[np.newaxis, :]
            (
                turbo_observation,
                turbo_reward,
                turbo_terminated,
                turbo_truncated,
                turbo_info,
            ) = turbo.step(turbo_action)
            np.testing.assert_array_equal(turbo_observation[0], stable_stack)
            np.testing.assert_array_equal(turbo_reward, [stable_reward])
            np.testing.assert_array_equal(turbo_terminated, [stable_terminated])
            np.testing.assert_array_equal(turbo_truncated, [stable_truncated])
            for key in ("ball_y", "lives", "score"):
                np.testing.assert_array_equal(turbo_info[key], [stable_info[key]])
            if stable_terminated or stable_truncated:
                break
    finally:
        turbo.close()


def test_live_cartridge_has_two_walls_864_top_score_and_lives_only_done(
    stable_reference,
):
    from compare_stable_retro import Point

    reference = stable_reference
    reference.reopen()

    def set_byte(address: int, value: int) -> None:
        reference.env.data.memory[{"address": address, "type": "|u1"}] = value

    def force_last_brick(score: int) -> None:
        for address in range(0x80, 0xA4):
            set_byte(address, 0)
        # The low two PF bits are one complete bottom-row brick at x=128..135.
        set_byte(0x80, 0x03)
        reference.env.set_value("score", score)
        reference.env.data.update_ram()

    def wall_is_empty() -> bool:
        return not np.any(reference.env.get_ram()[:36])

    def step(fire: bool = False):
        return reference.env.step(reference.action(fire=fire))

    state = reference.find_flight(1, -1)
    reference.force(state, Point(130, 94))
    force_last_brick(431)

    for _ in range(8):
        _, reward, terminated, _, _ = step()
        if reference.score() == 432:
            break
    assert reward == 1.0
    assert not terminated
    assert reference.score() == 432
    assert wall_is_empty()

    # Preserve the cartridge phase and velocity, but place the returning ball
    # immediately above the paddle to expose the exact refill boundary.
    reference.force(bytes(reference.env.em.get_state()), Point(80, 184))
    reference.force_paddle(92)
    previous_y = reference.point().y
    for _ in range(8):
        reference.force_paddle(92)
        step()
        current_y = reference.point().y
        if current_y < previous_y:
            break
        previous_y = current_y
    assert wall_is_empty(), "the paddle-bounce frame must still be empty"
    step()
    assert not wall_is_empty(), "wall two must appear on the following native frame"

    # Reach the second-wall boundary without resetting the cartridge phase.
    reference.force(bytes(reference.env.em.get_state()), Point(130, 94))
    force_last_brick(863)
    for _ in range(8):
        _, reward, terminated, _, _ = step()
        if reference.score() == 864:
            break
    assert reward == 1.0
    assert not terminated
    assert reference.score() == 864
    assert wall_is_empty()

    # A later paddle return must not create wall three.
    reference.force(bytes(reference.env.em.get_state()), Point(80, 184))
    reference.force_paddle(92)
    previous_y = reference.point().y
    for _ in range(8):
        reference.force_paddle(92)
        _, _, terminated, _, _ = step()
        current_y = reference.point().y
        if current_y < previous_y:
            break
        previous_y = current_y
    assert not terminated
    step()
    assert reference.score() == 864
    assert wall_is_empty()

    # The empty post-wall-two game ends only as each remaining life is lost.
    while reference.lives() > 0:
        if reference.awaiting_fire():
            _, _, terminated, _, _ = step(fire=True)
            assert not terminated
        before = reference.lives()
        reference.force(bytes(reference.env.em.get_state()), Point(80, 217))
        _, _, terminated, _, _ = step()
        assert reference.lives() == before - 1
        assert terminated == (reference.lives() == 0)
        assert reference.score() == 864
        assert wall_is_empty()
