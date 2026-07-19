"""Live differential tests against the Stable Retro Breakout cartridge.

These tests contain no reference trace. Each case generates actions at runtime
and applies the same action to Stable Retro and breakout-turbo-env before
comparing the resulting native frame and transition values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STABLE_REPO = Path(
    os.environ.get("BREAKOUT_STABLE_RETRO_REPO", REPO_ROOT.parent / "stable-retro-turbo")
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
