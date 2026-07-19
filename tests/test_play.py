from __future__ import annotations

import pytest
import numpy as np

from breakout_turbo_env.play import (
    _hud_text,
    _limit_frame_rate,
    _print_episode_stats,
    _scaled_frame_size,
    build_parser,
    run,
)


def test_play_parser_defaults_and_layout_selection():
    defaults = build_parser().parse_args([])
    assert defaults.layout == "full"
    assert defaults.scale == 4
    assert defaults.frame_skip == 1
    assert defaults.show_obs is False
    assert defaults.uncapped is False
    selected = build_parser().parse_args(
        ["--layout", "tunnel", "--scale", "4", "--uncapped"]
    )
    assert selected.layout == "tunnel"
    assert selected.scale == 4
    assert selected.uncapped is True


def test_default_player_window_preserves_the_native_frame_aspect_ratio():
    assert _scaled_frame_size(160, 210, 4) == (640, 840)


def test_uncapped_mode_skips_the_frame_limiter():
    class Clock:
        calls: list[int] = []

        def tick(self, fps):
            self.calls.append(fps)

    clock = Clock()
    _limit_frame_rate(clock, 60)
    _limit_frame_rate(None, 60)
    assert clock.calls == [60]


def test_play_rejects_invalid_runtime_values():
    with pytest.raises(ValueError, match="positive"):
        run(layout="full", scale=0, fps=60, frame_skip=1, max_frames=1)


def test_episode_stats_are_printed(capsys):
    info = {
        "score": np.array([48]),
        "lives": np.array([2]),
        "bricks_remaining": np.array([0]),
        "tick": np.array([1234]),
    }
    _print_episode_stats(
        info,
        episode=3,
        layout="full",
        episode_return=48.0,
        display_steps=309,
        elapsed=5.25,
    )
    output = capsys.readouterr().out
    assert "episode_end episode=3" in output
    assert "outcome=cleared" in output
    assert "score=48" in output
    assert "return=48.0" in output
    assert "bricks_cleared=108" in output
    assert "native_ticks=1234" in output


def test_hud_shows_score_lives_and_bricks():
    info = {
        "score": np.array([7]),
        "lives": np.array([2]),
        "bricks_remaining": np.array([41]),
    }
    assert _hud_text(info, paused=False) == "SCORE 007    LIVES 2    BRICKS 41"
    assert _hud_text(info, paused=True).endswith("PAUSED")
