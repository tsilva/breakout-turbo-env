from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).parents[1] / "play_jerk.py"
_SPEC = importlib.util.spec_from_file_location("play_jerk", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
play_jerk = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = play_jerk
_SPEC.loader.exec_module(play_jerk)


def _write_policy(path: Path, **updates) -> None:
    payload = {"layout": "full", "frame_skip": 1, "actions": [0, 1, 2]}
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_policy_reads_action_tape(tmp_path):
    path = tmp_path / "jerk.json"
    _write_policy(path)
    policy = play_jerk.load_policy(path)
    assert policy.actions == (0, 1, 2)
    assert policy.layout == "full"
    assert policy.frame_skip == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"layout": "unknown"}, "layout"),
        ({"frame_skip": 0}, "frame_skip"),
        ({"actions": []}, "actions"),
        ({"actions": [0, 3]}, "every policy action"),
    ],
)
def test_load_policy_rejects_invalid_artifacts(tmp_path, updates, message):
    path = tmp_path / "jerk.json"
    _write_policy(path, **updates)
    with pytest.raises(ValueError, match=message):
        play_jerk.load_policy(path)


def test_run_rejects_invalid_display_values():
    policy = play_jerk.JerkPolicy(actions=(0,), layout="full", frame_skip=1)
    with pytest.raises(ValueError, match="positive"):
        play_jerk.run(policy, scale=0, fps=60, loop=False)


def test_parser_uses_the_shared_rlab_sized_viewer_scale():
    assert play_jerk._parser().parse_args([]).scale == 4


def test_latest_policy_selects_newest_timestamped_run(tmp_path):
    old = tmp_path / "jerk" / "20260712-100000" / "policy.json"
    new = tmp_path / "jerk" / "20260712-110000" / "policy.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    _write_policy(old)
    _write_policy(new)
    assert play_jerk.latest_policy(tmp_path) == new


def test_latest_policy_explains_how_to_train(tmp_path):
    with pytest.raises(ValueError, match="train.py jerk"):
        play_jerk.latest_policy(tmp_path)
