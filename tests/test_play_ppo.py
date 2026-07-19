from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parents[1]
_TRAIN_SPEC = importlib.util.spec_from_file_location("train_ppo", _ROOT / "train_ppo.py")
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
train_ppo = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules[_TRAIN_SPEC.name] = train_ppo
_TRAIN_SPEC.loader.exec_module(train_ppo)

_SCRIPT = _ROOT / "play_ppo.py"
_SPEC = importlib.util.spec_from_file_location("play_ppo", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
play_ppo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = play_ppo
_SPEC.loader.exec_module(play_ppo)


def _write_policy(path: Path, *, algorithm: str = "PPO") -> None:
    metadata = {"algorithm": algorithm, "format_version": 1, "layout": "full", "frame_skip": 1}
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        trunk__0__weight=np.zeros((64, 6), dtype=np.float32),
        trunk__0__bias=np.zeros(64, dtype=np.float32),
        trunk__2__weight=np.zeros((64, 64), dtype=np.float32),
        trunk__2__bias=np.zeros(64, dtype=np.float32),
        actor__weight=np.zeros((4, 64), dtype=np.float32),
        actor__bias=np.zeros(4, dtype=np.float32),
    )


def test_load_policy_and_choose_an_action(tmp_path):
    path = tmp_path / "policy.npz"
    _write_policy(path)
    policy = play_ppo.load_policy(path)
    assert policy.layout == "full"
    assert policy.action({
        "paddle_x": np.array([0]), "ball_x": np.array([0]), "ball_y": np.array([0]),
        "ball_vx": np.array([0]), "ball_vy": np.array([0]),
        "awaiting_fire": np.array([0]),
    }).tolist() == [0]


def test_load_policy_rejects_a_non_ppo_artifact(tmp_path):
    path = tmp_path / "policy.npz"
    _write_policy(path, algorithm="JERK")
    with pytest.raises(ValueError, match="PPO"):
        play_ppo.load_policy(path)


def test_latest_policy_uses_newest_run(tmp_path):
    old = tmp_path / "ppo" / "20260712-100000" / "policy.npz"
    new = tmp_path / "ppo" / "20260712-110000" / "policy.npz"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    _write_policy(old)
    _write_policy(new)
    assert play_ppo.latest_policy(tmp_path) == new


def test_parser_uses_the_shared_rlab_sized_viewer_scale():
    assert play_ppo._parser().parse_args([]).scale == 4
