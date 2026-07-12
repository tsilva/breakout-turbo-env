from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np

from breakout_turbo_env import BreakoutVecEnv


_SCRIPT = Path(__file__).parents[1] / "train_jerk.py"
_SPEC = importlib.util.spec_from_file_location("train_jerk", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
train_jerk = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_jerk
_SPEC.loader.exec_module(train_jerk)


def test_train_generation_retains_only_actions():
    env = BreakoutVecEnv(
        num_envs=4,
        num_threads=1,
        obs_resize=(8, 8),
        frame_stack=1,
        frame_skip=1,
        info_filter="all",
    )
    candidate = train_jerk.train_generation(
        env,
        layout="sparse",
        champion=[],
        max_steps=8,
        exploration=0.1,
        rng=np.random.default_rng(1),
    )
    assert candidate.actions
    assert set(candidate.actions) <= {0, 1, 2}
    assert not hasattr(candidate, "state")


def test_saved_policy_contains_actions_but_no_state(tmp_path):
    candidate = train_jerk.Candidate(actions=[0, 1, 2], score=1, reward=1.0, lives=3, solved=False)
    output = tmp_path / "policy.json"
    train_jerk.save_policy(output, candidate, layout="full", frame_skip=1, seed=7)
    payload = json.loads(output.read_text())
    assert payload["actions"] == [0, 1, 2]
    assert not any("state" in key.lower() for key in payload)


def test_create_run_dir_uses_algorithm_and_timestamp(tmp_path):
    first = train_jerk.create_run_dir(tmp_path)
    second = train_jerk.create_run_dir(tmp_path)
    assert first.parent == tmp_path / "jerk"
    assert second.parent == tmp_path / "jerk"
    assert first != second
    assert first.name[:8].isdigit()
    assert first.name[8] == "-"
