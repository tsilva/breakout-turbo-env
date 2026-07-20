from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
from breakout_turbo_env.sb3 import _lane_infos, make_sb3_vec_env


class FakeVecEnv:
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.reset_infos = [{} for _ in range(num_envs)]
        self._seeds = [None for _ in range(num_envs)]
        self._options = [{} for _ in range(num_envs)]

    def _reset_seeds(self):
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self):
        self._options = [{} for _ in range(self.num_envs)]

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class FakeBreakout:
    num_envs = 2
    single_observation_space = object()
    single_action_space = object()
    render_mode = "rgb_array"

    def __init__(self):
        self.reset_masks = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        mask = np.ones(2, dtype=np.bool_) if options is None else options["reset_mask"]
        self.reset_masks.append(mask.copy())
        observations = np.asarray([[[10]], [[20]]], dtype=np.uint8)
        infos = {
            "start_id": np.asarray(["Start", "Start"], dtype=object),
            "_start_id": mask.copy(),
        }
        return observations, infos

    def step(self, actions):
        assert actions.tolist() == [2, 3]
        observations = np.asarray([[[99]], [[42]]], dtype=np.uint8)
        rewards = np.asarray([1.0, 0.0], dtype=np.float32)
        terminated = np.asarray([True, False])
        truncated = np.asarray([False, False])
        infos = {
            "score": np.asarray([7, 0]),
            "_score": np.asarray([True, True]),
            "pending_reset": np.asarray([1, 0]),
            "_pending_reset": np.asarray([True, True]),
        }
        return observations, rewards, terminated, truncated, infos

    def close(self):
        self.closed = True


def install_fake_sb3(monkeypatch):
    root = ModuleType("stable_baselines3")
    common = ModuleType("stable_baselines3.common")
    vec_env = ModuleType("stable_baselines3.common.vec_env")
    vec_env.VecEnv = FakeVecEnv
    monkeypatch.setitem(sys.modules, "stable_baselines3", root)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", vec_env)


def test_lane_infos_honors_gymnasium_presence_masks():
    infos = {
        "score": np.asarray([7, 0]),
        "_score": np.asarray([True, False]),
    }
    assert _lane_infos(infos, 2) == [{"score": 7}, {}]


def test_adapter_preserves_terminal_observation_and_resets_only_done_lane(monkeypatch):
    install_fake_sb3(monkeypatch)
    native = FakeBreakout()
    env = make_sb3_vec_env(native)

    initial = env.reset()
    assert initial.tolist() == [[[10]], [[20]]]
    observations, rewards, done, infos = env.step(np.asarray([2, 3]))

    assert observations.tolist() == [[[10]], [[20]]]
    assert rewards.tolist() == [1.0, 0.0]
    assert done.tolist() == [True, False]
    assert infos[0]["terminal_observation"].tolist() == [[99]]
    assert infos[0]["TimeLimit.truncated"] is False
    assert native.reset_masks[-1].tolist() == [True, False]
    assert env.reset_infos[0]["start_id"] == "Start"

    env.close()
    assert native.closed
