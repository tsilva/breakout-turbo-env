from importlib import resources

import numpy as np
import pytest
from breakout_turbo_env import ACTION_SETS, ACTION_TABLES, BreakoutVecEnv


def test_packaged_metadata_is_available_and_defines_simple():
    metadata = resources.files("breakout_turbo_env").joinpath(
        "data", "Breakout-Atari2600-v0", "metadata.json"
    )

    assert metadata.is_file()
    assert ACTION_TABLES["simple"] == ((), ("BUTTON",), ("RIGHT",), ("LEFT",))
    assert ACTION_SETS["simple"] == ("noop", "button", "right", "left")


def test_simple_preset_exposes_exact_discrete_contract():
    env = BreakoutVecEnv(use_restricted_actions="simple", num_envs=4, num_threads=1)
    try:
        assert env.action_preset == "simple"
        assert env.action_meanings == ("noop", "button", "right", "left")
        assert env.single_action_space.n == 4
        np.testing.assert_array_equal(env._native_actions([0, 1, 2, 3]), [0, 1, 2, 3])
    finally:
        env.close()


def test_inline_subset_and_reordering_map_to_native_commands():
    env = BreakoutVecEnv(
        use_restricted_actions=[["LEFT"], [], ["RIGHT"]],
        num_envs=3,
        num_threads=1,
    )
    try:
        assert env.action_preset is None
        assert env.action_meanings == ("left", "noop", "right")
        np.testing.assert_array_equal(env._native_actions([0, 1, 2]), [3, 0, 2])
    finally:
        env.close()


@pytest.mark.parametrize(
    "value",
    ["all", "discrete", "multi_discrete"],
)
def test_unsupported_builtin_modes_are_rejected(value):
    with pytest.raises(ValueError, match="does not support"):
        BreakoutVecEnv(use_restricted_actions=value)


def test_unreproducible_button_combination_is_rejected():
    with pytest.raises(ValueError, match="cannot reproduce"):
        BreakoutVecEnv(use_restricted_actions=[["BUTTON", "RIGHT"]])


def test_native_default_and_simple_have_identical_semantic_hashes():
    native = BreakoutVecEnv(num_envs=1)
    simple = BreakoutVecEnv(use_restricted_actions="simple", num_envs=1)
    try:
        assert native.action_preset is None
        assert native.action_table == simple.action_table
        assert native.action_table_hash == simple.action_table_hash
    finally:
        native.close()
        simple.close()
