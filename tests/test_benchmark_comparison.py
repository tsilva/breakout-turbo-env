from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark_comparison.py"


def comparison_module():
    spec = importlib.util.spec_from_file_location("benchmark_comparison", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_batches_map_native_contract_to_stable_retro_buttons():
    module = comparison_module()
    native, stable = module.action_batches(8)

    np.testing.assert_array_equal(native, [0, 1, 2, 3, 0, 1, 2, 3])
    assert stable.shape == (8, 8)
    np.testing.assert_array_equal(stable[:, 0], native == 1)
    np.testing.assert_array_equal(stable[:, 7], native == 2)
    np.testing.assert_array_equal(stable[:, 6], native == 3)
    assert stable.sum() == 6


def test_summary_reports_median_and_sample_stdev():
    module = comparison_module()
    summary = module.summarize([10.0, 20.0, 30.0])

    assert summary["mean_env_steps_per_sec"] == 20.0
    assert summary["median_env_steps_per_sec"] == 20.0
    assert summary["stdev_env_steps_per_sec"] == 10.0
    assert summary["best_env_steps_per_sec"] == 30.0
