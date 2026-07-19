from __future__ import annotations

import json

from breakout_turbo_env.benchmark import NUM_ENVS, build_parser, run_benchmark


def test_benchmark_contract_is_fixed_to_16_envs(capsys):
    args = build_parser().parse_args(
        ["--steps", "2", "--warmup", "0", "--repeats", "1"]
    )
    assert NUM_ENVS == 16
    rates = run_benchmark(
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=2,
    )
    assert len(rates) == 1
    assert rates[0] > 0
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith(
        "config=num_envs=16 steps=2 repeats=1 warmup=0 threads=2"
    )
    assert output[1] == "load_preflight=disabled"
    assert output[2].startswith("obs_shape=(16, 4, 84, 84) obs_dtype=uint8")
    assert output[3].startswith("run=1 elapsed_s=")
    assert "batch_steps_per_sec=" in output[3]
    assert "env_steps_per_sec=" in output[3]
    assert output[4].startswith("summary=env_steps_per_sec_mean=")
    assert "env_steps_per_sec_stdev=" in output[4]
    assert "best_env_steps_per_sec=" in output[4]
    assert "obs_buffer_gib_per_sec=" in output[4]


def test_benchmark_json_output_contract(capsys):
    args = build_parser().parse_args(
        ["--steps", "2", "--warmup", "0", "--repeats", "2", "--json"]
    )
    rates = run_benchmark(
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=2,
        json_output=args.json,
    )
    assert len(rates) == 2

    payload = json.loads(capsys.readouterr().out)
    config = payload["config"]
    assert config["num_envs"] == NUM_ENVS
    assert config["steps"] == 2
    assert config["repeats"] == 2
    assert config["obs_shape"] == [16, 4, 84, 84]

    runs = payload["runs"]
    assert len(runs) == 2
    for run in runs:
        assert set(run) >= {
            "index",
            "steps",
            "elapsed_s",
            "env_steps_per_sec",
            "batch_steps_per_sec",
            "emulated_frames_per_sec",
        }
        assert run["env_steps_per_sec"] > 0

    summary = payload["summary"]
    esp = summary["env_steps_per_sec"]
    assert esp["mean"] > 0
    assert esp["median"] > 0
    assert esp["best"] > 0
    assert esp["unit"] == "env_steps_per_sec"
    assert "stdev" in esp
    assert summary["batch_steps_per_sec"]["unit"] == "batch_steps_per_sec"
    assert summary["emulated_frames_per_sec"]["unit"] == "emulated_frames_per_sec"
    assert summary["obs_buffer_gib_per_sec"] >= 0


def test_benchmark_default_preserves_human_output(capsys):
    args = build_parser().parse_args(
        ["--steps", "2", "--warmup", "0", "--repeats", "1"]
    )
    run_benchmark(
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=2,
        json_output=False,
    )
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("config=num_envs=16")
    assert output[-1].startswith("summary=")
