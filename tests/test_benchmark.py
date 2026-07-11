from __future__ import annotations

from breakout_turbo_env.benchmark import NUM_ENVS, build_parser, run_benchmark


def test_benchmark_contract_is_fixed_to_16_envs(capsys):
    args = build_parser().parse_args(["--steps", "2", "--warmup", "0", "--repeats", "1"])
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
    assert output[0].startswith("config=num_envs=16 steps=2 repeats=1 warmup=0 threads=2")
    assert output[1] == "load_preflight=disabled"
    assert output[2].startswith("obs_shape=(16, 4, 84, 84) obs_dtype=uint8")
    assert output[3].startswith("run=1 elapsed_s=")
    assert "batch_steps_per_sec=" in output[3]
    assert "env_steps_per_sec=" in output[3]
    assert output[4].startswith("summary=env_steps_per_sec_mean=")
    assert "env_steps_per_sec_stdev=" in output[4]
    assert "best_env_steps_per_sec=" in output[4]
    assert "obs_buffer_gib_per_sec=" in output[4]
