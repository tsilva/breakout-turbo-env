from __future__ import annotations

from breakout_turbo_env.benchmark import NUM_ENVS, build_parser, run_benchmark


def test_benchmark_contract_is_fixed_to_16_envs():
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
