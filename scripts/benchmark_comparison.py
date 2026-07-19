#!/usr/bin/env python3
"""Compare equivalent breakout-turbo-env and Stable Retro vector workloads."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

NUM_ENVS = 16
FRAME_SKIP = 4
FRAME_STACK = 4
OBSERVATION_SIZE = 84


def action_batches(num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return equivalent native and Stable Retro action batches."""
    native = np.arange(num_envs, dtype=np.uint8) % 4
    stable = np.zeros((num_envs, 8), dtype=np.int8)
    stable[native == 1, 0] = 1  # FIRE
    stable[native == 2, 7] = 1  # RIGHT
    stable[native == 3, 6] = 1  # LEFT
    return native, stable


def run_backend(
    env: Any,
    actions: np.ndarray,
    *,
    steps: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    env.reset()
    for _ in range(warmup):
        _, _, terminated, truncated, _ = env.step(actions)
        done = np.asarray(terminated) | np.asarray(truncated)
        if done.any():
            env.reset(options={"reset_mask": done})

    rates: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(actions)
            done = np.asarray(terminated) | np.asarray(truncated)
            if done.any():
                env.reset(options={"reset_mask": done})
        elapsed = time.perf_counter() - started
        rates.append(steps * NUM_ENVS / elapsed)
    return rates


def summarize(rates: list[float]) -> dict[str, float]:
    return {
        "mean_env_steps_per_sec": statistics.fmean(rates),
        "median_env_steps_per_sec": statistics.median(rates),
        "stdev_env_steps_per_sec": (
            statistics.stdev(rates) if len(rates) > 1 else 0.0
        ),
        "best_env_steps_per_sec": max(rates),
    }


def build_turbo(num_threads: int):
    from breakout_turbo_env import BreakoutVecEnv

    return BreakoutVecEnv(
        num_envs=NUM_ENVS,
        num_threads=num_threads,
        obs_resize=(OBSERVATION_SIZE, OBSERVATION_SIZE),
        obs_resize_algorithm="area",
        obs_grayscale=True,
        obs_layout="chw",
        obs_copy="safe_view",
        frame_skip=FRAME_SKIP,
        frame_stack=FRAME_STACK,
        maxpool_last_two=False,
        info_filter="none",
    )


def build_stable(num_threads: int, stable_retro_repo: Path):
    sys.path.insert(0, str(stable_retro_repo))
    import stable_retro as retro

    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")
    return retro.RetroVecEnv(
        "Breakout-Atari2600-v0",
        state="Start",
        num_envs=NUM_ENVS,
        num_threads=num_threads,
        render_mode="rgb_array",
        obs_resize=(OBSERVATION_SIZE, OBSERVATION_SIZE),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=FRAME_SKIP,
        frame_stack=FRAME_STACK,
        maxpool_last_two=False,
        info_filter="none",
        obs_copy="safe_view",
    )


def benchmark(
    factory: Callable[[], Any],
    actions: np.ndarray,
    *,
    steps: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    env = factory()
    try:
        return run_backend(
            env,
            actions,
            steps=steps,
            warmup=warmup,
            repeats=repeats,
        )
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--stable-retro-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "stable-retro-turbo",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if min(args.steps, args.repeats, args.threads) <= 0 or args.warmup < 0:
        raise SystemExit("steps, repeats, and threads must be positive")
    stable_repo = args.stable_retro_repo.resolve()
    if not stable_repo.is_dir():
        raise SystemExit(f"Stable Retro checkout not found: {stable_repo}")

    native_actions, stable_actions = action_batches(NUM_ENVS)
    turbo_rates = benchmark(
        lambda: build_turbo(args.threads),
        native_actions,
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    stable_rates = benchmark(
        lambda: build_stable(args.threads, stable_repo),
        stable_actions,
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    turbo = summarize(turbo_rates)
    stable = summarize(stable_rates)
    result = {
        "config": {
            "num_envs": NUM_ENVS,
            "num_threads": args.threads,
            "steps": args.steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "frame_skip": FRAME_SKIP,
            "frame_stack": FRAME_STACK,
            "observation": "grayscale CHW 4x84x84 safe_view",
            "info_filter": "none",
            "maxpool_last_two": False,
        },
        "breakout_turbo_env": turbo,
        "stable_retro_turbo": stable,
        "median_speedup": (
            turbo["median_env_steps_per_sec"]
            / stable["median_env_steps_per_sec"]
        ),
        "raw_rates": {
            "breakout_turbo_env": turbo_rates,
            "stable_retro_turbo": stable_rates,
        },
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(json.dumps(result["config"], sort_keys=True))
    print(
        "breakout-turbo-env "
        f"median={turbo['median_env_steps_per_sec']:.1f} env-steps/s"
    )
    print(
        "stable-retro-turbo "
        f"median={stable['median_env_steps_per_sec']:.1f} env-steps/s"
    )
    print(f"median_speedup={result['median_speedup']:.2f}x")


if __name__ == "__main__":
    main()
