from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np

from .env import BreakoutVecEnv

NUM_ENVS = 16
OBSERVATION_SIZE = 84
FRAME_SKIP = 4
FRAME_STACK = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the fixed 16-env full-preprocessing policy path"
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(NUM_ENVS, os.cpu_count() or NUM_ENVS),
    )
    return parser


def run_benchmark(*, steps: int, warmup: int, repeats: int, threads: int) -> list[float]:
    if steps <= 0 or warmup < 0 or repeats <= 0 or threads <= 0:
        raise ValueError("steps, repeats, and threads must be positive; warmup must be non-negative")
    env = BreakoutVecEnv(
        num_envs=NUM_ENVS,
        num_threads=threads,
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
    actions = (np.arange(NUM_ENVS, dtype=np.uint8) % 3).astype(np.uint8)
    rates: list[float] = []
    try:
        env.reset()
        for _ in range(warmup):
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if done.any():
                env.reset(options={"reset_mask": done})

        print(
            "contract"
            f" num_envs={NUM_ENVS} threads={threads} obs=uint8_chw_{FRAME_STACK}x{OBSERVATION_SIZE}x{OBSERVATION_SIZE}"
            f" grayscale=true resize=area frame_skip={FRAME_SKIP} frame_stack={FRAME_STACK}"
            " maxpool=false obs_copy=safe_view autoreset=disabled",
            flush=True,
        )
        for repeat in range(1, repeats + 1):
            started = time.perf_counter()
            for _ in range(steps):
                _, _, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated
                if done.any():
                    env.reset(options={"reset_mask": done})
            elapsed = time.perf_counter() - started
            transitions = NUM_ENVS * steps
            rate = transitions / elapsed
            rates.append(rate)
            output_gbps = transitions * FRAME_STACK * OBSERVATION_SIZE**2 / elapsed / 1e9
            print(
                f"repeat={repeat} elapsed_seconds={elapsed:.3f}"
                f" policy_transitions_per_second={rate:.0f}"
                f" native_ticks_per_second={rate * FRAME_SKIP:.0f}"
                f" observation_gb_per_second={output_gbps:.3f}",
                flush=True,
            )
    finally:
        env.close()
    print(
        f"summary mean_policy_transitions_per_second={statistics.fmean(rates):.0f}"
        f" median={statistics.median(rates):.0f} best={max(rates):.0f}",
        flush=True,
    )
    return rates


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
