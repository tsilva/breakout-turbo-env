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


def build_parser(prog: str = "breakout-turbo-env benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Benchmark the fixed 16-env full-preprocessing policy path"
    )
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(NUM_ENVS, os.cpu_count() or NUM_ENVS),
    )
    return parser


def _print_human_header(
    *, steps: int, warmup: int, repeats: int, threads: int, obs: np.ndarray
) -> None:
    print(
        "config="
        f"num_envs={NUM_ENVS} steps={steps} repeats={repeats} warmup={warmup} "
        f"threads={threads} frame_skip={FRAME_SKIP} frame_stack={FRAME_STACK} "
        "grayscale=True crop=(0,0) obs_crop_mode=remove "
        f"resize={OBSERVATION_SIZE}x{OBSERVATION_SIZE} "
        "obs_resize_algorithm=area action_set=breakout "
        "actions=['noop','left','right'] action_seed=none obs_layout=chw maxpool_last_two=False "
        "obs_copy=safe_view info_filter=none autoreset=disabled",
        flush=True,
    )
    print("load_preflight=disabled", flush=True)
    print(
        f"obs_shape={tuple(obs.shape)} obs_dtype={obs.dtype} "
        f"obs_mib={obs.nbytes / (1024**2):.2f}",
        flush=True,
    )


def _print_human_run(*, index: int, elapsed: float, steps: int) -> float:
    batch_steps_per_sec = steps / elapsed
    env_steps_per_sec = batch_steps_per_sec * NUM_ENVS
    emulated_frames_per_sec = env_steps_per_sec * FRAME_SKIP
    print(
        f"run={index} elapsed_s={elapsed:.6f} "
        f"batch_steps_per_sec={batch_steps_per_sec:.1f} "
        f"env_steps_per_sec={env_steps_per_sec:.1f} "
        f"emulated_frames_per_sec={emulated_frames_per_sec:.1f}",
        flush=True,
    )
    return env_steps_per_sec


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
        obs, _ = env.reset()
        for _ in range(warmup):
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if done.any():
                env.reset(options={"reset_mask": done})

        _print_human_header(
            steps=steps,
            warmup=warmup,
            repeats=repeats,
            threads=threads,
            obs=obs,
        )
        for repeat in range(1, repeats + 1):
            started = time.perf_counter()
            for _ in range(steps):
                _, _, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated
                if done.any():
                    env.reset(options={"reset_mask": done})
            elapsed = time.perf_counter() - started
            rate = _print_human_run(index=repeat, elapsed=elapsed, steps=steps)
            rates.append(rate)
    finally:
        env.close()
    batch_steps_per_sec = [rate / NUM_ENVS for rate in rates]
    emulated_frames_per_sec = [rate * FRAME_SKIP for rate in rates]
    obs_buffer_gib_per_sec = (
        obs.nbytes * statistics.fmean(batch_steps_per_sec) / (1024**3)
    )
    print(
        "summary="
        f"env_steps_per_sec_mean={statistics.fmean(rates):.1f} "
        f"env_steps_per_sec_stdev={(statistics.stdev(rates) if len(rates) > 1 else 0.0):.1f} "
        f"best_env_steps_per_sec={max(rates):.1f} "
        f"emulated_frames_per_sec_mean={statistics.fmean(emulated_frames_per_sec):.1f} "
        f"obs_buffer_gib_per_sec={obs_buffer_gib_per_sec:.2f}",
        flush=True,
    )
    return rates


def main(argv=None, *, prog: str = "breakout-turbo-env benchmark") -> None:
    args = build_parser(prog=prog).parse_args(argv)
    run_benchmark(
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
