# breakout-turbo-env

Deterministic, fixed-point Breakout designed for fast vectorized policy training
and learned-emulator datasets.

The hot path is Rust. It owns physics, indexed rendering, preprocessing, frame
skip, frame stacking, lane resets, and observation buffers. The public API is a
Gymnasium `VectorEnv` with the manual lifecycle expected by `rlab`.

```python
import numpy as np
from breakout_turbo_env import BreakoutVecEnv

env = BreakoutVecEnv(
    num_envs=4096,
    num_threads=8,
    obs_resize=(84, 84),
    frame_skip=4,
    frame_stack=4,
    obs_copy="safe_view",
    info_filter="none",
)
obs, infos = env.reset()
obs, rewards, terminated, truncated, infos = env.step(
    np.zeros(env.num_envs, dtype=np.uint8)
)

done = terminated | truncated
if done.any():
    obs, reset_infos = env.reset(options={"reset_mask": done})
```

Observations are always grayscale `uint8` in CHW order. The default batch shape
is `(num_envs, 4, 84, 84)`. Actions are `0=noop`, `1=left`, and `2=right`.

There is deliberately no autoreset implementation and no max-pooling path.
`autoreset_mode` only accepts `AutoresetMode.DISABLED`, and
`maxpool_last_two` only accepts `False`.

## Exact states and branching

```python
states = env.get_state()
branches = env.branch(states[:128], actions=(0, 1, 2))
```

`branch` returns the Cartesian product of source states and actions, including
next states, observations, rewards, terminal flags, actions, source indices,
and ground-truth state signals. It does not mutate the live environment.

Selected lanes can be reset even while active. Unselected lanes preserve their
physics state and observation stack exactly:

```python
mask = np.array([True, False, True, False], dtype=np.bool_)
obs, infos = env.reset(
    options={"reset_mask": mask, "start_ids": ["checker", None, "tunnel", None]}
)
```

Built-in deterministic starts are `full`, `checker`, `tunnel`, and `sparse`.
There is no hidden reset randomness.

## Play manually

```bash
uv run breakout-turbo-play
```

Use Left/Right or A/D to move, Space or R to reset, P to pause, and Escape to
quit. Select another deterministic board with `--layout checker`, `--layout
tunnel`, or `--layout sparse`. Use `--scale 4` for a smaller window.

The play window has a HUD above the game showing score, lives, remaining
bricks, and pause state. This HUD is play-only and never enters raw or processed
training observations.

At every natural episode end, the player prints the outcome, score, return,
lives, bricks cleared, native ticks, displayed steps, and elapsed time.

Add `--show-obs` to open a second window containing the exact four grayscale
84x84 policy frames, tiled oldest-to-newest:

```bash
uv run breakout-turbo-play --show-obs
```

## Benchmark

Run the fixed 16-environment benchmark with grayscale area resize, CHW layout,
frame skip 4, frame stack 4, safe-view buffers, disabled max-pooling, and manual
autoreset:

```bash
uv run breakout-turbo-benchmark
```

Use `--steps`, `--warmup`, `--repeats`, and `--threads` to control measurement
length and CPU concurrency without changing the 16-environment preprocessing
contract.

On the development Apple Silicon host, the default three-repeat run measured a
mean of about 79,000 policy transitions/s (316,000 native ticks/s) while
returning roughly 2.23 GB/s of processed observation bytes.

## Development

```bash
uv sync --extra dev
uv run maturin develop --release
uv run pytest
uv run breakout-turbo-play --help
uv run python scripts/benchmark.py
```

On the development Apple Silicon host, a release build with 4,096 lanes, 12
threads, frame skip 4, four stacked 84x84 frames, safe-view observations, and
no info materialization sustained about 138,000 policy transitions/s (551,000
native physics ticks/s). Treat this as a local smoke result, not a portable
hardware guarantee; the benchmark script prints both rates for every run.
