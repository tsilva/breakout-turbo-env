# breakout-turbo-env

<p align="center">
  <img src="./logo.png" alt="breakout-turbo-env logo" width="220">
  <br>
  <strong>🕹️ Blazing-fast, deterministic Breakout for Reinforcement Learning 🕹️</strong>
</p>

`breakout-turbo-env` is a deterministic Breakout environment for reinforcement-learning experiments, policy training, and learned-emulator datasets. It is aimed at researchers and engineers who need many identical game lanes, reproducible transitions, and observations that match a fixed Gymnasium vector-environment contract. Install it from source, build the native extension, then use the Python API, the interactive player, or the benchmark command.

The public API is `BreakoutVecEnv`. Rust owns fixed-point physics, parallel lane stepping, indexed rendering, frame skip, frame stacking, and observation preprocessing; Python provides the Gymnasium lifecycle and state/branching helpers.

## Install

Requirements: Python 3.11+, `uv`, and a Rust toolchain.

```bash
git clone https://github.com/tsilva/breakout-turbo-env.git
cd breakout-turbo-env
uv sync --extra dev
uv run maturin develop --release
```

## Use

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

The default observation batch is grayscale `uint8` in CHW order with shape
`(num_envs, 4, 84, 84)`. Actions are `0` (noop), `1` (left), and `2` (right).

Exact snapshots can be replayed or branched without mutating the live lanes:

```python
states = env.get_state()
branches = env.branch(states[:128], actions=(0, 1, 2))
```

## Play

```bash
uv run breakout-turbo-env play
```

Use Left/Right or A/D to move, Space or R to reset, P to pause, and Escape to quit. Choose a deterministic layout with `--layout checker`, `--layout tunnel`, or `--layout sparse`; `--show-obs` opens the four-frame processed observation stack in a second window.

## Benchmark

```bash
uv run breakout-turbo-env benchmark
```

The benchmark uses 16 environments with grayscale area resize, CHW observations, frame skip 4, frame stack 4, safe-view buffers, disabled max-pooling, and manual resets. Use `--steps`, `--warmup`, `--repeats`, and `--threads` to change measurement length or CPU concurrency. Its console output follows the sibling `SuperMarioBros-Nes-turbo` benchmark shape: `config=...`, `obs_shape=...`, one `run=...` line per repeat, and a `summary=...` line with environment steps, emulated frames, and observation-buffer throughput.

On the development Apple Silicon host, the optimized 8-thread path sustains roughly 277,000 policy transitions/s (1.11 million native ticks/s) and 7.8 GB/s of processed observation output after warmup.

## Commands

```bash
uv run pytest                         # run contract and regression tests
uv run breakout-turbo-env play --help      # list player options
uv run breakout-turbo-env benchmark --help # list benchmark options
```

## Release

With the locked development environment installed, launch the repository-owned
release flow:

```bash
make release
```

For a new PyPI project, this releases the current unused version. Afterwards it
defaults to the next patch version; use `scripts/release.py --part minor`,
`--part major`, or `--to <version>` for another release shape. The command
requires a clean tree synchronized with its upstream, validates locally, creates
the release commit when version files change, tags the release, and atomically
pushes the branch and tag.

The tag triggers `.github/workflows/release.yml`, which builds and audits
macOS arm64 and Linux x86_64 wheels, then publishes them to PyPI through trusted
publishing. Manual workflow runs build and audit artifacts without publishing.

## Notes

- The environment is manual-reset only. `AutoresetMode.DISABLED` is the only accepted mode, and a terminal lane must be included in an explicit `reset(options={"reset_mask": mask})` before its next `step`.
- There is no hidden reset randomness. Built-in starts are `full`, `checker`, `tunnel`, and `sparse`; selected lanes can be reset while other lanes continue unchanged.
- `obs_layout="chw"`, grayscale observations, area resize, `render_mode="rgb_array"`, and `maxpool_last_two=False` are fixed by the policy contract.
- `render()` returns the raw 96×96 RGB game frame. Training observations are preprocessed grayscale `uint8` frames and do not include the play window HUD.
- The Rust extension is built in release mode by `maturin`; `uv.lock` pins the Python dependency resolution window.

## Architecture

![breakout-turbo-env architecture](./architecture.png)

## License

[MIT](https://opensource.org/license/mit/)
