<div align="center">
  <img src="./logo.png" alt="breakout-turbo-env logo" width="220" />

  **🕹️ Blazing-fast, deterministic Breakout for Reinforcement Learning 🕹️**
</div>

breakout-turbo-env is a Python library for running many deterministic Breakout games at once. It gives reinforcement-learning researchers and engineers reproducible transitions, policy-ready observations, and a Gymnasium vector-environment API. Install the package, create `BreakoutVecEnv`, and step every game with one NumPy action batch.

Fixed-point Rust physics owns game state and parallel stepping. Python exposes the Gymnasium lifecycle, rendering, snapshots, and branching helpers.

## Install

Requires Python 3.11+.

```bash
pip install breakout-turbo-env
```

The core environment depends only on Gymnasium and NumPy. Install optional
tools explicitly when needed:

```bash
pip install "breakout-turbo-env[play]"   # interactive Pygame player
pip install "breakout-turbo-env[train]"  # local PPO training with PyTorch
```

To work from source, install [uv](https://docs.astral.sh/uv/) and a Rust toolchain, then run:

```bash
git clone https://github.com/tsilva/breakout-turbo-env.git
cd breakout-turbo-env
uv sync --extra dev --extra play --extra train
make develop-release
```

Import `BreakoutVecEnv` from the installed environment:

```python
import numpy as np
from breakout_turbo_env import BreakoutVecEnv

env = BreakoutVecEnv(num_envs=4096, num_threads=8)
obs, infos = env.reset()
obs, rewards, terminated, truncated, infos = env.step(
    np.zeros(env.num_envs, dtype=np.uint8)
)

done = terminated | truncated
if done.any():
    obs, reset_infos = env.reset(options={"reset_mask": done})

env.close()
```

## Commands

```bash
uv run --extra play breakout-turbo-env play    # open the interactive player
uv run --extra play breakout-turbo-env play --uncapped  # visible play without an FPS limit
uv run breakout-turbo-env benchmark            # measure the fixed 16-lane policy path
uv run python scripts/compare_stable_retro.py  # live corner + full-episode Stable Retro differential
uv run pytest                                  # run Python contract and regression tests
cargo test --lib                               # run Rust library tests
make test-stable-retro                         # require live frame-by-frame ROM parity
uv run python train.py jerk                    # train a deterministic JERK action tape
uv run --extra train python train.py ppo       # train a PPO policy
uv run --extra play python play.py jerk        # replay the newest JERK policy
uv run --extra play python play.py ppo         # replay the newest PPO policy
make release                                   # validate, tag, and publish a release
```

For player, benchmark, training, and replay options, append `--help` to the corresponding command.

## Notes

- The standard observation batch is grayscale `uint8`, CHW, and defaults to `(num_envs, 4, 84, 84)`. The native action contract is `0` (noop), `1` (FIRE), `2` (right), and `3` (left). Rewards match Stable Retro's Breakout scenario exactly: each reward is the score delta, using `7, 7, 4, 4, 1, 1` points from the top brick row to the bottom, with no life-loss penalty or board-clear bonus.
- The environment is manual-reset only: after a terminal lane, call `reset(options={"reset_mask": mask})` before stepping that lane again. Built-in layouts are `full`, `checker`, `tunnel`, and `sparse`.
- The `full` layout reproduces Stable Retro's `Breakout-Atari2600-v0` `Start` state: native 160×210 frames and frame aspect, Stella palette, 18×6 brick wall, 2×4 ball, initially 16×4 paddle that narrows after a ceiling return, five lives, FIRE serving, digital-paddle inertia, delayed hardware collision latches, breakthrough speed, score raster, and scanline clipping. The other layouts deliberately change only the brick mask for experiments.
- `render()` returns the native 160×210 RGB frame. Policy observations are resized directly from that native frame into the configured grayscale stack. The interactive player accepts Left/Right or A/D, Space to FIRE, R to reset, P to pause, and Escape to quit. Pass `--uncapped` for the fastest visible mode; headless stepping has no frame limiter.
- The Stable Retro differential suite requires a sibling `stable-retro-turbo` checkout with the locally installed Breakout ROM. Regular pytest runs it automatically when those prerequisites are available and otherwise reports it as skipped; `make test-stable-retro` requires the reference and fails if it is unavailable. The suite discovers coordinates from live RAM/rendered motion, then compares native RGB frames, rewards, score, lives, and terminal flags under forced corners plus generated tracking, predictive, and seeded-random action streams. It contains no recorded reference trace, and the ROM is never copied into or distributed with this package.
- PyPI provides wheels for macOS 11+ on Apple silicon and glibc 2.28+ Linux on x86-64. Other platforms require a source build.
- Training outputs live in `runs/<algorithm>/<timestamp>/`. JERK policies use `policy.json`; PPO policies use `policy.npz`.
- `make release` requires a clean branch synchronized with its upstream and a usable local Stable Retro cartridge reference; its local checks fail rather than skip live parity. The release workflow builds and audits macOS arm64 and Linux x86_64 wheels before publishing to PyPI.

## Architecture

![breakout-turbo-env architecture](./architecture.png)

## License

[MIT](https://opensource.org/license/mit/)
