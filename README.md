<div align="center">
  <img src="./logo.png" alt="breakout-turbo-env logo" width="220" />
  <br />
  <strong>🕹️ Blazing-fast, deterministic Breakout for Reinforcement Learning 🕹️</strong>
</div>

breakout-turbo-env is a Python environment for running many deterministic Breakout games at once. It is for reinforcement-learning researchers and engineers who need reproducible transitions, fixed observations, and a Gymnasium vector-environment API. Build the native extension, create `BreakoutVecEnv`, and step it from Python; the repository also includes a playable window, benchmark, and two small training paths.

Fixed-point Rust physics owns game state and parallel stepping. Python exposes the Gymnasium lifecycle, rendering, snapshots, and branching helpers.

## Install

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a Rust toolchain.

```bash
git clone https://github.com/tsilva/breakout-turbo-env.git
cd breakout-turbo-env
uv sync --extra dev
uv run maturin develop --release
```

Use `BreakoutVecEnv` from the repository root or an environment where the extension has been installed.

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
```

## Commands

```bash
uv run breakout-turbo-env play                 # open the interactive player
uv run breakout-turbo-env benchmark            # measure the fixed 16-lane policy path
uv run pytest                                  # run Python contract and regression tests
cargo test --lib                               # run Rust library tests
uv run python train.py jerk                    # train a deterministic JERK action tape
uv run python train.py ppo                     # train a PPO policy
uv run python play.py jerk                     # replay the newest JERK policy
uv run python play.py ppo                      # replay the newest PPO policy
make release                                   # validate, tag, and publish a release
```

For player, benchmark, training, and replay options, append `--help` to the corresponding command.

## Notes

- The standard observation batch is grayscale `uint8`, CHW, and defaults to `(num_envs, 4, 84, 84)`. Actions are `0` (noop), `1` (left), and `2` (right).
- The environment is manual-reset only: after a terminal lane, call `reset(options={"reset_mask": mask})` before stepping that lane again. Built-in layouts are `full`, `checker`, `tunnel`, and `sparse`.
- `render()` returns the raw 96×96 RGB game frame; training observations use the processed grayscale stack. The interactive player accepts Left/Right or A/D, Space or R to reset, P to pause, and Escape to quit.
- Training outputs live in `runs/<algorithm>/<timestamp>/`. JERK policies use `policy.json`; PPO policies use `policy.npz`.
- `make release` requires a clean branch synchronized with its upstream. The release workflow builds and audits macOS arm64 and Linux x86_64 wheels before publishing to PyPI.

## Architecture

![breakout-turbo-env architecture](./architecture.png)

## License

[MIT](https://opensource.org/license/mit/)
