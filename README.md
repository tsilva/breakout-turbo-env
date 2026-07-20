<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/breakout-turbo-env/main/logo.png" alt="breakout-turbo-env logo" width="220" />

  **🕹️ Blazing-fast, deterministic Breakout for Reinforcement Learning 🕹️**
</div>

breakout-turbo-env is a Python library for reinforcement-learning researchers
and engineers who need many reproducible Breakout games behind one Gymnasium
vector-environment API. Install it from PyPI, create `BreakoutVecEnv`, and step
every lane with one NumPy action batch.

Fixed-point Rust physics owns game state and parallel stepping. Python exposes
manual reset, policy-ready observations, native rendering, exact snapshots, and
side-effect-free action branching.

<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/breakout-turbo-env/main/demo.gif" alt="Native Breakout gameplay rendered by breakout-turbo-env" width="320" />
</div>

## Install

Requires Python 3.11+ on Apple-silicon macOS 11+ or x86-64 Linux with glibc
2.28+.

```bash
pip install breakout-turbo-env
```

Install optional tools only when needed:

```bash
pip install "breakout-turbo-env[play]"   # interactive Pygame player
pip install "breakout-turbo-env[train]"  # local PPO training with PyTorch
```

To work from source, install [uv](https://docs.astral.sh/uv/) and a Rust
toolchain, then run:

```bash
git clone https://github.com/tsilva/breakout-turbo-env.git
cd breakout-turbo-env
uv sync --locked --extra dev --extra play --extra train
make develop-release
```

## Use

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

Interesting live positions can be archived without advancing the game and
restored into any lane of the same environment:

```python
capture_mask = np.zeros(env.num_envs, dtype=np.bool_)
capture_mask[0] = True
captured = env.capture_snapshots(capture_mask)

restore_mask = np.zeros(env.num_envs, dtype=np.bool_)
restore_mask[3] = True
starts = [None] * env.num_envs
starts[3] = captured[0]
obs, infos = env.reset(
    options={"reset_mask": restore_mask, "snapshots": starts},
)
env.close()
```

Importing the package registers the Stable Retro-compatible
`Breakout-Atari2600-v0` environment. `BreakoutTurbo-v0` remains available as a
legacy native-action alias. The complete lifecycle, configuration, snapshot,
and branching contract is in the
[environment documentation](docs/environment.md).

Stable-Baselines3 users can wrap the already-vectorized environment with the
optional, explicitly auto-resetting adapter in
[`examples/sb3_manual_reset.py`](examples/sb3_manual_reset.py). SB3 remains a
separate install and is not part of the core dependency set.

## Commands

```bash
uv run --extra play breakout-turbo-env play       # open the player
uv run --extra play breakout-turbo-env play --uncapped
uv run breakout-turbo-env benchmark               # benchmark the policy path
uv run python scripts/compare_stable_retro.py     # run live differential checks
uv run ruff check .                               # lint Python
uv run pytest -m "not stable_retro"               # run regular Python tests
cargo test --locked --lib                         # run Rust tests
make test-stable-retro                            # require live cartridge parity
uv run python train.py jerk                       # train a deterministic action tape
uv run --extra train python train.py ppo          # train a PPO policy
uv run --extra play python play.py jerk           # replay the newest JERK policy
uv run --extra play python play.py ppo            # replay the newest PPO policy
```

Append `--help` to the player, benchmark, training, or replay command for its
options.

## Notes

- Native actions are `0` noop, `1` FIRE, `2` right, and `3` left. The default
  policy observation is grayscale `uint8`, CHW, and shaped
  `(num_envs, 4, 84, 84)`.
- Rewards are score deltas using Atari row scoring. There is no life-loss or
  board-clear shaping. The cartridge presents two walls: the first refills
  after the next paddle return, the second ends at score 864 without another
  refill, and only losing all five lives terminates the episode.
- Autoreset is disabled. Reset terminated lanes explicitly with a Boolean
  `reset_mask`; unselected lanes remain byte-exact.
- The canonical `Start` state targets Stable Retro's native 160×210 Atari Breakout frame,
  lifecycle, physics, raster, rewards, collision behavior, and public trajectory
  values. In particular, `ball_y` uses the Atari RAM convention where zero
  means the serve is waiting for FIRE. `render()` returns lane zero's RGB frame,
  while `render_lane(index)` selects any lane, separately from policy
  observations. The legacy start name `full` aliases `Start`.
- Live validation requires a separately obtained lawful ROM and a sibling
  `stable-retro-turbo` checkout. No ROM, save state, or recorded reference
  frame is distributed by this project.
- Only Apple-silicon macOS and x86-64 Linux are supported. See
  [support](SUPPORT.md), [benchmarking](docs/benchmarking.md), and
  [release validation](docs/release-validation.md) for exact boundaries.
- The project is a `0.x` community preview. Public changes are recorded in the
  [changelog](CHANGELOG.md). Serialized `get_state()` snapshots are portable
  only within the same package version and compatible configuration; live
  snapshot handles are session-local and intentionally not pickleable.

## Architecture

![breakout-turbo-env architecture](https://raw.githubusercontent.com/tsilva/breakout-turbo-env/main/architecture.png)

## License

[MIT](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for Atari,
Stable Retro, ROM, and trademark boundaries.
