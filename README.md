<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/breakout-turbo-env/main/logo.png" alt="breakout-turbo-env logo" width="220" />

  **🕹️ Blazing-fast, deterministic Breakout for Reinforcement Learning 🕹️**
</div>

breakout-turbo-env is a Python library for reinforcement-learning researchers
and engineers who need many reproducible Breakout games behind one Gymnasium
vector-environment API. Add it to a uv project from PyPI, create
`BreakoutVecEnv`, and step every lane with one NumPy action batch.

Fixed-point Rust physics owns game state and parallel stepping. Python exposes
manual reset, policy-ready observations, native rendering, exact snapshots, and
side-effect-free action branching.

<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/breakout-turbo-env/main/demo.gif" alt="Native Breakout gameplay rendered by breakout-turbo-env" width="320" />
</div>

## Install

Requires Python 3.11+ on Apple-silicon macOS 11+ or x86-64 Linux with glibc
2.28+.

Install [uv](https://docs.astral.sh/uv/), then add the library to your project:

```bash
uv add breakout-turbo-env
```

Choose the corresponding requirement instead when you need an optional tool:

```bash
uv add "breakout-turbo-env[play]"  # interactive Pygame player
```

To work from source, also install a Rust toolchain, then run:

```bash
git clone https://github.com/tsilva/breakout-turbo-env.git
cd breakout-turbo-env
uv sync --frozen --extra dev --extra play
make develop-release
```

## Use

```python
import gymnasium as gym
import numpy as np

env = gym.make_vec(
    "breakout_turbo_env:Breakout-Turbo-v0",
    game="Breakout-Atari2600-v0",
    num_envs=4096,
    num_threads=8,
)
obs, infos = env.reset()
obs, rewards, terminated, truncated, infos = env.step(
    np.zeros(env.num_envs, dtype=np.uint8)
)

done = terminated | truncated
if done.any():
    obs, reset_infos = env.reset(options={"reset_mask": done})
```

The module-qualified ID imports the package and registers the factory. This ID
is vector-only and requires an explicit `game`; `BreakoutVecEnv` remains
available for direct use.

## Train with GradLab

Training recipes and implementations live in
[GradLab](https://github.com/tsilva/rlab), keeping this repository focused on
the environment. Run a published recipe from any directory without installing
GradLab or cloning either repository.

For the default high-throughput PPO recipe:

```bash
uvx gradlab@0.1.1 train Breakout-Atari2600-v0/ppo
```

For PPO with learning-rate decay and KL-based update stopping:

```bash
uvx gradlab@0.1.1 train Breakout-Atari2600-v0/ppo-stable-updates
```

Breakout Turbo is ROM-free, so neither command needs a ROM path or registration.
GradLab shows live progress, writes a playable `final_model.zip` below `./runs`,
and prints the matching version-pinned `uvx ... play` command when training
finishes or is stopped safely. Local runs disable W&B and checkpoint evaluation
by default, so they cannot establish acceptance or promotion. These are
full-cap research recipes rather than short timed demos.

## Turbo Vector API v1

`BreakoutVecEnv` implements the strict Turbo Vector API v1:

- `metadata["turbo_api_version"]` is `1`, and `metadata["render_modes"]`
  advertises `rgb_array`.
- Immutable `capabilities` and `signal_schema` declarations describe supported
  features and the dtype, shape, and reset/step availability of every signal.
- `buttons`, `action_mode`, `action_preset`, `action_table`,
  `action_meanings`, and `action_table_hash` expose the resolved action
  semantics without provider-specific probing.
- `state_catalog` is an immutable ordered tuple. Callers select reset states
  with an `int32` `state_indices` array and inspect the read-only active indices
  with `active_state_indices()`; state sampling and lane routing remain
  caller-owned.
- `observation_ownership` and `observation_buffer_depth` declare the exact
  lifetime of returned observations. Rendering is opt-in: with
  `render_mode="rgb_array"`, `render_lane(index)` renders one lane,
  `get_images()` renders all lanes, and `render()` renders lane zero. With the
  default `render_mode=None`, the first two methods return `None` and
  `get_images()` returns one `None` entry per lane.

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

Importing the package also preserves the Stable Retro-compatible
`Breakout-Atari2600-v0` vector ID. The complete lifecycle, configuration,
snapshot, and branching contract is in the
[environment documentation](docs/environment.md).

Stable-Baselines3 users can wrap the already-vectorized environment with the
optional, explicitly auto-resetting adapter described in the
[environment documentation](docs/environment.md#stable-baselines3). SB3
remains a separate install and is not part of the core dependency set.

## Commands

```bash
uv run --frozen --extra play breakout-turbo-env play       # open the player
uv run --frozen --extra play breakout-turbo-env play --uncapped
uv run --frozen breakout-turbo-env benchmark               # benchmark the policy path
uv run --frozen python scripts/compare_stable_retro.py     # run live differential checks
uv run --frozen ruff check .                               # lint Python
uv run --frozen pytest -m "not stable_retro"               # run regular Python tests
cargo test --locked --lib                                  # run Rust tests
make test-stable-retro                                     # require live cartridge parity
make test-semantic-oracle                                  # compare to original Stable Retro authority
```

Append `--help` to the player or benchmark command for its options.

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
- `noop_reset_max=N` samples `1..N` seeded raw-frame noops for each static
  reset, matching the conventional Atari reset distribution. FIRE is not
  issued automatically: `use_fire_reset` remains unavailable and the policy
  must start each serve.
- The canonical `Start` state targets Stable Retro's native 160×210 Atari Breakout frame,
  lifecycle, physics, raster, rewards, collision behavior, and public trajectory
  values. In particular, `ball_y` uses the Atari RAM convention where zero
  means the serve is waiting for FIRE. Opt into raw frames with
  `render_mode="rgb_array"`; `render()` then returns lane zero's RGB frame while
  `render_lane(index)` selects any lane, separately from policy observations.
- Live validation requires a separately obtained lawful ROM. The TurboBench
  semantic oracle pins original `stable-retro==1.0.1` as the authority; the
  sibling `stable-retro-turbo` differential remains a secondary regression
  check. No ROM, save state, or recorded reference frame is distributed by
  this project.
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
