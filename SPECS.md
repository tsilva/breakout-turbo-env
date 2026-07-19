## PROJECT PURPOSE

breakout-turbo-env is a Python library for reinforcement-learning researchers and engineers that runs many independent Breakout games through a Gymnasium-compatible vector-environment API while preserving deterministic, reproducible transitions.

## PROJECT REQUIREMENTS

- Given the same environment configuration, starting state, and action sequence, the environment must produce identical transition outputs.
- The vector environment must step every game from one batched action input and return observations and transition results that conform to its declared Gymnasium spaces.
- The environment must never automatically reset a terminal game; callers must be able to reset selected games without changing unselected games.
- Callers must be able to snapshot and restore game state for exact continuation and evaluate action branches from snapshots without mutating the source state.
- The environment must expose visual rendering separately from the processed observations supplied to policies.
- The human-facing render must reproduce Stable Retro's native Atari 2600 Breakout frame geometry and Stella palette while remaining separate from policy observations.
- The `full` start must reproduce Stable Retro's `Breakout-Atari2600-v0` `Start` state at 160×210 pixels, including its 18×6 brick wall, 2×4 ball, 16×4 paddle, five-life counter, FIRE-gated serves, paddle inertia, delayed collision latches, score raster, scanline priority, and wall/corner behavior.
- Native actions must be `0` noop, `1` FIRE, `2` right, and `3` left.
- Policy observations must be derived from the same native 160×210 indexed frame returned by the renderer before grayscale resizing and frame stacking.
- Rewards must equal the per-step change in game score using Atari's row scoring, without independent life-loss or board-clear shaping.
- Clearing the brick wall must not independently terminate an episode; terminal state follows the five-life Atari game lifecycle.
- The interactive player must support both a configurable display-rate limit and visible uncapped play.
