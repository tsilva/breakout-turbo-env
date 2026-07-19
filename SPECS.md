## PROJECT PURPOSE

breakout-turbo-env is a Python library for reinforcement-learning researchers and engineers that runs many independent Breakout games through a Gymnasium-compatible vector-environment API while preserving deterministic, reproducible transitions.

## PROJECT REQUIREMENTS

- Given the same environment configuration, starting state, and action sequence, the environment must produce identical transition outputs.
- The vector environment must step every game from one batched action input and return observations and transition results that conform to its declared Gymnasium spaces.
- The environment must never automatically reset a terminal game; callers must be able to reset selected games without changing unselected games.
- Callers must be able to snapshot and restore game state for exact continuation and evaluate action branches from snapshots without mutating the source state.
- The environment must expose visual rendering separately from the processed observations supplied to policies.
- The human-facing render must reproduce Stable Retro's native Atari 2600 Breakout frame geometry and Stella palette while remaining separate from policy observations.
- Rewards must equal the per-step change in game score using Atari's row scoring, without independent life-loss or board-clear shaping.
- The interactive player must support both a configurable display-rate limit and visible uncapped play.
