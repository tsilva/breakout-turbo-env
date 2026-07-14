## PROJECT PURPOSE

breakout-turbo-env is a Python library for reinforcement-learning researchers and engineers that runs many independent Breakout games through a Gymnasium-compatible vector-environment API while preserving deterministic, reproducible transitions.

## PROJECT REQUIREMENTS

- Given the same environment configuration, starting state, and action sequence, the environment must produce identical transition outputs.
- The vector environment must step every game from one batched action input and return observations and transition results that conform to its declared Gymnasium spaces.
- The environment must never automatically reset a terminal game; callers must be able to reset selected games without changing unselected games.
- Callers must be able to snapshot and restore game state for exact continuation and evaluate action branches from snapshots without mutating the source state.
- The environment must expose visual rendering separately from the processed observations supplied to policies.
