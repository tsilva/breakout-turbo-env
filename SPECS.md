## PROJECT PURPOSE

`env-BreakoutAtari2600-turbo-native` is a Python library for reinforcement-learning researchers and engineers that runs many independent Breakout games through a Gymnasium-compatible vector-environment API while preserving deterministic, reproducible transitions.

## PROJECT REQUIREMENTS

- Use `env-BreakoutAtari2600-turbo-native` as the project and GitHub repository name, `env-breakoutatari2600-turbo-native` as the Python distribution name, and `env_breakoutatari2600_turbo_native` as the public Python import package; current project-owned identities must not use any former project, distribution, import, or command identifier.
- Given the same environment configuration, starting state, and action sequence, the environment must produce identical transition outputs.
- The vector environment must step every game from one batched action input and return observations and transition results that conform to its declared Gymnasium spaces.
- The environment must never automatically reset a terminal game; callers must be able to reset selected games without changing unselected games.
- Callers must be able to snapshot and restore game state for exact continuation and evaluate action branches from snapshots without mutating the source state.
- The environment must expose visual rendering separately from the processed observations supplied to policies.
- The human-facing render must reproduce Stable Retro's native Atari 2600 Breakout frame geometry and Stella palette while remaining separate from policy observations.
- The canonical `Breakout-Atari2600-v0` `Start` state must reproduce Stable Retro's state at 160×210 pixels, including its 18×6 brick wall, 2×4 ball, initially 16×4 ceiling-narrowing paddle, five-life counter, FIRE-gated serves, paddle inertia, delayed collision latches, breakthrough speed, score raster, scanline priority, and wall/corner behavior.
- Callers must be able to replace Stable Retro Turbo with `env-BreakoutAtari2600-turbo-native` for `Breakout-Atari2600-v0` without changing the game, start, observation, action, reward, reset, termination, truncation, or information contract; equivalent seeds may produce different stochastic traces but not different semantics.
- Every externally observable canonical-start trajectory detail must match the same Atari ROM trajectory running through the pinned original Stable Retro release under equivalent configuration and actions, including canonical RGB frame geometry and content after transport normalization, policy observations, rewards, score, lives, termination, truncation, and shared info values; Stable Retro Turbo remains a secondary compatibility target, and changes must be acceptance-tested side by side against both providers.
- Native actions must be `0` noop, `1` FIRE, `2` right, and `3` left.
- `use_restricted_actions` must accept the game-owned `simple` table and exact caller-supplied subsets or reorderings of reproducible Atari button commands, while rejecting commands the native kernel cannot reproduce.
- Policy observations must be derived from the same native 160×210 indexed frame returned by the renderer before grayscale resizing and frame stacking.
- Rewards must equal the per-step change in game score using Atari's row scoring, without independent life-loss or board-clear shaping.
- Clearing the first brick wall must refill the same layout one native frame after the next paddle return; clearing the second wall must leave the board permanently empty at the Atari top score of 864, and neither clear may terminate the episode before the five-life lifecycle ends.
- The interactive player must support both a configurable display-rate limit and visible uncapped play.
- Supported distributions must be limited to Apple-silicon macOS and x86-64 Linux.
