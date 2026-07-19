# Contributing

Thanks for helping make breakout-turbo-env more useful and trustworthy for the
reinforcement-learning community.

## Before opening a change

- Search existing issues and discussions first.
- Use an issue for a bug report or proposed user-facing change.
- Do not submit ROMs, extracted game assets, reference frames, or save states.
- Keep the supported distribution boundary to Apple-silicon macOS and x86-64
  Linux.

## Development setup

Install [uv](https://docs.astral.sh/uv/) and a Rust toolchain, then run:

```bash
git clone https://github.com/tsilva/breakout-turbo-env.git
cd breakout-turbo-env
uv sync --frozen --extra dev --extra play --extra train
make develop-release
```

## Required checks

```bash
uv run ruff check .
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
uv run pytest -m "not stable_retro"
```

Changes that can affect the `full` start's physics, rewards, lifecycle,
observations, or native rendering must also pass:

```bash
make test-stable-retro
```

That differential suite requires a sibling `stable-retro-turbo` checkout and a
locally configured Breakout ROM. See
[`docs/release-validation.md`](docs/release-validation.md).

## Pull requests

Keep each pull request focused. Explain the user-visible result, tests run, and
any compatibility impact. Add or update tests for behavior changes and update
documentation when the public API changes. By contributing, you agree that your
contribution is distributed under this repository's MIT license.
