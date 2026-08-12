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
uv sync --locked --extra dev --extra play
make develop-release
```

## Required checks

```bash
uv run ruff check .
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --lib
uv run pytest -m "not stable_retro"
```

Changes that can affect the `Start` state's physics, rewards, lifecycle,
observations, or native rendering must pass both the sibling-fork differential
and the exact original-Stable-Retro authority suite:

```bash
make test-stable-retro
make test-semantic-oracle
```

The TurboBench suite pins original `stable-retro==1.0.1` and compares scalar
and four-lane runs for 4,096 seeded transitions, including public native RGB
frames, processed observations, rewards, termination and truncation, selected
info, lane resets, and snapshot continuation. The sibling-fork suite remains a
useful secondary regression check. Both require a locally configured lawful
Breakout ROM. Checkout receipts are development evidence. After publishing the
candidate, regenerate the oracle with `breakout-turbo-env@VERSION` and verify
that PyPI-candidate receipt outside the repository:

```bash
make verify-semantic-oracle ORACLE_RECEIPT=/external/evidence/receipt
```

See
[`docs/release-validation.md`](docs/release-validation.md).

## Pull requests

Keep each pull request focused. Explain the user-visible result, tests run, and
any compatibility impact. Add or update tests for behavior changes and update
documentation when the public API changes. By contributing, you agree that your
contribution is distributed under this repository's MIT license.
