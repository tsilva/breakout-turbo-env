## Outcome

Describe the user-visible result and why it belongs in breakout-turbo-env.

## Validation

- [ ] `uv run ruff check .`
- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test --lib`
- [ ] `uv run pytest -m "not stable_retro"`
- [ ] Live Stable Retro parity run, or not applicable

## Compatibility and provenance

- [ ] I documented any public API, snapshot, or behavior change.
- [ ] I did not add a ROM, save state, extracted game asset, or unlicensed material.
- [ ] The change preserves the supported platform boundary.
