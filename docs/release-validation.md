# Release validation

Every published release is created from a clean, synchronized branch by the
repository-owned release command. The release path validates version metadata,
formatting, Rust compilation, unit and regression tests, live cartridge parity,
artifact contents, package rendering, and clean installation before publishing
through PyPI Trusted Publishing.

## Local gates

```bash
uv sync --frozen --extra dev
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo check --release
make test
make test-stable-retro
```

The live parity suite requires:

- a sibling `stable-retro-turbo` checkout, configurable with
  `STABLE_RETRO_REPO`;
- a locally imported lawful Breakout ROM; and
- the same generated action streams applied to both environments.

The suite compares native RGB frames, rewards, score, lives, and terminal flags
through forced collision cases and complete generated episodes. It fails rather
than skips during a release. No ROM, reference state, or recorded trace is
copied into the package.

## Published artifacts

The release workflow publishes exactly two binary wheels:

- CPython 3.11+ ABI3, macOS 11+, ARM64; and
- CPython 3.11+ ABI3, manylinux glibc 2.28+, x86-64.

It also publishes a source archive for inspection and reproducible builds. The
source archive does not extend the supported platform set. Every artifact must
pass metadata validation, content auditing, and SHA-256 checksum generation.
The supported wheels are smoke-installed with the oldest and newest supported
Python versions before publication.

PyPI receives artifacts through OpenID Connect trusted publishing. The tag also
creates a GitHub Release containing release notes, artifacts, and checksums.
