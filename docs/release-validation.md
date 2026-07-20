# Release validation

Releases use three isolated workflows. No pushed tag can publish a package.
Every transition is tied to one final version and one full commit SHA.

1. `Controlled Stable Retro parity` runs the live cartridge suite on the
   dedicated `breakout-parity` runner and emits only a JSON receipt. The ROM,
   save state, generated frames, and traces never enter workflow artifacts.
2. `Release candidate` accepts that successful run id, requires the candidate
   to be the current `main` commit, builds the exact supported distributions,
   smoke-installs them on Python 3.11 and 3.14, creates an SPDX SBOM and
   content-addressed manifest, and signs GitHub build-provenance and SBOM
   attestations.
3. `Publish approved release` downloads one named candidate run, revalidates
   its commit, parity receipt, file set, sizes, SHA-256 values, and GitHub
   attestations, then waits behind the protected `pypi` environment. Only the
   exact candidate may be sent through PyPI Trusted Publishing. A dedicated
   GitHub App creates the protected tag and immutable GitHub Release.

The state machine rejects a reused version unless PyPI already contains the
complete, byte-identical candidate. This permits safe recovery after a
post-upload interruption without enabling replacement or partial releases.

## Prepare a release commit

Use the repository command from clean, synchronized `main`:

```bash
uv sync --locked --extra dev
scripts/release.py prepare
```

The command promotes human-authored changelog notes and updates only the
version metadata files. It does not commit, tag, push, regenerate dependency
graphs, or publish. Dependency changes belong in a separate commit.

Local and CI checks use the committed locks and pinned Rust toolchain:

```bash
python scripts/lock.py
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --release
cargo test --locked --lib
pytest -m "not stable_retro"
```

## Controlled parity gate

The self-hosted runner must have the custom label `breakout-parity` and the
repository variable `PARITY_STABLE_RETRO_REPO` must point to its controlled
`stable-retro-turbo` checkout. Dispatch the workflow with:

- the exact candidate commit SHA;
- the version stored at that commit; and
- the exact reference checkout commit SHA.

The workflow refuses tracked modifications in the reference checkout. Its
required live suite compares native frames, policy observations, rewards,
score, lives, and terminal flags using runtime-generated actions. No reference
trace or restricted asset is retained.

## Candidate artifacts

Each candidate contains exactly:

- CPython 3.11+ ABI3, macOS 11+, ARM64 wheel;
- CPython 3.11+ ABI3, manylinux glibc 2.28+, x86-64 wheel;
- source archive;
- `SHA256SUMS`;
- `release-manifest.json`;
- `parity-receipt.json`; and
- `sbom.spdx.json`.

Linux wheels are built with `Cargo.lock --locked` inside a digest-pinned
official maturin image. The old network-fetched Rust installer is not used.
The source archive is available for inspection and reproducible builds but
does not extend the supported platform set.

## Repository controls

`main` intentionally permits direct maintainer pushes and does not require pull
requests or status checks. CI, package tests, dependency review, and CodeQL
still run as detection and feedback. Release publication remains separate: the
`pypi` environment accepts only the `main` branch, has a wait timer and required
approval, and disallows administrator bypass.

Before the first publication through the new flow, install a dedicated GitHub
App on this repository with metadata-read and contents-write access only. Store
`RELEASE_APP_ID` and `RELEASE_APP_PRIVATE_KEY` as `pypi` environment secrets,
allow only that App to create `v*` tags, configure the parity runner, and enable
immutable GitHub Releases after the historical evidence archive is stored in a
verified object-locked destination.

PyPI remains OIDC-only. Never add an API token or use a manual upload as a
shortcut around a failed gate.
