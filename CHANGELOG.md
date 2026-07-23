# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the
major version is zero, documented public APIs may still change between minor
releases; changes will be recorded here.

## Unreleased

### Added

- Added seeded `noop_reset_max` support for static resets, using raw emulator
  frames with lane-isolated masked-reset random streams and reset info counts.
  Automatic FIRE reset remains intentionally unavailable.

## [0.4.0] - 2026-07-21

### Added

- Added game-owned preset and inline exact action tables under
  `use_restricted_actions`, loaded from packaged `metadata.json` with
  validated Atari controller labels and deterministic semantic hashes.
- Added an optional Stable-Baselines3 adapter and example that preserves
  terminal observations while resetting only completed lanes.
- Added CodeQL coverage for Python, Rust, and GitHub Actions, plus SPDX SBOM
  and signed build-provenance attestations for release distributions.

### Changed

- Replaced tag-triggered publication with a content-addressed release
  candidate, protected manual approval, and GitHub Actions tag/release
  authority.
- Preserved the existing PyPI Trusted Publisher identity through the
  `.github/workflows/release.yml` publication workflow.
- Made Python and Rust lock enforcement hermetic and replaced the Linux
  network bootstrap with a digest-pinned official maturin builder.
- Made clean-install smoke checks compare canonical paths so macOS `/var` and
  `/private/var` aliases cannot cause false failures.

## [0.3.5] - 2026-07-20

### Added

- Added `render_lane(index)` for inspecting any vector-environment lane without
  advancing game state; `render()` remains the lane-zero Gymnasium interface.

## [0.3.4] - 2026-07-20

### Changed

- Matched Stable Retro's RGB565 luminance and resize behavior when deriving
  grayscale policy observations from native frames.

## [0.3.3] - 2026-07-20

### Added

- Added reusable, per-lane live snapshot handles through
  `capture_snapshots(mask)` and mixed snapshot/catalog restoration through
  masked `reset()`, including exact cross-lane fan-out without advancing
  emulation.

## [0.3.2] - 2026-07-20

### Changed

- Matched the cartridge's two-wall lifecycle: delayed first-wall refill,
  864-point maximum, permanent empty board after wall two, and lives-only
  episode termination.
- Added `walls_cleared` and a lossless high-word companion for the 108-bit
  brick mask; snapshots now use the phase-aware `BTO10` format.

## [0.3.1] - 2026-07-19

### Changed

- Made public `ball_y` match Stable Retro's Atari RAM value, including zero
  while waiting for FIRE, and removed the redundant public `awaiting_fire`
  info field.

## [0.3.0] - 2026-07-19

### Added

- Community contribution, conduct, security, support, citation, and legal
  documentation.
- Pull-request CI, supported-Python validation, release artifact checksums, and
  source distributions.
- Public environment, benchmark, and release-validation documentation.
- GitHub release notes and clean-install artifact smoke tests.
- A reproducible matched Stable Retro benchmark harness and v0.3.0 evidence
  report.
- Patched PyO3, PyTorch, and pytest dependency lines for a clean community
  security baseline.

### Changed

- Declared Apple-silicon macOS and x86-64 Linux as the only supported
  distribution platforms.
- Expanded package metadata and made README images render correctly on PyPI.

## [0.2.5] - 2026-07-19

- Added live frame-by-frame Stable Retro parity coverage and made it a local
  release requirement.
- Completed Atari collision, corner, breakthrough-speed, and scanline parity.

## [0.2.4] - 2026-07-19

- Matched native Atari frame geometry, presentation, physics, and rewards.

## [0.2.3] - 2026-07-19

- Added Atari-native rendering and reward behavior.

## [0.2.2] - 2026-07-15

- Corrected info presence masks.

## [0.2.1] - 2026-07-15

- Kept player and training dependencies optional.

## [0.2.0] - 2026-07-14

- Established the manual-reset Gymnasium vector-environment contract.

## [0.1.0] - 2026-07-12

- Initial public release.

[0.4.0]: https://github.com/tsilva/breakout-turbo-env/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/tsilva/breakout-turbo-env/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/tsilva/breakout-turbo-env/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/tsilva/breakout-turbo-env/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/tsilva/breakout-turbo-env/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/tsilva/breakout-turbo-env/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/tsilva/breakout-turbo-env/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tsilva/breakout-turbo-env/releases/tag/v0.1.0
