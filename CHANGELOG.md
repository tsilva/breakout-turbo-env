# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the
major version is zero, documented public APIs may still change between minor
releases; changes will be recorded here.

## Unreleased

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

[0.3.0]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/tsilva/breakout-turbo-env/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/tsilva/breakout-turbo-env/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tsilva/breakout-turbo-env/releases/tag/v0.1.0
