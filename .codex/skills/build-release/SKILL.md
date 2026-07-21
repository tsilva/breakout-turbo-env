---
name: build-release
description: Prepare, build, approve, publish, and verify a breakout-turbo-env release through its gated candidate state machine.
---

# Build Release

Use only the repository-owned release state machine. A release has three
reviewable transitions: prepared release commit, attested candidate, and
approved publication. Never create or push a release tag by hand, never
manually upload to PyPI, and never substitute a different workflow artifact
after candidate validation.

## Preconditions

Before beginning, verify all of these controls rather than assuming them:

- the legacy tag-triggered `Release` workflow remains disabled;
- the unprotected direct-push `main` workflow and the protected `pypi`
  environment match `docs/release-validation.md`;
- immutable GitHub Releases are enabled only after the pre-hardening evidence
  archive is verified in object-locked storage; and
- PyPI Trusted Publishing is restricted to
  `.github/workflows/release.yml` and the `pypi` environment.

The release path does not require a self-hosted parity runner,
`PARITY_STABLE_RETRO_REPO`, release GitHub App secrets, or a tag ruleset.

If a retained precondition is absent, stop before publication and report it.
Do not weaken a retained gate to make progress.

## 1. Prepare the release commit

From a clean branch synchronized with its upstream:

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked --extra dev
scripts/release.py prepare
```

Use `prepare --to <version>` or `prepare --part minor|major|patch` only when the
user explicitly chose that target. The command may modify only changelog and
version metadata. It never commits, tags, pushes, resolves dependencies, or
publishes. Review the diff, commit it directly on `main`, and push only after
the local checks pass.

## 2. Build the attested candidate

After the release commit is pushed, capture the exact `main` SHA. Dispatch
`.github/workflows/release-build.yml` with that exact SHA. The workflow requires
the SHA to remain current `main`, checks that the PyPI version is unused, builds
the candidate, and attests its provenance and SBOM. Monitor it to completion
and record its run id.

Do not rebuild a single artifact locally. If a build or audit fails, fix the
cause in a new direct `main` commit and build a new candidate for the new SHA.

## 3. Approve and publish

Dispatch `.github/workflows/release.yml` with the candidate run id,
version, and commit SHA. Inspect the candidate manifest, checksums, SBOM, and
attestation summaries before approving the `pypi` environment deployment.
Monitor through PyPI verification, tag creation, and GitHub Release creation.

The workflow may resume only when PyPI's complete file set is byte-identical to
the candidate. A partial or conflicting version is a hard stop.

## 4. Verify externally

Confirm the exact wheel and source filenames at:

```text
https://pypi.org/project/breakout-turbo-env/<version>/
```

Then verify each downloaded distribution with:

```bash
gh attestation verify <distribution> --repo tsilva/breakout-turbo-env
```

Confirm the `v<version>` tag resolves to the candidate SHA and the GitHub
Release contains all six candidate files. Report the candidate and publish
workflow URLs in the final response.
