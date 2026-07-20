---
name: build-release
description: Prepare, build, approve, publish, and verify a breakout-turbo-env release through its gated candidate state machine.
---

# Build Release

Use only the repository-owned release state machine. A release has four
reviewable transitions: prepared release commit, controlled parity receipt, attested
candidate, and approved publication. Never create or push a release tag by
hand, never manually upload to PyPI, and never substitute a different workflow
artifact after candidate validation.

## Preconditions

Before beginning, verify all of these controls rather than assuming them:

- the legacy tag-triggered `Release` workflow remains disabled;
- the unprotected direct-push `main` workflow and the protected `pypi`
  environment match `docs/release-validation.md`;
- the `breakout-parity` runner and `PARITY_STABLE_RETRO_REPO` variable exist;
- `RELEASE_APP_ID` and `RELEASE_APP_PRIVATE_KEY` are `pypi` environment secrets
  for a repository-only GitHub App with metadata-read and contents-write access,
  allowed to create `v*` tags and releases;
- immutable GitHub Releases are enabled only after the pre-hardening evidence
  archive is verified in object-locked storage; and
- PyPI Trusted Publishing is restricted to
  `.github/workflows/release-publish.yml` and the `pypi` environment.

If any precondition is absent, stop before publication and report it. Do not
weaken a gate to make progress.

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

## 2. Run controlled parity

After the release commit is pushed, capture the exact `main` SHA and the controlled
reference checkout SHA. Dispatch `.github/workflows/parity.yml` with the SHA,
version, and reference SHA. Monitor the run to completion. The resulting
artifact must be named `parity-receipt-<40-character-sha>` and must contain only
`parity-receipt.json`.

Do not copy a ROM, save state, frame, trace, or parity workspace into a GitHub
artifact or log.

## 3. Build the attested candidate

Dispatch `.github/workflows/release-build.yml` with that exact SHA and the
successful parity run id. The workflow requires the SHA to remain current
`main`, checks that the PyPI version is unused, and builds the candidate.
Monitor it to completion and record its run id.

Do not rebuild a single artifact locally. If a build or audit fails, fix the
cause in a new direct `main` commit, rerun parity for the new SHA, and build a
new candidate.

## 4. Approve and publish

Dispatch `.github/workflows/release-publish.yml` with the candidate run id,
version, and commit SHA. Inspect the candidate manifest, parity receipt,
checksums, SBOM, and attestation summaries before approving the `pypi`
environment deployment. Monitor through PyPI verification, protected tag
creation, and GitHub Release creation.

The workflow may resume only when PyPI's complete file set is byte-identical to
the candidate. A partial or conflicting version is a hard stop.

## 5. Verify externally

Confirm the exact wheel and source filenames at:

```text
https://pypi.org/project/breakout-turbo-env/<version>/
```

Then verify each downloaded distribution with:

```bash
gh attestation verify <distribution> --repo tsilva/breakout-turbo-env
```

Confirm the `v<version>` tag resolves to the candidate SHA and the immutable
GitHub Release contains all seven candidate files. Report the parity, candidate,
and publish workflow URLs in the final response.
