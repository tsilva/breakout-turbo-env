---
name: build-release
description: Launch and monitor a breakout-turbo-env PyPI release. Use when the user says /build-release or $build-release, asks to cut, tag, or publish a release, asks whether a release reached PyPI, or asks for validated macOS arm64 and Linux x86_64 breakout-turbo-env wheels.
---

# Build Release

Use the repository-owned release flow and monitor it until the package is
visible on PyPI. The implementation lives in `scripts/release.py`, the
`Makefile` `release` target, and this skill's
`scripts/release_build.py` helper.

`make release` installs the locked release environment and runs the release
script. The script enforces a clean tree, a configured and synchronized
upstream, an unused PyPI version, synchronized versions, lock refresh, local
checks, release commit and tag creation, and an atomic push. The pushed tag
triggers `.github/workflows/release.yml`, which builds, audits, and publishes
macOS arm64 and Linux x86_64 wheels through PyPI trusted publishing.

For a new project, release the checked-in version when it is unused on PyPI.
After that version exists, default to the next patch release. Do not manually
upload to PyPI unless the user explicitly asks for recovery after the trusted
publishing path fails. Never print or commit PyPI tokens. Work on the current
branch unless the user explicitly requests another branch.

## Release flow

1. For the current version or default next release, run:

```bash
make release
```

For an explicit version or bump shape, install the locked environment and run
exactly one command:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra dev
scripts/release.py --to <version>
```

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra dev
scripts/release.py --part minor
```

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra dev
scripts/release.py --part major
```

2. Let the script own the release gates. If it fails, report the exact failing
stage and stop. Do not work around a dirty tree, unsynchronized upstream,
existing PyPI version, version mismatch, failed check, tag collision, or push
failure.

3. Capture the released tag and confirm it if necessary:

```bash
git describe --tags --exact-match HEAD
```

4. Monitor the tag-triggered workflow:

```bash
release_sha="$(git rev-list -n 1 v<version>)"
gh run list --workflow release.yml --commit "$release_sha" --limit 5 \
  --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
gh run watch <run-id> --exit-status
```

If that query is empty, list recent release runs and choose the matching tag:

```bash
gh run list --workflow release.yml --limit 10 \
  --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
```

Manual `workflow_dispatch` runs build and audit but never publish.

5. After success, poll PyPI:

```bash
.venv/bin/python - <<'PY'
import json
import time
import urllib.request

package = "breakout-turbo-env"
version = "<version>"
url = f"https://pypi.org/pypi/{package}/json"

for attempt in range(30):
    with urllib.request.urlopen(url, timeout=20) as response:
        data = json.load(response)
    files = data.get("releases", {}).get(version, [])
    if files:
        print(f"https://pypi.org/project/{package}/{version}/")
        for file in files:
            print(file["filename"])
        break
    print(f"waiting for {package} {version} ({attempt + 1}/30)")
    time.sleep(20)
else:
    raise SystemExit(f"{package} {version} did not appear on PyPI")
PY
```

PyPI indexing can lag briefly. If publishing fails, report the workflow URL and
failing step; do not attempt a manual upload without explicit approval.

## Diagnostics

Use the bundled helper only for narrow release diagnostics:

```bash
.venv/bin/python .codex/skills/build-release/scripts/release_build.py check-version
.venv/bin/python .codex/skills/build-release/scripts/release_build.py check-tools
.venv/bin/python .codex/skills/build-release/scripts/release_build.py latest-pypi
```

Useful workflow inspection commands:

```bash
gh run view <run-id> --web
gh run view <run-id> --log-failed
gh run view <run-id> --json url,status,conclusion,event,headBranch,headSha,displayTitle
```

## Final response

When publishing succeeds, lead with
`https://pypi.org/project/breakout-turbo-env/<version>/`. Report the tag,
GitHub Actions run URL and conclusion, and wheel filenames. On failure, report
the exact failed command, job, or step and the next safe recovery action.
