#!/usr/bin/env python3
"""Validate, tag, and push a breakout-turbo-env release."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def run(
    args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    return subprocess.run(args, cwd=REPO_ROOT, env=env, check=True, text=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def ensure_clean() -> None:
    status = capture(["git", "status", "--short"])
    if status:
        raise SystemExit(f"release tree must be clean before releasing:\n{status}")


def upstream_ref() -> str:
    try:
        return capture(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "current branch must have an upstream before cutting a release"
        ) from error


def ensure_synced() -> tuple[str, str]:
    upstream = upstream_ref()
    if "/" not in upstream:
        raise SystemExit(f"unexpected upstream ref: {upstream}")
    remote, branch = upstream.split("/", 1)
    run(["git", "fetch", "--prune", "--tags", remote])
    left_right = capture(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"]
    )
    ahead, behind = [int(part) for part in left_right.split()]
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synced with {upstream} before release; "
            f"ahead={ahead} behind={behind}"
        )
    return remote, branch


def helper(*args: str) -> None:
    run([str(PYTHON), str(RELEASE_HELPER), *args])


def helper_capture(*args: str) -> str:
    return capture([str(PYTHON), str(RELEASE_HELPER), *args])


def target_version(args: argparse.Namespace) -> str:
    helper("check-version")
    if args.to:
        version = args.to
    elif args.part:
        version = helper_capture("bump-version", "--part", args.part).splitlines()[-1]
    else:
        version = helper_capture("resolve-version", "--part", "patch").splitlines()[-1]
    helper("check-pypi", "--version", version)
    return version


def refresh_locks() -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    run(["uv", "lock"], env=env)
    run(["cargo", "generate-lockfile"])


def run_checks(skip_checks: bool) -> None:
    if skip_checks:
        return
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    run([str(PYTHON), "-m", "ruff", "check", "."], env=env)
    run(["cargo", "fmt", "--check"])
    run(["cargo", "clippy", "--all-targets", "--", "-D", "warnings"])
    run(["cargo", "check", "--release"])
    run([str(PYTHON), "-m", "maturin", "develop", "--release"], env=env)
    env["BREAKOUT_REQUIRE_STABLE_RETRO"] = "1"
    run(["make", "test", "PYTHON=.venv/bin/python"], env=env)


def create_commit_and_tag(version: str) -> tuple[str, bool]:
    tag = f"v{version}"
    if (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", tag],
            cwd=REPO_ROOT,
        ).returncode
        == 0
    ):
        raise SystemExit(f"tag already exists locally: {tag}")
    run(
        [
            "git",
            "add",
            "VERSION.txt",
            "pyproject.toml",
            "Cargo.toml",
            "Cargo.lock",
            "CITATION.cff",
            "uv.lock",
        ]
    )
    has_version_changes = (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode
        != 0
    )
    if has_version_changes:
        run(["git", "commit", "-m", f"Release {tag}"])
    run(["git", "tag", tag, "HEAD"])
    return tag, has_version_changes


def push_release(remote: str, branch: str, tag: str) -> None:
    run(["git", "push", "--atomic", remote, f"HEAD:{branch}", tag])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--to", help="Exact release version, for example 0.1.1")
    group.add_argument(
        "--part",
        choices=("patch", "minor", "major"),
        help="Explicitly bump this component; otherwise release the current unused version",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip local cargo, maturin, and test gates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if not PYTHON.exists():
        raise SystemExit(
            "expected .venv/bin/python; run `uv sync --frozen --extra dev`"
        )
    ensure_clean()
    remote, branch = ensure_synced()
    version = target_version(args)
    helper("bump-version", "--to", version, "--write")
    refresh_locks()
    helper("check-version", "--version", version)
    run_checks(args.skip_checks)
    tag, committed = create_commit_and_tag(version)
    push_release(remote, branch, tag)
    print()
    if committed:
        print(f"Released {tag}: committed version files and pushed {branch} plus tag.")
    else:
        print(f"Released {tag}: tagged the existing {branch} commit and pushed the tag.")
    print("GitHub Actions will build, audit, and publish the release wheels.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
