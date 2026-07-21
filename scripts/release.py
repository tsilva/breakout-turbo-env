#!/usr/bin/env python3
"""Prepare a reviewable breakout-turbo-env release change in the local worktree."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)
RELEASE_NOTES = REPO_ROOT / "scripts" / "release_notes.py"
LOCK_SCRIPT = REPO_ROOT / "scripts" / "lock.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PACKAGE_NAME = "breakout-turbo-env"
ALLOWED_RELEASE_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "CHANGELOG.md",
    "CITATION.cff",
    "VERSION.txt",
    "pyproject.toml",
    "uv.lock",
}


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
        raise SystemExit(f"release preparation requires a clean tree:\n{status}")


def upstream_ref() -> str:
    try:
        return capture(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "current branch must have an upstream before release preparation"
        ) from error


def ensure_synced() -> str:
    upstream = upstream_ref()
    if "/" not in upstream:
        raise SystemExit(f"unexpected upstream ref: {upstream}")
    remote, _branch = upstream.split("/", 1)
    run(["git", "fetch", "--prune", "--tags", remote])
    left_right = capture(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"]
    )
    ahead, behind = [int(part) for part in left_right.split()]
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synced with {upstream}; "
            f"ahead={ahead} behind={behind}"
        )
    return upstream


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


def previous_release_version() -> str | None:
    try:
        tag = capture(["git", "describe", "--tags", "--abbrev=0"])
    except subprocess.CalledProcessError:
        return None
    if not tag.startswith("v"):
        raise SystemExit(f"latest release tag must start with 'v': {tag}")
    return tag.removeprefix("v")


def finalize_release_notes(version: str) -> None:
    command = [str(PYTHON), str(RELEASE_NOTES), "--version", version, "--finalize"]
    previous_version = previous_release_version()
    if previous_version is not None:
        command.extend(["--previous-version", previous_version])
    run(command)


def validate_release_notes(version: str) -> None:
    run([str(PYTHON), str(RELEASE_NOTES), "--version", version])


def read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        return tomllib.load(file)


def dependency_graph_snapshot() -> str:
    uv_lock = read_toml(REPO_ROOT / "uv.lock")
    cargo_lock = read_toml(REPO_ROOT / "Cargo.lock")
    normalized: dict[str, object] = {
        "uv_options": uv_lock.get("options"),
        "uv_manifest": uv_lock.get("manifest"),
        "uv_packages": copy.deepcopy(uv_lock.get("package", [])),
        "cargo_packages": copy.deepcopy(cargo_lock.get("package", [])),
    }
    for package in normalized["uv_packages"]:  # type: ignore[union-attr]
        if package.get("name") == PACKAGE_NAME:
            package.pop("version", None)
    for package in normalized["cargo_packages"]:  # type: ignore[union-attr]
        if package.get("name") == PACKAGE_NAME:
            package.pop("version", None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def ensure_dependency_graph_unchanged(before: str) -> None:
    after = dependency_graph_snapshot()
    if after != before:
        raise SystemExit(
            "release preparation changed a third-party lock graph; "
            "move dependency changes to a separate PR"
        )


def changed_paths() -> list[str]:
    changed: set[str] = set()
    for command in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        changed.update(filter(None, capture(command).splitlines()))
    return sorted(changed)


def ensure_only_release_files_changed() -> list[str]:
    changed = changed_paths()
    unexpected = sorted(set(changed) - ALLOWED_RELEASE_FILES)
    if unexpected:
        raise SystemExit(
            "release preparation changed unexpected files: " + ", ".join(unexpected)
        )
    if not changed:
        raise SystemExit("release preparation produced no reviewable changes")
    return changed


def run_checks() -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    run([str(PYTHON), str(LOCK_SCRIPT)], env=env)
    run([str(PYTHON), "-m", "ruff", "check", "."], env=env)
    run(["cargo", "fmt", "--check"])
    run(["cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings"])
    run(["cargo", "check", "--locked", "--release"])
    run(["cargo", "test", "--locked", "--lib"])
    run(
        [str(PYTHON), "-m", "maturin", "develop", "--release", "--locked"],
        env=env,
    )
    run([str(PYTHON), "-m", "pytest", "-m", "not stable_retro"], env=env)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare a release commit diff")
    group = prepare.add_mutually_exclusive_group()
    group.add_argument("--to", help="Exact release version, for example 0.3.6")
    group.add_argument(
        "--part",
        choices=("patch", "minor", "major"),
        help="Explicitly bump this component; otherwise use the next patch",
    )
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace) -> None:
    if not PYTHON.exists():
        raise SystemExit("expected .venv/bin/python; run `uv sync --locked --extra dev`")
    ensure_clean()
    upstream = ensure_synced()
    version = target_version(args)
    graph_before = dependency_graph_snapshot()
    finalize_release_notes(version)
    helper("bump-version", "--to", version, "--write")
    helper("check-version", "--version", version)
    helper("check-lock-policy")
    ensure_dependency_graph_unchanged(graph_before)
    validate_release_notes(version)
    changed = ensure_only_release_files_changed()
    run_checks()
    ensure_dependency_graph_unchanged(graph_before)

    print()
    print(f"Prepared v{version} from {upstream}; no commit, tag, push, or publish occurred.")
    print("Review and commit these files directly on main:")
    for path in changed:
        print(f"  {path}")
    print()
    run(["git", "diff", "--stat"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.command == "prepare":
        prepare(args)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
