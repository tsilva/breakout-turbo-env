#!/usr/bin/env python3
"""Create and verify immutable release-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = "breakout-turbo-env"
REPOSITORY = "tsilva/breakout-turbo-env"
SCHEMA = 1
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_identity(version: str, commit: str, repository: str) -> None:
    if VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"invalid final release version: {version!r}")
    if SHA_RE.fullmatch(commit) is None:
        raise ValueError("commit must be a full lowercase 40-character SHA")
    if repository != REPOSITORY:
        raise ValueError(f"repository must be {REPOSITORY!r}")


def expected_distribution_names(version: str) -> set[str]:
    normalized = version.replace("-", "_")
    return {
        f"breakout_turbo_env-{normalized}-cp311-abi3-macosx_11_0_arm64.whl",
        f"breakout_turbo_env-{normalized}-cp311-abi3-manylinux_2_28_x86_64.whl",
        f"breakout_turbo_env-{normalized}.tar.gz",
    }


def candidate_files(root: Path) -> list[Path]:
    files = [path for path in (root / "dist").iterdir() if path.is_file()]
    sbom = root / "sbom.spdx.json"
    if not sbom.is_file():
        raise ValueError("candidate is missing sbom.spdx.json")
    return sorted([*files, sbom])


def validate_candidate_layout(root: Path) -> None:
    expected_root_files = {
        "SHA256SUMS",
        "release-manifest.json",
        "sbom.spdx.json",
    }
    actual_root_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_root_files != expected_root_files or actual_directories != {"dist"}:
        raise ValueError("candidate bundle contains an unexpected root entry")


def create_candidate(args: argparse.Namespace) -> None:
    validate_identity(args.version, args.commit, args.repository)
    root = args.candidate.resolve()
    distribution_names = {path.name for path in (root / "dist").iterdir()}
    expected = expected_distribution_names(args.version)
    if distribution_names != expected:
        raise ValueError(
            f"candidate distributions differ: expected {sorted(expected)}, "
            f"found {sorted(distribution_names)}"
        )
    files = candidate_files(root)
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        for path in files
    ]
    checksums = root / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{entry['sha256']}  {Path(entry['path']).name}\n"
            for entry in artifacts
            if str(entry["path"]).startswith("dist/")
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "kind": "release-candidate",
        "state": "built",
        "package": PACKAGE,
        "version": args.version,
        "repository": args.repository,
        "commit": args.commit,
        "builder": {
            "workflow": "Release candidate",
            "run_id": str(args.run_id),
            "run_attempt": str(args.run_attempt),
        },
        "artifacts": artifacts,
        "checksums": {"path": checksums.name, "sha256": sha256(checksums)},
    }
    write_json(root / "release-manifest.json", manifest)


def verify_candidate(
    root: Path,
    *,
    version: str,
    commit: str,
    repository: str,
    run_id: str | None = None,
) -> dict:
    validate_identity(version, commit, repository)
    root = root.resolve()
    manifest = read_json(root / "release-manifest.json")
    expected = {
        "schema": SCHEMA,
        "kind": "release-candidate",
        "state": "built",
        "package": PACKAGE,
        "version": version,
        "repository": repository,
        "commit": commit,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"candidate manifest {key!r} must equal {value!r}")
    builder = manifest.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("candidate manifest lacks builder identity")
    if run_id is not None and str(builder.get("run_id")) != str(run_id):
        raise ValueError("candidate was produced by a different workflow run")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate manifest artifacts must be a list")
    recorded_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("candidate artifact entry must be an object")
        relative = str(artifact.get("path", ""))
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe candidate artifact path: {relative!r}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"candidate artifact escapes its bundle: {relative}")
        if not path.is_file():
            raise ValueError(f"candidate artifact is missing: {relative}")
        if artifact.get("sha256") != sha256(path) or artifact.get("size") != path.stat().st_size:
            raise ValueError(f"candidate artifact digest or size changed: {relative}")
        recorded_paths.add(relative)
    if len(recorded_paths) != len(artifacts):
        raise ValueError("candidate manifest contains duplicate artifact paths")
    actual_paths = {str(path.relative_to(root)) for path in candidate_files(root)}
    if actual_paths != recorded_paths:
        raise ValueError("candidate contains unrecorded or missing artifacts")
    distribution_names = {
        Path(path).name for path in recorded_paths if path.startswith("dist/")
    }
    if distribution_names != expected_distribution_names(version):
        raise ValueError("candidate does not contain the exact supported distributions")
    checksums = manifest.get("checksums")
    checksum_path = root / "SHA256SUMS"
    if not isinstance(checksums, dict) or checksums != {
        "path": "SHA256SUMS",
        "sha256": sha256(checksum_path),
    }:
        raise ValueError("candidate checksum ledger changed")
    validate_candidate_layout(root)
    return manifest


def verify_command(args: argparse.Namespace) -> None:
    manifest = verify_candidate(
        args.candidate,
        version=args.version,
        commit=args.commit,
        repository=args.repository,
        run_id=args.run_id,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def pypi_files(version: str) -> dict[str, str]:
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{PACKAGE}/{version}/json", timeout=30
        ) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise
    return {
        item["filename"]: item["digests"]["sha256"]
        for item in data.get("urls", [])
        if not item.get("yanked", False)
    }


def pypi_status(args: argparse.Namespace) -> None:
    manifest = verify_candidate(
        args.candidate,
        version=args.version,
        commit=args.commit,
        repository=args.repository,
        run_id=args.run_id,
    )
    expected = {
        Path(entry["path"]).name: entry["sha256"]
        for entry in manifest["artifacts"]
        if entry["path"].startswith("dist/")
    }
    actual = pypi_files(args.version)
    if actual and actual != expected:
        raise SystemExit(
            "PyPI already contains a different or incomplete file set for this version"
        )
    complete = actual == expected
    if args.require_complete and not complete:
        raise SystemExit("the exact candidate is not yet complete on PyPI")
    status = "complete" if complete else "absent"
    result = {"status": status, "publish_needed": not complete, "files": actual}
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"publish_needed={'true' if not complete else 'false'}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    candidate = commands.add_parser("candidate")
    candidate.add_argument("--candidate", type=Path, required=True)
    candidate.add_argument("--version", required=True)
    candidate.add_argument("--commit", required=True)
    candidate.add_argument("--repository", default=REPOSITORY)
    candidate.add_argument("--run-id", required=True)
    candidate.add_argument("--run-attempt", required=True)
    candidate.set_defaults(func=create_candidate)

    for name, func in (("verify", verify_command), ("pypi-status", pypi_status)):
        command = commands.add_parser(name)
        command.add_argument("--candidate", type=Path, required=True)
        command.add_argument("--version", required=True)
        command.add_argument("--commit", required=True)
        command.add_argument("--repository", default=REPOSITORY)
        command.add_argument("--run-id")
        if name == "pypi-status":
            command.add_argument("--require-complete", action="store_true")
            command.add_argument("--github-output", type=Path)
        command.set_defaults(func=func)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
