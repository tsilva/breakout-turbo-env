#!/usr/bin/env python3
"""Deterministic helpers for breakout-turbo-env release builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[4]
VERSION_PATH = REPO_ROOT / "VERSION.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CARGO_TOML = REPO_ROOT / "Cargo.toml"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PACKAGE_NAME = "breakout-turbo-env"
IMPORT_NAME = "breakout_turbo_env"
EXTENSION_NAME = "_breakout_turbo"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+|\.dev\d+)?$")


def read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        return tomllib.load(file)


def read_version() -> str:
    return VERSION_PATH.read_text(encoding="utf-8").strip()


def pyproject_name() -> str:
    return str(read_toml(PYPROJECT)["project"]["name"])  # type: ignore[index]


def pyproject_version() -> str:
    return str(read_toml(PYPROJECT)["project"]["version"])  # type: ignore[index]


def section_version(path: Path, section: str, *, package_name: str | None = None) -> str:
    current_section: str | None = None
    matching_package = package_name is None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            current_section = stripped[2:-2].strip()
            matching_package = package_name is None
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            matching_package = package_name is None
            continue
        if current_section != section:
            continue
        if stripped.startswith("name = ") and package_name is not None:
            matching_package = stripped.split("=", 1)[1].strip().strip('"') == package_name
            continue
        if matching_package and stripped.startswith("version = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"could not find version in [{section}] of {path}")


def cargo_version() -> str:
    return section_version(CARGO_TOML, "package")


def cargo_lock_version() -> str:
    return section_version(CARGO_LOCK, "package", package_name=PACKAGE_NAME)


def validate_version(version: str) -> None:
    if VERSION_RE.fullmatch(version) is None:
        raise SystemExit(f"unsupported version format: {version!r}")


def split_release(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise SystemExit(
            f"cannot compute a major/minor/patch bump from {version!r}; pass --to"
        )
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def next_version(version: str, part: str) -> str:
    major, minor, patch = split_release(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(part)


def replace_section_version(path: Path, section: str, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_section: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            current_section = stripped[1:-1].strip()
            continue
        if current_section == section and stripped.startswith("version = "):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'version = "{version}"{newline}'
            path.write_text("".join(lines), encoding="utf-8")
            return
    raise RuntimeError(f"could not replace version in [{section}] of {path}")


def write_version(version: str) -> None:
    VERSION_PATH.write_text(f"{version}\n", encoding="utf-8")
    replace_section_version(PYPROJECT, "project", version)
    replace_section_version(CARGO_TOML, "package", version)


def versions() -> dict[str, str]:
    return {
        "version_txt": read_version(),
        "pyproject": pyproject_version(),
        "cargo_toml": cargo_version(),
        "cargo_lock": cargo_lock_version(),
    }


def check_version(args: argparse.Namespace) -> None:
    found = versions()
    failures: list[str] = []
    if pyproject_name() != PACKAGE_NAME:
        failures.append(
            f"pyproject package name is {pyproject_name()!r}, expected {PACKAGE_NAME!r}"
        )
    if len(set(found.values())) != 1:
        failures.append(f"version mismatch: {found}")
    if args.version is not None and set(found.values()) != {args.version}:
        failures.append(f"expected version {args.version!r}, saw {found}")
    print(json.dumps({"package": pyproject_name(), "versions": found}, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


def run_capture(args: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as error:
        return 127, str(error)
    return completed.returncode, completed.stdout.strip()


def check_tools(_args: argparse.Namespace) -> None:
    commands = {
        "cargo": ["cargo", "--version"],
        "docker": ["docker", "--version"],
        "maturin": [str(PYTHON), "-m", "maturin", "--version"],
        "cibuildwheel": [
            str(PYTHON),
            "-c",
            "from importlib.metadata import version; print('cibuildwheel ' + version('cibuildwheel'))",
        ],
        "twine": [str(PYTHON), "-m", "twine", "--version"],
    }
    result = {
        name: {"ok": code == 0, "output": output}
        for name, command in commands.items()
        for code, output in [run_capture(command)]
    }
    print(json.dumps(result, indent=2))
    missing = [name for name, check in result.items() if not check["ok"]]
    if missing:
        raise SystemExit(f"missing release tooling: {', '.join(missing)}")


def bump_version(args: argparse.Namespace) -> None:
    target = args.to or next_version(read_version(), args.part)
    validate_version(target)
    if args.write:
        write_version(target)
    print(target)


def fetch_pypi_project() -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{PACKAGE_NAME}/json", timeout=20
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def pypi_version_exists(version: str) -> bool:
    data = fetch_pypi_project()
    if data is None:
        return False
    releases = data.get("releases")
    return isinstance(releases, dict) and bool(releases.get(version))


def check_pypi(args: argparse.Namespace) -> None:
    validate_version(args.version)
    exists = pypi_version_exists(args.version)
    print(
        json.dumps(
            {"package": PACKAGE_NAME, "version": args.version, "version_exists": exists},
            indent=2,
        )
    )
    if exists:
        raise SystemExit(f"{PACKAGE_NAME} {args.version} already exists on PyPI")


def resolve_version(args: argparse.Namespace) -> None:
    current = read_version()
    validate_version(current)
    target = next_version(current, args.part) if pypi_version_exists(current) else current
    print(target)


def version_sort_key(version: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", version)
    if match is None:
        raise ValueError(version)
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def latest_pypi(args: argparse.Namespace) -> None:
    data = fetch_pypi_project()
    if data is None:
        print(
            json.dumps(
                {"package": PACKAGE_NAME, "exists": False, "latest_non_yanked": None},
                indent=2,
            )
        )
        return
    releases = data.get("releases")
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    if isinstance(releases, dict):
        for version, files in releases.items():
            if not isinstance(version, str) or not isinstance(files, list):
                continue
            if not any(isinstance(file, dict) and not file.get("yanked", False) for file in files):
                continue
            try:
                candidates.append((version_sort_key(version), version))
            except ValueError:
                continue
    latest = max(candidates)[1] if candidates else None
    info = data.get("info")
    info_version = info.get("version") if isinstance(info, dict) else None
    print(
        json.dumps(
            {
                "package": PACKAGE_NAME,
                "exists": True,
                "latest_non_yanked": latest,
                "pypi_info_version": info_version,
            },
            indent=2,
        )
    )
    if args.fail_if_mismatch and latest != info_version:
        raise SystemExit(
            f"PyPI info.version {info_version!r} does not match latest non-yanked {latest!r}"
        )


def wheelhouse(version: str, platform: str) -> Path:
    return REPO_ROOT / f"wheelhouse-v{version}-{platform}"


def shell_quote(value: str | Path) -> str:
    import shlex

    return shlex.quote(str(value))


def run(args: list[str], **kwargs: object) -> None:
    print("+", " ".join(shell_quote(arg) for arg in args))
    subprocess.run(args, cwd=REPO_ROOT, check=True, **kwargs)


def cargo_target_dir(platform: str, root: Path = REPO_ROOT) -> Path:
    if platform not in {"macos", "linux"}:
        raise ValueError(f"unknown platform: {platform}")
    return root / "target-release" / platform


def macos_build_env(root: Path = REPO_ROOT) -> dict[str, str]:
    return {
        "ARCHFLAGS": "-arch arm64",
        "CARGO_TARGET_DIR": str(cargo_target_dir("macos", root)),
        "MACOSX_DEPLOYMENT_TARGET": "11.0",
    }


def linux_build_env(root: Path = REPO_ROOT) -> dict[str, str]:
    target_dir = cargo_target_dir("linux", root).resolve()
    return {
        "CIBW_ARCHS_LINUX": "x86_64",
        "CIBW_BEFORE_ALL_LINUX": (
            "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal"
        ),
        "CIBW_BUILD": "cp311-manylinux_x86_64",
        "CIBW_CONTAINER_ENGINE": (
            f"docker; create_args: --volume={target_dir}:/cargo-target"
        ),
        "CIBW_ENVIRONMENT_LINUX": (
            'PATH="$HOME/.cargo/bin:$PATH" '
            "CARGO_NET_GIT_FETCH_WITH_CLI=true CARGO_TARGET_DIR=/cargo-target"
        ),
        "CIBW_SKIP": "*-musllinux_*",
    }


def build_platform(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    validate_version(version)
    output = wheelhouse(version, args.platform)
    output.mkdir(parents=True, exist_ok=True)
    cargo_target_dir(args.platform).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.platform == "macos":
        env.update(macos_build_env())
        run(
            [str(PYTHON), "-m", "maturin", "build", "--release", "--out", str(output)],
            env=env,
        )
        return
    env.update(linux_build_env())
    run(
        [
            str(PYTHON),
            "-m",
            "cibuildwheel",
            "--platform",
            "linux",
            "--output-dir",
            str(output),
        ],
        env=env,
    )


def audit_wheel(wheel: Path, version: str) -> dict[str, object]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    extension_entries = [
        name
        for name in names
        if name.startswith(f"{IMPORT_NAME}/{EXTENSION_NAME}")
        and name.endswith((".so", ".pyd"))
    ]
    checks = {
        "version_in_filename": version in wheel.name,
        "abi3_wheel": "abi3" in wheel.name,
        "has_package_init": f"{IMPORT_NAME}/__init__.py" in names,
        "has_env_source": f"{IMPORT_NAME}/env.py" in names,
        "has_extension": bool(extension_entries),
        "has_metadata": any(name.endswith(".dist-info/METADATA") for name in names),
        "no_bytecode": not any(
            "__pycache__" in Path(name).parts or name.endswith(".pyc") for name in names
        ),
    }
    return {
        "wheel": str(wheel),
        "extension_entries": extension_entries,
        "checks": checks,
    }


def assert_audits(results: list[dict[str, object]]) -> None:
    failures: dict[str, list[str]] = {}
    for result in results:
        checks = result["checks"]
        assert isinstance(checks, dict)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures[str(result["wheel"])] = failed
    if failures:
        print(json.dumps(results, indent=2))
        raise SystemExit(f"wheel audit failed: {failures}")


def find_wheels(version: str) -> list[Path]:
    wheels = list(wheelhouse(version, "macos").glob(f"*{version}*.whl"))
    wheels.extend(wheelhouse(version, "linux").glob(f"*{version}*.whl"))
    return sorted(wheels)


def audit_wheels(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    wheels = [wheel.resolve() for wheel in args.wheels] or find_wheels(version)
    if len(wheels) < 2:
        raise SystemExit(f"expected macOS and Linux wheels for {version}, found {wheels}")
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audits(results)
    print(json.dumps(results, indent=2))


def release_temp_dir() -> Path:
    configured = os.environ.get("RELEASE_BUILD_TMPDIR")
    root = Path(configured) if configured else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = args.wheel.resolve()
    with tempfile.TemporaryDirectory(
        prefix="breakout-turbo-env-wheel-smoke.", dir=release_temp_dir()
    ) as temporary:
        target = Path(temporary)
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(args.python),
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ]
        )
        code = f"""
import {IMPORT_NAME}
from {IMPORT_NAME} import {EXTENSION_NAME}
assert {IMPORT_NAME}.__file__.startswith({str(target)!r})
assert {EXTENSION_NAME}.__file__.startswith({str(target)!r})
assert hasattr({IMPORT_NAME}, "BreakoutVecEnv")
print({IMPORT_NAME}.__file__)
print({EXTENSION_NAME}.__file__)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        subprocess.run(
            [str(args.python), "-c", code],
            cwd=release_temp_dir(),
            env=env,
            check=True,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_check(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    wheels = find_wheels(version)
    if len(wheels) != 2:
        raise SystemExit(f"expected exactly two wheels for {version}, found {wheels}")
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audits(results)
    run([str(PYTHON), "-m", "twine", "check", *[str(wheel) for wheel in wheels]])
    print(
        json.dumps(
            {
                "audits": results,
                "sha256": {str(wheel): sha256(wheel) for wheel in wheels},
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-version")
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    tools = commands.add_parser("check-tools")
    tools.set_defaults(func=check_tools)

    bump = commands.add_parser("bump-version")
    bump.add_argument("--to")
    bump.add_argument("--part", choices=("major", "minor", "patch"), default="patch")
    bump.add_argument("--write", action="store_true")
    bump.set_defaults(func=bump_version)

    resolve = commands.add_parser("resolve-version")
    resolve.add_argument("--part", choices=("major", "minor", "patch"), default="patch")
    resolve.set_defaults(func=resolve_version)

    pypi = commands.add_parser("check-pypi")
    pypi.add_argument("--version", required=True)
    pypi.set_defaults(func=check_pypi)

    latest = commands.add_parser("latest-pypi")
    latest.add_argument("--fail-if-mismatch", action="store_true")
    latest.set_defaults(func=latest_pypi)

    platform = commands.add_parser("build-platform")
    platform.add_argument("--version")
    platform.add_argument("--platform", choices=("macos", "linux"), required=True)
    platform.set_defaults(func=build_platform)

    audit = commands.add_parser("audit-wheels")
    audit.add_argument("--version")
    audit.add_argument("wheels", nargs="*", type=Path)
    audit.set_defaults(func=audit_wheels)

    smoke = commands.add_parser("smoke-wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--python", type=Path, default=PYTHON)
    smoke.set_defaults(func=smoke_wheel)

    final = commands.add_parser("final-check")
    final.add_argument("--version")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
