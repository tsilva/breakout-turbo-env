#!/usr/bin/env python3
"""Check or regenerate uv.lock without inheriting host uv configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UV_IMAGE = (
    "ghcr.io/astral-sh/uv@"
    "sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca"
)
RELEASE_HELPER = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def docker_command(state_dir: Path, *, check: bool) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{REPO_ROOT}:/workspace",
        "--volume",
        f"{state_dir}:/uv-state",
        "--workdir",
        "/workspace",
        "--env",
        "XDG_CONFIG_HOME=/uv-state/config",
        "--env",
        "UV_CACHE_DIR=/uv-state/cache",
        "--env",
        "UV_PROJECT_ENVIRONMENT=/uv-state/venv",
        UV_IMAGE,
        "uv",
        "lock",
    ]
    if check:
        command.append("--check")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate uv.lock; the default only checks it",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="breakout-uv-lock.") as temporary:
        state_dir = Path(temporary)
        for child in ("cache", "config", "venv"):
            (state_dir / child).mkdir()
        subprocess.run(
            docker_command(state_dir, check=not args.write),
            cwd=REPO_ROOT,
            check=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
            },
        )

    subprocess.run(
        [str(PYTHON), str(RELEASE_HELPER), "check-lock-policy"],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
