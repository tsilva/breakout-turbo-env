from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_BUILD = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)


def release_build_module():
    spec = importlib.util.spec_from_file_location("release_build", RELEASE_BUILD)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_workflow_restores_platform_scoped_cargo_cache():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "uses: actions/cache@v4" in workflow
    assert "path: target-release" in workflow
    assert (
        "key: cargo-release-v1-${{ matrix.platform }}-${{ runner.arch }}-${{ github.sha }}"
        in workflow
    )
    assert "cargo-release-v1-${{ matrix.platform }}-${{ runner.arch }}-" in workflow


def test_release_build_uses_persistent_platform_scoped_cargo_targets(tmp_path):
    release_build = release_build_module()

    macos = release_build.macos_build_env(tmp_path)
    linux = release_build.linux_build_env(tmp_path)

    assert macos["CARGO_TARGET_DIR"] == str(tmp_path / "target-release" / "macos")
    assert linux["CIBW_CONTAINER_ENGINE"] == (
        f"docker; create_args: --volume="
        f"{(tmp_path / 'target-release' / 'linux').resolve()}:/cargo-target"
    )
    assert "CARGO_TARGET_DIR=/cargo-target" in linux["CIBW_ENVIRONMENT_LINUX"]
