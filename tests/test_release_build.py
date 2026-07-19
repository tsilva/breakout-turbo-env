from __future__ import annotations

import importlib.util
import re
import tomllib
import zipfile
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

    assert re.search(r"uses: actions/cache@[0-9a-f]{40} # v\d+", workflow)
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


def test_core_package_keeps_play_and_training_dependencies_optional():
    metadata = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]

    assert project["dependencies"] == ["gymnasium>=1.1,<2", "numpy>=1.26,<3"]
    assert project["optional-dependencies"]["play"] == ["pygame>=2.6,<3"]
    assert project["optional-dependencies"]["train"] == ["torch>=2.13,<3"]
    assert "pytest>=9.0.3,<10" in project["optional-dependencies"]["dev"]


def test_wheel_audit_accepts_only_supported_platform_metadata(tmp_path):
    release_build = release_build_module()
    version = release_build.read_version()
    wheel = tmp_path / (
        f"breakout_turbo_env-{version}-cp311-abi3-macosx_11_0_arm64.whl"
    )
    dist_info = f"breakout_turbo_env-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("breakout_turbo_env/__init__.py", "")
        archive.writestr("breakout_turbo_env/env.py", "")
        archive.writestr("breakout_turbo_env/_breakout_turbo.abi3.so", "")
        archive.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                (
                    "Metadata-Version: 2.4",
                    "License-Expression: MIT",
                    "Project-URL: Repository, https://github.com/tsilva/breakout-turbo-env",
                )
            ),
        )
        archive.writestr(f"{dist_info}/licenses/LICENSE", "MIT License")

    result = release_build.audit_wheel(wheel, version)
    assert all(result["checks"].values())

    unsupported = tmp_path / (
        f"breakout_turbo_env-{version}-cp311-abi3-macosx_11_0_x86_64.whl"
    )
    wheel.rename(unsupported)
    result = release_build.audit_wheel(unsupported, version)
    assert not result["checks"]["supported_platform_tag"]


def test_release_workflow_publishes_sdist_checksums_and_github_release():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "build-sdist" in workflow
    assert "*.tar.gz" in workflow
    assert "uv python install 3.11 3.14" in workflow
    assert "sha256sum * > SHA256SUMS" in workflow
    assert "gh release create" in workflow
