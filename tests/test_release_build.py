from __future__ import annotations

import importlib.util
import os
import re
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_BUILD = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)
LOCK_SCRIPT = REPO_ROOT / "scripts" / "lock.py"


def release_build_module():
    spec = importlib.util.spec_from_file_location("release_build", RELEASE_BUILD)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lock_script_module():
    spec = importlib.util.spec_from_file_location("lock_script", LOCK_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_workflow_restores_platform_scoped_cargo_cache():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "release-build.yml"
    ).read_text(encoding="utf-8")

    assert re.search(r"uses: actions/cache@[0-9a-f]{40} # v\d+", workflow)
    assert "path: target-release" in workflow
    assert (
        "key: cargo-release-v2-${{ matrix.platform }}-${{ runner.arch }}-${{ needs.validate.outputs.commit }}"
        in workflow
    )
    assert "cargo-release-v2-${{ matrix.platform }}-${{ runner.arch }}-" in workflow


def test_release_build_uses_persistent_platform_scoped_cargo_targets(tmp_path):
    release_build = release_build_module()

    macos = release_build.macos_build_env(tmp_path)
    output = tmp_path / "wheelhouse-v0.0.0-linux"
    linux = release_build.linux_build_command(output, tmp_path)

    assert macos["CARGO_TARGET_DIR"] == str(tmp_path / "target-release" / "macos")
    assert "linux/amd64" in linux
    assert (
        f"{(tmp_path / 'target-release' / 'linux').resolve()}:/cargo-target"
        in linux
    )
    assert "CARGO_TARGET_DIR=/cargo-target" in linux
    assert "RUSTUP_TOOLCHAIN=stable" in linux
    assert release_build.MATURIN_IMAGE in linux
    assert "--locked" in linux
    assert "manylinux_2_28" in linux
    assert not any("curl" in argument or "rustup" in argument for argument in linux)


def test_core_package_keeps_play_and_training_dependencies_optional():
    metadata = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]

    assert project["dependencies"] == ["gymnasium>=1.1,<2", "numpy>=1.26,<3"]
    assert project["optional-dependencies"]["play"] == ["pygame>=2.6,<3"]
    assert project["optional-dependencies"]["train"] == ["torch>=2.13,<3"]
    assert "pytest>=9.0.3,<10" in project["optional-dependencies"]["dev"]
    assert not any(
        dependency.startswith("cibuildwheel")
        for dependency in project["optional-dependencies"]["dev"]
    )


def test_lock_policy_is_repository_owned_and_has_no_exemptions():
    release_build = release_build_module()
    release_build.check_lock_policy(None)

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert lock["options"]["exclude-newer-span"] == "P7D"
    assert "exclude-newer-package" not in lock["options"]


def test_hermetic_lock_command_does_not_forward_host_uv_configuration(tmp_path):
    lock_script = lock_script_module()
    command = lock_script.docker_command(tmp_path, check=True)

    assert "--check" in command
    assert "XDG_CONFIG_HOME=/uv-state/config" in command
    assert "UV_CACHE_DIR=/uv-state/cache" in command
    assert not any(part.startswith("UV_CONFIG_FILE=") for part in command)
    assert os.environ.get("UV_CONFIG_FILE") not in command


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
    build = (REPO_ROOT / ".github" / "workflows" / "release-build.yml").read_text(
        encoding="utf-8"
    )
    publish = (
        REPO_ROOT / ".github" / "workflows" / "release-publish.yml"
    ).read_text(encoding="utf-8")

    assert "build-sdist" in build
    assert "*.tar.gz" in build
    assert "uv python install 3.11 3.14" in build
    assert "release_state.py candidate" in build
    assert "attest-build-provenance" in build
    assert "attest-sbom" in build
    assert "gh release create" in publish
    assert "gh-action-pypi-publish" in publish
    assert "contents: write" in publish
    assert "create-github-app-token" not in publish
    assert "RELEASE_APP_ID" not in publish
    assert "parity_run_id" not in build
    assert "push:\n    tags:" not in build + publish


def test_release_notes_are_validated_before_pypi_publication():
    build = (REPO_ROOT / ".github" / "workflows" / "release-build.yml").read_text(
        encoding="utf-8"
    )
    publish = (
        REPO_ROOT / ".github" / "workflows" / "release-publish.yml"
    ).read_text(encoding="utf-8")
    release_script = (REPO_ROOT / "scripts" / "release.py").read_text(
        encoding="utf-8"
    )

    assert build.index("release_notes.py") < build.index("check-pypi")
    assert "release_state.py verify" in publish
    assert publish.index("release_state.py verify") < publish.index(
        "Publish exact candidate to PyPI"
    )
    assert release_script.index("finalize_release_notes(version)") < release_script.index(
        'helper("bump-version", "--to", version, "--write")'
    )
    assert release_script.index("validate_release_notes(version)") < release_script.rindex(
        "run_checks()"
    )
    assert "create_commit_and_tag" not in release_script
    assert "push_release" not in release_script
    assert '"CHANGELOG.md"' in release_script


def test_built_wheel_smoke_exercises_exact_live_snapshot_replay():
    source = RELEASE_BUILD.read_text(encoding="utf-8")

    for required in (
        "Path(module.__file__).resolve()",
        "module_path.is_relative_to(environment_root)",
        "supports_live_snapshots",
        "capture_snapshots",
        '"snapshots": [handles[0], handles[0]]',
        'restored_infos["start_source"]',
        "np.testing.assert_array_equal(expected, actual)",
    ):
        assert required in source

    assert "__file__.startswith" not in source


def test_resolved_path_containment_rejects_lexical_prefix_collisions(tmp_path):
    environment = tmp_path / "venv"
    package = environment / "lib" / "package.py"
    sibling = tmp_path / "venv-malicious" / "package.py"
    package.parent.mkdir(parents=True)
    sibling.parent.mkdir(parents=True)
    package.touch()
    sibling.touch()

    assert package.resolve().is_relative_to(environment.resolve())
    assert not sibling.resolve().is_relative_to(environment.resolve())


def test_resolved_path_containment_canonicalizes_symlinks(tmp_path):
    canonical = tmp_path / "private" / "var" / "venv"
    canonical.mkdir(parents=True)
    alias_root = tmp_path / "var"
    alias_root.symlink_to(tmp_path / "private" / "var", target_is_directory=True)
    imported = canonical / "lib" / "package.py"
    imported.parent.mkdir()
    imported.touch()

    assert imported.resolve().is_relative_to((alias_root / "venv").resolve())


def test_resolved_path_containment_rejects_checkout_import(tmp_path):
    environment = tmp_path / "venv"
    checkout_module = tmp_path / "checkout" / "breakout_turbo_env" / "__init__.py"
    environment.mkdir()
    checkout_module.parent.mkdir(parents=True)
    checkout_module.touch()

    assert not checkout_module.resolve().is_relative_to(environment.resolve())
