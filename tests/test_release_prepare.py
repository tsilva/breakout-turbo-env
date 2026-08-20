from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCRIPT = REPO_ROOT / "scripts" / "release.py"


def release_module():
    spec = importlib.util.spec_from_file_location("release_script", RELEASE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_cli_requires_explicit_prepare_command():
    release = release_module()

    args = release.parse_args(["prepare", "--to", "0.3.6"])

    assert args.command == "prepare"
    assert args.to == "0.3.6"


def test_release_script_has_no_commit_tag_push_or_skip_authority():
    source = RELEASE_SCRIPT.read_text(encoding="utf-8")

    for forbidden in (
        '"git", "commit"',
        '"git", "tag"',
        '"git", "push"',
        "--skip-checks",
        '"uv", "lock"',
        '"cargo", "generate-lockfile"',
    ):
        assert forbidden not in source

    assert "Review and commit these files directly on main" in source
    assert "pull request" not in source


def test_dependency_snapshot_ignores_only_first_party_version():
    release = release_module()

    before = release.dependency_graph_snapshot()
    assert "env-breakoutatari2600-turbo-native" in before
    assert '"uv_options"' in before


def test_prepare_change_allowlist_contains_only_release_metadata():
    release = release_module()

    assert release.ALLOWED_RELEASE_FILES == {
        "Cargo.lock",
        "Cargo.toml",
        "CHANGELOG.md",
        "CITATION.cff",
        "VERSION.txt",
        "pyproject.toml",
        "redirect/pyproject.toml",
        "uv.lock",
    }


def test_changed_paths_does_not_parse_porcelain_status_columns(monkeypatch):
    release = release_module()
    outputs = iter(("CHANGELOG.md\nVERSION.txt", "", ""))
    monkeypatch.setattr(release, "capture", lambda _command: next(outputs))

    assert release.changed_paths() == ["CHANGELOG.md", "VERSION.txt"]


def test_update_redirect_version_updates_project_and_forwarded_extras(
    monkeypatch, tmp_path
):
    release = release_module()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "breakout-turbo-env"\nversion = "0.5.6"\n'
        'dependencies = ["env-breakoutatari2600-turbo-native==0.5.6"]\n\n'
        '[project.optional-dependencies]\n'
        'play = ["env-breakoutatari2600-turbo-native[play]==0.5.6"]\n'
        'dev = ["env-breakoutatari2600-turbo-native[dev]==0.5.6"]\n\n'
        '[tool.setuptools]\npackages = []\n',
        encoding="utf-8",
    )
    commands = []
    monkeypatch.setattr(release, "REDIRECT_PYPROJECT", pyproject)
    monkeypatch.setattr(release, "run", commands.append)

    release.update_redirect_version("0.5.7")

    source = pyproject.read_text(encoding="utf-8")
    assert 'version = "0.5.7"' in source
    assert source.count("==0.5.7") == 3
    assert "0.5.6" not in source
    assert commands == [
        [
            str(release.PYTHON),
            str(release.REDIRECT_RELEASE),
            "check-source",
            "--version",
            "0.5.7",
        ]
    ]
