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
    assert "breakout-turbo-env" in before
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
        "uv.lock",
    }
