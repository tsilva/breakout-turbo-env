from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMUNITY_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "CITATION.cff",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/pull_request_template.md",
)


def test_community_files_and_platform_contract_are_present():
    for relative in COMMUNITY_FILES:
        assert (REPO_ROOT / relative).is_file(), relative

    specs = (REPO_ROOT / "SPECS.md").read_text(encoding="utf-8")
    support = (REPO_ROOT / "SUPPORT.md").read_text(encoding="utf-8")
    expected = "Apple-silicon macOS and x86-64 Linux"
    assert expected in specs
    assert expected in support
    assert "unsupported" in support.lower()


def test_package_metadata_exposes_public_project_identity():
    metadata = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert metadata["license"] == "MIT"
    assert metadata["license-files"] == ["LICENSE"]
    assert metadata["authors"]
    assert metadata["urls"]["Repository"] == (
        "https://github.com/tsilva/breakout-turbo-env"
    )
    assert metadata["urls"]["Documentation"].endswith("#readme")
    assert "Programming Language :: Python :: 3.14" in metadata["classifiers"]


def test_readme_uses_pypi_safe_images_and_local_links_resolve():
    markdown_paths = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").glob("*.md")),
    ]
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert 'src="./' not in readme
    assert "raw.githubusercontent.com/tsilva/breakout-turbo-env" in readme

    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            assert (markdown_path.parent / relative).resolve().exists(), (
                markdown_path,
                target,
            )


def test_workflow_actions_are_pinned_to_full_commits():
    pattern = re.compile(r"^\s*uses:\s+[^@\s]+@([^\s]+)", re.MULTILINE)
    workflows = list((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows
    for workflow in workflows:
        refs = pattern.findall(workflow.read_text(encoding="utf-8"))
        assert refs, workflow
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs), workflow


def test_ci_covers_supported_python_versions_and_platforms():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in workflow
    assert "runner: macos-15" in workflow
    assert "runner: ubuntu-24.04" in workflow
    assert "cargo clippy --all-targets -- -D warnings" in workflow
    assert 'pytest -m "not stable_retro"' in workflow
