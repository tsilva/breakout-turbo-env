from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release_notes.py"


def release_notes_module():
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_release_notes_returns_only_selected_section():
    module = release_notes_module()
    changelog = "# Changelog\n\n## [2.0.0] - now\n\nNew.\n\n## [1.0.0] - then\n\nOld.\n"

    assert module.extract_release_notes(changelog, "2.0.0") == "New."
    assert module.extract_release_notes(changelog, "1.0.0") == "Old."
    with pytest.raises(ValueError, match="no section"):
        module.extract_release_notes(changelog, "3.0.0")


def test_extract_release_notes_rejects_empty_or_heading_only_sections():
    module = release_notes_module()

    for body in ("", "### Changed"):
        changelog = f"# Changelog\n\n## [2.0.0] - now\n\n{body}\n"
        with pytest.raises(ValueError, match="section for 2.0.0.*empty"):
            module.extract_release_notes(changelog, "2.0.0")


def test_extract_release_notes_rejects_duplicate_sections():
    module = release_notes_module()
    changelog = (
        "# Changelog\n\n"
        "## [2.0.0] - now\n\n- First.\n\n"
        "## [2.0.0] - earlier\n\n- Duplicate.\n"
    )

    with pytest.raises(ValueError, match="multiple sections"):
        module.extract_release_notes(changelog, "2.0.0")


def test_finalize_changelog_promotes_unreleased_notes_and_adds_compare_link():
    module = release_notes_module()
    changelog = (
        "# Changelog\n\n"
        "## Unreleased\n\n"
        "### Added\n\n"
        "- New behavior.\n\n"
        "## [1.0.0] - earlier\n\n"
        "- Old behavior.\n\n"
        "[1.0.0]: https://example.test/v1.0.0\n"
    )

    result = module.finalize_changelog(
        changelog,
        "1.1.0",
        datetime.date(2026, 7, 20),
        "1.0.0",
    )

    assert "## Unreleased\n\n## [1.1.0] - 2026-07-20" in result
    assert module.extract_release_notes(result, "1.1.0") == (
        "### Added\n\n- New behavior."
    )
    assert "[1.1.0]: https://github.com/tsilva/breakout-turbo-env/compare/v1.0.0...v1.1.0" in result


def test_finalize_changelog_rejects_empty_unreleased_without_changes():
    module = release_notes_module()
    changelog = "# Changelog\n\n## Unreleased\n\n### Changed\n"

    with pytest.raises(ValueError, match="Unreleased.*empty"):
        module.finalize_changelog(
            changelog,
            "1.1.0",
            datetime.date(2026, 7, 20),
            "1.0.0",
        )


def test_finalize_changelog_accepts_prepared_release_section():
    module = release_notes_module()
    changelog = "# Changelog\n\n## Unreleased\n\n## [1.1.0] - now\n\n- Ready.\n"

    assert (
        module.finalize_changelog(
            changelog,
            "1.1.0",
            datetime.date(2026, 7, 20),
            "1.0.0",
        )
        == changelog
    )
