from __future__ import annotations

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
