#!/usr/bin/env python3
"""Finalize and extract one release's human-readable changelog notes."""

from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
REPOSITORY_URL = "https://github.com/tsilva/breakout-turbo-env"


def _section_body(changelog: str, heading: re.Match[str]) -> tuple[str, int]:
    next_section = re.search(
        r"^(?:## |\[[^]]+\]:)", changelog[heading.end() :], re.MULTILINE
    )
    end = (
        heading.end() + next_section.start()
        if next_section is not None
        else len(changelog)
    )
    return changelog[heading.end() : end].strip(), end


def _require_meaningful_notes(body: str, label: str) -> None:
    has_prose = any(
        line.strip() and not line.lstrip().startswith("#")
        for line in body.splitlines()
    )
    if not has_prose:
        raise ValueError(f"CHANGELOG.md {label} section is empty")


def extract_release_notes(changelog: str, version: str) -> str:
    heading = re.compile(rf"^## \[{re.escape(version)}\].*$", re.MULTILINE)
    matches = list(heading.finditer(changelog))
    if not matches:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    if len(matches) != 1:
        raise ValueError(f"CHANGELOG.md has multiple sections for {version}")
    body, _ = _section_body(changelog, matches[0])
    _require_meaningful_notes(body, f"section for {version}")
    return body


def finalize_changelog(
    changelog: str,
    version: str,
    release_date: datetime.date,
    previous_version: str | None,
) -> str:
    """Promote human-authored Unreleased notes to an immutable release section."""
    release_heading = re.compile(rf"^## \[{re.escape(version)}\].*$", re.MULTILINE)
    existing_releases = list(release_heading.finditer(changelog))
    if existing_releases:
        extract_release_notes(changelog, version)
        return changelog

    unreleased_matches = list(re.finditer(r"^## Unreleased\s*$", changelog, re.MULTILINE))
    if len(unreleased_matches) != 1:
        raise ValueError(
            "CHANGELOG.md must contain exactly one '## Unreleased' section"
        )
    unreleased = unreleased_matches[0]
    body, section_end = _section_body(changelog, unreleased)
    _require_meaningful_notes(body, "Unreleased")

    promoted = (
        f"## Unreleased\n\n"
        f"## [{version}] - {release_date.isoformat()}\n\n"
        f"{body}\n\n"
    )
    result = changelog[: unreleased.start()] + promoted + changelog[section_end:]

    link = (
        f"[{version}]: {REPOSITORY_URL}/compare/v{previous_version}...v{version}"
        if previous_version is not None
        else f"[{version}]: {REPOSITORY_URL}/releases/tag/v{version}"
    )
    link_heading = re.compile(rf"^\[{re.escape(version)}\]:", re.MULTILINE)
    if link_heading.search(result) is None:
        first_link = re.search(r"^\[[^]]+\]:", result, re.MULTILINE)
        if first_link is None:
            result = result.rstrip() + f"\n\n{link}\n"
        else:
            result = result[: first_link.start()] + f"{link}\n" + result[first_link.start() :]
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Promote the Unreleased section to the requested version before extracting it",
    )
    parser.add_argument("--previous-version")
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        default=datetime.date.today(),
        help="Release date used with --finalize (ISO 8601)",
    )
    args = parser.parse_args(argv)
    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    if args.finalize:
        changelog = finalize_changelog(
            changelog,
            args.version,
            args.date,
            args.previous_version,
        )
        CHANGELOG_PATH.write_text(changelog, encoding="utf-8")
    print(extract_release_notes(changelog, args.version))


if __name__ == "__main__":
    main()
