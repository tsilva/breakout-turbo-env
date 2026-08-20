from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release_state.py"
VERSION = "1.2.3"
COMMIT = "a" * 40


def module():
    spec = importlib.util.spec_from_file_location("release_state", SCRIPT)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_candidate(tmp_path):
    release_state = module()
    root = tmp_path / "candidate"
    dist = root / "dist"
    dist.mkdir(parents=True)
    for name in release_state.expected_distribution_names(VERSION):
        (dist / name).write_bytes(name.encode())
    (root / "sbom.spdx.json").write_text('{"spdxVersion":"SPDX-2.3"}\n')
    release_state.create_candidate(
        argparse.Namespace(
            version=VERSION,
            commit=COMMIT,
            repository=release_state.REPOSITORY,
            run_id="20",
            run_attempt="1",
            candidate=root,
        )
    )
    return release_state, root


def test_candidate_round_trip_binds_artifacts_and_commit(tmp_path):
    release_state, root = make_candidate(tmp_path)
    manifest = release_state.verify_candidate(
        root,
        version=VERSION,
        commit=COMMIT,
        repository=release_state.REPOSITORY,
        run_id="20",
    )
    assert manifest["state"] == "built"
    assert len(manifest["artifacts"]) == 4


def test_candidate_verification_rejects_artifact_mutation(tmp_path):
    release_state, root = make_candidate(tmp_path)
    manifest = json.loads((root / "release-manifest.json").read_text())
    artifact = root / manifest["artifacts"][0]["path"]
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest or size changed"):
        release_state.verify_candidate(
            root,
            version=VERSION,
            commit=COMMIT,
            repository=release_state.REPOSITORY,
        )


def test_candidate_rejects_unrecorded_root_file(tmp_path):
    release_state, root = make_candidate(tmp_path)
    (root / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected root entry"):
        release_state.verify_candidate(
            root,
            version=VERSION,
            commit=COMMIT,
            repository=release_state.REPOSITORY,
        )


def test_candidate_rejects_symlink_that_escapes_bundle(tmp_path):
    release_state, root = make_candidate(tmp_path)
    manifest = json.loads((root / "release-manifest.json").read_text())
    artifact = root / manifest["artifacts"][0]["path"]
    outside = tmp_path / "outside"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its bundle"):
        release_state.verify_candidate(
            root,
            version=VERSION,
            commit=COMMIT,
            repository=release_state.REPOSITORY,
        )
