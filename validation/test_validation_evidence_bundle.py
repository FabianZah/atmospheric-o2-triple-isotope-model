"""Integrity tests for the canonical publication-validation evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
BUNDLE = ROOT / "model_data" / "validation_evidence"
MANIFEST = BUNDLE / "manifest.json"
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def test_evidence_manifest_matches_versioned_files() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["bundle_id"] == "publication_validation_evidence_v1"
    assert len(manifest["files"]) == 15
    for name, metadata in manifest["files"].items():
        path = ROOT / metadata["path"]
        assert path == BUNDLE / name
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]


def test_evidence_bundle_contains_no_absolute_local_paths() -> None:
    for path in BUNDLE.glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        for value in _strings(document):
            assert not value.startswith("/")
            assert not WINDOWS_ABSOLUTE.match(value)


def test_evidence_matrix_points_to_versioned_reports() -> None:
    matrix = json.loads(
        (ROOT / "model_data" / "literature" / "multimodel_evidence_matrix_v1.json")
        .read_text(encoding="utf-8")
    )
    for item in matrix["evidence"]:
        report = ROOT / item["source_report"]
        assert report.is_relative_to(BUNDLE)
        assert report.is_file()
