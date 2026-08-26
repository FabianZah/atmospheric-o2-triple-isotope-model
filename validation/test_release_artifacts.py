"""Release metadata, documentation, and deployment hygiene gates."""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote

import yaml

from public_model_service import model_metadata


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
PUBLIC_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "SETUP.md",
    ROOT / "LICENSING.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "PROJECT_STRUCTURE.md",
    ROOT / "docs" / "server_deployment.md",
    ROOT / "docs" / "spherule_inversion_workflow.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_citation_and_license_are_release_ready() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == model_metadata()["citation"]["version"]
    assert citation["repository-code"] == model_metadata()["citation"]["repository"]
    assert citation["authors"] == [
        {
            "family-names": "Zahnow",
            "given-names": "Fabian",
            "email": "fabs2906@gmail.com",
        }
    ]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Fabian Zahnow" in license_text
    assert "AUTHOR NAME" not in license_text


def test_public_documents_use_current_entry_points_and_portable_paths() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_MARKDOWN)
    assert "run_model.py app" not in combined
    assert "C:\\Users\\" not in combined
    assert "AUTHOR NAME" not in combined
    assert not (ROOT / "SNAPSHOT_MANIFEST.md").exists()
    assert not (ROOT / "SNAPSHOT_FILE_LIST.txt").exists()


def test_public_markdown_links_resolve() -> None:
    missing: list[str] = []
    for document in PUBLIC_MARKDOWN:
        for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if path_part and not (document.parent / path_part).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_compose_disables_cross_origin_access_by_default() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    environment = compose["services"]["model-api"]["environment"]
    assert environment["O2_MODEL_CORS_ORIGINS"] == "${O2_MODEL_CORS_ORIGINS:-}"
