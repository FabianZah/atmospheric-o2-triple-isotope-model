"""Release metadata, documentation, and deployment hygiene gates."""

from __future__ import annotations

from pathlib import Path
import re
import shlex
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
    *sorted((ROOT / "docs").glob("*.md")),
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
            "orcid": "https://orcid.org/0009-0006-4557-7155",
        }
    ]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Fabian Zahnow" in license_text
    assert "AUTHOR NAME" not in license_text


def test_author_orcid_is_valid_and_in_citation_downloads() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    orcid = citation["authors"][0]["orcid"]
    digits = orcid.rsplit("/", 1)[1].replace("-", "")
    assert re.fullmatch(r"\d{15}[\dX]", digits)
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    assert digits[-1] == ("X" if check == 10 else str(check))
    for filename in ("CITATION.bib", "CITATION.ris"):
        assert f"Author ORCID: {orcid}" in (ROOT / filename).read_text(encoding="utf-8")


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


def test_public_documentation_describes_one_operational_model() -> None:
    obsolete = (
        "equation_ledger.md", "equation_ledger.csv", "updated_model_validation.md",
        "young_reproduction_validation.md", "modern_o2_offset_attribution.md",
        "extrapolation_bounds.md",
    )
    assert all(not (ROOT / "docs" / name).exists() for name in obsolete)
    prose = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_MARKDOWN)
    assert "current model" not in prose.lower()
    assert "updated model" not in prose.lower()
    for name in (
        "web_api.py", "public_model_service.py",
        "updated_constrained_pco2_posterior.py",
        "updated_output_surface_joint_posterior.py",
        "updated_output_surface_posterior.py",
    ):
        source = (ROOT / "code" / name).read_text(encoding="utf-8")
        assert "updated model" not in source.lower(), name
    assert "## Branch Policy" not in prose
    gpp = (ROOT / "docs/GPP_NORMALIZATION_POLICY.md").read_text(encoding="utf-8")
    assert "100% modern GPP = 290 Pg C per year" in gpp


def test_compose_disables_cross_origin_access_by_default() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    environment = compose["services"]["model-api"]["environment"]
    assert environment["OXYTIB_CORS_ORIGINS"] == "${OXYTIB_CORS_ORIGINS:-}"


def test_ci_explicitly_runs_every_validation_test_module() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    covered = set()
    for step in workflow["jobs"]["test"]["steps"]:
        command = step.get("run", "").replace("\\\n", " ")
        for line in command.splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens[:3] == ["python", "-m", "pytest"]:
                covered.update(token for token in tokens[3:] if token.startswith("validation/test_") and token.endswith(".py"))
    discovered = {path.relative_to(ROOT).as_posix() for path in (ROOT / "validation").rglob("test_*.py")}
    assert discovered == covered, {
        "tests_missing_from_ci": sorted(discovered - covered),
        "ci_paths_missing_from_package": sorted(covered - discovered),
    }


def test_bundled_equation_renderer_has_its_license_and_local_assets() -> None:
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert 'src="assets/mathjax-config.js?' in index
    assert 'src="assets/vendor/mathjax/tex-svg.js?' in index
    license_path = ROOT / "web" / "vendor" / "mathjax" / "LICENSE"
    assert "Apache License" in license_path.read_text(encoding="utf-8")
    assert "MathJax" in (ROOT / "LICENSING.md").read_text(encoding="utf-8")
