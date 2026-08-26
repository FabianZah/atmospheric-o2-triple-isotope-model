"""Export canonical, versioned evidence used by publication acceptance.

Development audits write detailed reports under outputs/. This exporter
removes machine-local artifact paths, preserves the scientific report content,
and writes the compact JSON evidence required by a clean source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
DEFAULT_SOURCE = ROOT / "outputs"
DEFAULT_TARGET = ROOT / "model_data" / "validation_evidence"

EVIDENCE_SPECS = {
    "updated_molecular_release_scorecard.json": {
        "generator": "validation/audit_updated_molecular_release_scorecard.py",
        "role": "central release gates and observational comparisons",
    },
    "liu_2021_low_gpp_multimodel_benchmark.json": {
        "generator": "validation/merge_liu_2021_gpp_grid.py",
        "role": "low-GPP published-model comparison",
    },
    "cao_bao_2013_multimodel_benchmark.json": {
        "generator": "validation/audit_cao_bao_2013_benchmark.py",
        "role": "high-pCO2 published-model comparison",
    },
    "luz_1999_productivity_benchmark.json": {
        "generator": "validation/audit_luz_1999_productivity.py",
        "role": "inverse productivity architecture comparison",
    },
    "yang_2022_co2_tracking_audit.json": {
        "generator": "validation/audit_yang_2022_co2_tracking.py",
        "role": "low-CO2 observational predictive validation",
    },
    "brandon_2020_termination_v_audit.json": {
        "generator": "validation/audit_brandon_2020_termination_v.py",
        "role": "event-scale observational productivity validation",
    },
    "marine_o2_accessibility_audit.json": {
        "generator": "validation/audit_marine_o2_accessibility.py",
        "role": "rejected structural marine-access sensitivity",
    },
    "updated_uncertainty_layers_audit.json": {
        "generator": "validation/audit_uncertainty_layers.py",
        "role": "separated uncertainty-layer validation",
    },
    "clima_global_o2_response.json": {
        "generator": "validation/audit_clima_global_o2_response.py",
        "role": "one-PAL additive-CO2 climate structural end member",
    },
    "clima_global_o2_response_fixed_total.json": {
        "generator": "validation/audit_clima_global_o2_response.py",
        "role": "one-PAL fixed-total-pressure climate structural end member",
    },
    "clima_global_o2_response_0p1pal.json": {
        "generator": "validation/audit_clima_global_o2_response.py",
        "role": "low-pO2 climate structural end member",
    },
    "clima_global_o2_response_2pal.json": {
        "generator": "validation/audit_clima_global_o2_response.py",
        "role": "high-pO2 climate structural end member",
    },
    "clima_pressure_convention_audit.json": {
        "generator": "validation/audit_clima_pressure_conventions.py",
        "role": "climate pressure-convention robustness audit",
    },
    "clima_po2_cross_audit.json": {
        "generator": "validation/audit_clima_po2_cross.py",
        "role": "climate and pO2 interaction audit",
    },
    "yang_lowco2_predictive_error.json": {
        "generator": "validation/audit_yang_lowco2_predictive_error.py",
        "role": "low-CO2 predictive-error candidate",
    },
}

LOCAL_ARTIFACT_KEYS = {
    "csv",
    "figure",
    "own_reference_percent_figure",
    "accessible_turnover_matched_figure",
    "local_csv",
    "global_csv",
    "outputs",
    "manifest",
    "artifact_directory",
}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_absolute_path(value: str) -> bool:
    return value.startswith("/") or bool(WINDOWS_ABSOLUTE.match(value))


def _project_relative_path(value: str) -> str | None:
    try:
        return Path(value).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _canonicalize(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str) and _is_absolute_path(value):
        if key == "source_benchmark":
            return (
                "model_data/validation_evidence/"
                "liu_2021_low_gpp_multimodel_benchmark.json"
            )
        project_path = _project_relative_path(value)
        if project_path is not None:
            return project_path
        raise ValueError(f"unexpected absolute path in evidence field {key!r}")
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key in LOCAL_ARTIFACT_KEYS:
                continue
            cleaned[child_key] = _canonicalize(child_value, key=child_key)
        return cleaned
    if isinstance(value, list):
        return [_canonicalize(item, key=key) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: Any) -> None:
    """Write canonical JSON with repository-stable LF line endings."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")


def run(
    source_directory: Path = DEFAULT_SOURCE,
    target_directory: Path = DEFAULT_TARGET,
) -> dict[str, Any]:
    source_directory = Path(source_directory)
    target_directory = Path(target_directory)
    missing = [
        name for name in EVIDENCE_SPECS if not (source_directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing generated evidence report(s): " + ", ".join(missing)
        )

    target_directory.mkdir(parents=True, exist_ok=True)
    manifest_files: dict[str, Any] = {}
    for name, metadata in EVIDENCE_SPECS.items():
        source = source_directory / name
        target = target_directory / name
        report = json.loads(source.read_text(encoding="utf-8-sig"))
        canonical = _canonicalize(report)
        _write_json(target, canonical)
        manifest_files[name] = {
            "path": target.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
            "generator": metadata["generator"],
            "role": metadata["role"],
            "development_source": f"outputs/{name}",
        }

    manifest = {
        "schema_version": 1,
        "bundle_id": "publication_validation_evidence_v1",
        "purpose": (
            "Canonical machine-readable evidence required to reproduce the "
            "integrated publication-model acceptance decision from a clean checkout"
        ),
        "files": manifest_files,
    }
    manifest_path = target_directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    manifest = run(args.source_directory, args.target_directory)
    print(
        f"Wrote {len(manifest['files'])} canonical evidence reports to "
        f"{args.target_directory}"
    )


if __name__ == "__main__":
    main()
