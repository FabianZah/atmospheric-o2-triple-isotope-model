"""Package corrected-core evidence separately from archived v0.1.0 reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from export_validation_evidence_bundle import EVIDENCE_SPECS, _canonicalize, _write_json

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "model_data/validation_evidence/corrected_core_20261019"
MATRIX = ROOT / "model_data/literature/multimodel_evidence_matrix_20261019.json"
ARCHIVE = ROOT / "model_data/validation_evidence"
CORE_PATHS = sorted([p for p in (ROOT / "code").glob("*.py")] + [
    ROOT / "model_data/updated_r7_response_surface_v1.json",
    ROOT / "model_data/updated_molecular_output_surface_v1.json",
    ROOT / "model_data/marine_molecular_output_surface_v1.json",
    ROOT / "model_data/biosphere_configurations_v1.json",
    ROOT / "model_data/uncertainty/updated_o2_uncertainty_layers_v1.json",
])
COMPARISON_INPUTS = sorted(
    [p for folder in (ROOT / "model_data/literature", ROOT / "validation/reference_data", ROOT / "code/data")
     for p in folder.rglob("*") if p.is_file() and p.name != MATRIX.name
     and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    + [ROOT / "outputs/young_fig7_digitized_contours.csv", ROOT / "outputs/young_fig8_digitized_curves.csv"]
)


def sha(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in {".py", ".json", ".csv", ".txt", ".md", ".yaml", ".yml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def evidence_matches_current_core():
    path = TARGET / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    expected = {p.relative_to(ROOT).as_posix(): sha(p) for p in CORE_PATHS}
    if manifest.get("core_source_hashes") != expected:
        return False
    if manifest.get("comparison_input_hashes") != {p.relative_to(ROOT).as_posix(): sha(p) for p in COMPARISON_INPUTS}:
        return False
    return all((ROOT / row["path"]).is_file() and sha(ROOT / row["path"]) == row["sha256_lf"]
               and sha(ROOT / row["generator"]) == row["generator_sha256_lf"]
               for row in manifest["files"].values())


def run():
    response = json.loads((ROOT / "model_data/updated_r7_response_surface_v1.json").read_text())
    model_id = response["model_data_id"]
    scorecard = json.loads((ROOT / "outputs/updated_molecular_release_scorecard.json").read_text())
    if scorecard["model_data_id"] != model_id:
        raise ValueError("Release scorecard belongs to a different atmospheric core")
    records = {}
    for name, metadata in EVIDENCE_SPECS.items():
        historical = name.startswith("clima_")
        source = (ARCHIVE if historical else ROOT / "outputs") / name
        report = _canonicalize(json.loads(source.read_text(encoding="utf-8-sig")))
        if "source_benchmark" in report and not historical:
            report["source_benchmark"] = (TARGET / "liu_2021_low_gpp_multimodel_benchmark.json").relative_to(ROOT).as_posix()
        target = TARGET / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, report)
        records[name] = dict(path=target.relative_to(ROOT).as_posix(), sha256_lf=sha(target),
            generator=metadata["generator"], generator_sha256_lf=sha(ROOT / metadata["generator"]), role=metadata["role"],
            generator_hash_role="available_script_not_original_execution_provenance" if historical else "current_execution_script",
            evidence_scope="archived_chemistry_structural_experiment" if historical else "current_core_comparison",
            applicable_model_data_id=None if historical else model_id,
            source_path=source.relative_to(ROOT).as_posix(), source_sha256_lf=sha(source))
    manifest = dict(schema_version=1, bundle_id="corrected_core_evidence_20261019",
        model_data_id=model_id, hash_convention="SHA-256 of LF-normalized text; binary comparison inputs use raw bytes",
        core_source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in CORE_PATHS},
        comparison_input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in COMPARISON_INPUTS}, files=records,
        scope="Current numerical and literature comparisons; climate records explicitly retain archived chemistry and provide no current numerical correction")
    _write_json(TARGET / "manifest.json", manifest)
    matrix = json.loads((ROOT / "model_data/literature/multimodel_evidence_matrix_v1.json").read_text())
    matrix.update(matrix_id="oxytib_evidence_matrix_20261019", frozen_on="2026-10-19",
                  forward_model_surface_id=scorecard["surface_data_id"])
    for item in matrix["evidence"]:
        item.pop("metrics", None)
        name = Path(item["source_report"]).name
        item["source_report"] = (TARGET / name).relative_to(ROOT).as_posix()
        item["evidence_scope"] = records[name]["evidence_scope"]
        if name == "liu_2021_low_gpp_multimodel_benchmark.json":
            comparison = json.loads((TARGET / name).read_text())
            metrics = comparison["metrics"]
            item["source_domain"] = item.pop("domain")
            item["shared_state_coverage"] = {
                "source_states": comparison["scenario_count"],
                "paired_states": metrics["paired_finite_state_count"],
                "unavailable_states": metrics["updated_unavailable_state_count"],
                "note": "The 5% own-reference GPP states lie below OXYTIB's declared production domain. Direction checks use available curves; unavailable curves are explicitly unassessed."
            }
    _write_json(MATRIX, matrix)
    assert evidence_matches_current_core()
    print(f"Wrote {len(records)} scoped reports to {TARGET}")


if __name__ == "__main__":
    run()
