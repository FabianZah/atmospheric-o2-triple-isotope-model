"""Export the single-model publication contract.

The contract deliberately distinguishes the deterministic publication model
from validation evidence and uncertainty end members.  It does not expose the
historical reconstruction presets as alternative predictive models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_output_surface import (  # noqa: E402
    UpdatedOutputSurfaceInput,
    run_updated_accelerated_forward,
)


DEFAULT_RESPONSE = ROOT / "model_data" / "updated_r7_response_surface_v1.json"
DEFAULT_OUTPUT_SURFACE = (
    ROOT / "model_data" / "updated_molecular_output_surface_v1.json"
)
DEFAULT_UNCERTAINTY = (
    ROOT / "model_data" / "uncertainty" / "updated_o2_uncertainty_layers_v1.json"
)
DEFAULT_EVIDENCE = (
    ROOT / "model_data" / "literature" / "multimodel_evidence_matrix_v1.json"
)
DEFAULT_EVIDENCE_BUNDLE = ROOT / "model_data" / "validation_evidence"
DEFAULT_OUTPUT = ROOT / "model_data" / "publication_model_contract_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _project_path(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "node_count": len(values),
    }


def run(
    response_path: Path = DEFAULT_RESPONSE,
    output_surface_path: Path = DEFAULT_OUTPUT_SURFACE,
    uncertainty_path: Path = DEFAULT_UNCERTAINTY,
    evidence_path: Path = DEFAULT_EVIDENCE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    response = _load(response_path)
    output_surface = _load(output_surface_path)
    uncertainty = _load(uncertainty_path)
    evidence = _load(evidence_path)

    model_id = response["model_data_id"]
    surface_id = output_surface["surface_data_id"]
    if output_surface["upstream_model_data_id"] != model_id:
        raise ValueError("output surface does not point to the central model")
    if uncertainty["central_model_data_id"] != model_id:
        raise ValueError("uncertainty contract does not point to the central model")

    modern_input = UpdatedOutputSurfaceInput(
        p_o2_pal=1.0,
        p_co2_ppm=294.0,
        gpp_pgC_per_year=290.0,
    )
    modern = run_updated_accelerated_forward(
        modern_input,
        surface_path=Path(output_surface_path),
    )
    conventional_delta18 = 1000.0 * math.expm1(
        modern.central_delta18_prime_permil / 1000.0
    )

    axes = output_surface["axes"]
    gpp_axis = [float(value) for value in axes["gpp_pgC_per_year"]]
    source_paths = {
        "central_forward_implementation": ROOT
        / "code"
        / "updated_molecular_forward_model.py",
        "accelerator_implementation": ROOT / "code" / "updated_output_surface.py",
        "inverse_implementation": ROOT
        / "code"
        / "updated_output_surface_inverse.py",
        "conditional_posterior_implementation": ROOT
        / "code"
        / "updated_output_surface_posterior.py",
        "joint_posterior_implementation": ROOT
        / "code"
        / "updated_output_surface_joint_posterior.py",
        "state_step_transient_implementation": ROOT
        / "code"
        / "updated_molecular_transient.py",
        "photosynthesis_step_transient_implementation": ROOT
        / "code"
        / "updated_photosynthesis_transient.py",
        "pco2_trajectory_transient_implementation": ROOT
        / "code"
        / "updated_pco2_trajectory_transient.py",
        "uncertainty_implementation": ROOT / "code" / "updated_uncertainty_layers.py",
        "observation_reference_implementation": ROOT
        / "code"
        / "observation_referenced_isotope.py",
        "central_response_surface": Path(response_path),
        "accelerated_output_surface": Path(output_surface_path),
        "uncertainty_contract": Path(uncertainty_path),
        "validation_evidence_matrix": Path(evidence_path),
        "validation_evidence_manifest": DEFAULT_EVIDENCE_BUNDLE / "manifest.json",
        "validation_release_scorecard": DEFAULT_EVIDENCE_BUNDLE
        / "updated_molecular_release_scorecard.json",
        "validation_liu_2021_benchmark": DEFAULT_EVIDENCE_BUNDLE
        / "liu_2021_low_gpp_multimodel_benchmark.json",
        "validation_cao_bao_2013_benchmark": DEFAULT_EVIDENCE_BUNDLE
        / "cao_bao_2013_multimodel_benchmark.json",
        "validation_luz_1999_benchmark": DEFAULT_EVIDENCE_BUNDLE
        / "luz_1999_productivity_benchmark.json",
        "validation_yang_2022_tracking": DEFAULT_EVIDENCE_BUNDLE
        / "yang_2022_co2_tracking_audit.json",
        "validation_brandon_2020_event": DEFAULT_EVIDENCE_BUNDLE
        / "brandon_2020_termination_v_audit.json",
        "validation_marine_accessibility": DEFAULT_EVIDENCE_BUNDLE
        / "marine_o2_accessibility_audit.json",
        "validation_uncertainty_layers": DEFAULT_EVIDENCE_BUNDLE
        / "updated_uncertainty_layers_audit.json",
        "validation_evidence_exporter": ROOT
        / "validation"
        / "export_validation_evidence_bundle.py",
        "integrated_acceptance_implementation": ROOT
        / "validation"
        / "audit_publication_model_acceptance.py",
    }

    contract: dict[str, Any] = {
        "schema_version": 1,
        "publication_model_id": "oxytib_publication_model_v1",
        "status": "active_publication_model",
        "single_model_policy": {
            "release_model_count": 1,
            "public_interface_model_count": 1,
            "historical_reconstructions_role": "validation evidence",
            "structural_endmembers_role": "structural sensitivity evidence",
            "parameter_update_policy": (
                "changes require declared physical inputs or equations"
            ),
            "output_adjustment_policy": "raw mechanistic outputs are preserved",
        },
        "deterministic_model": {
            "model_data_id": model_id,
            "model_class": (
                "resolved Photochem R1-R7 atmospheric column coupled to a "
                "conservative global atmospheric-O2 and biological-turnover budget"
            ),
            "central_kernel": "code/updated_molecular_forward_model.py",
            "numerical_accelerator": {
                "surface_data_id": surface_id,
                "implementation": "code/updated_output_surface.py",
                "extrapolation_permitted": bool(
                    output_surface["interpolation"]["extrapolation_permitted"]
                ),
                "output_offset_applied": bool(
                    output_surface["construction"]["output_offset_applied"]
                ),
                "smoothing_applied": bool(
                    output_surface["construction"]["smoothing_applied"]
                ),
            },
            "inverse_solver": "code/updated_output_surface_inverse.py",
            "conditional_posterior": "code/updated_output_surface_posterior.py",
            "joint_posterior": "code/updated_output_surface_joint_posterior.py",
            "transient_solvers": {
                "atmospheric_state_step": "code/updated_molecular_transient.py",
                "photosynthesis_step": "code/updated_photosynthesis_transient.py",
                "gradual_pCO2_trajectory": (
                    "code/updated_pco2_trajectory_transient.py"
                ),
            },
        },
        "operational_domain": {
            "pO2_PAL": _axis_summary(axes["po2_pal"]),
            "pCO2_ppm": _axis_summary(axes["pco2_ppm"]),
            "GPP_PgC_per_year": _axis_summary(gpp_axis),
            "GPP_percent_of_290_PgC_per_year": {
                "minimum": 100.0 * min(gpp_axis) / 290.0,
                "maximum": 100.0 * max(gpp_axis) / 290.0,
            },
            "outside_domain_policy": "reject; numerical extrapolation is not permitted",
        },
        "modern_reference_state": {
            "inputs": {
                "pO2_PAL": modern_input.p_o2_pal,
                "pCO2_ppm": modern_input.p_co2_ppm,
                "GPP_PgC_per_year": modern_input.gpp_pgC_per_year,
            },
            "raw_model": {
                "Delta_prime_17O_permil": modern.central_cap_delta17_prime_permil,
                "delta_prime_18O_permil": modern.central_delta18_prime_permil,
                "conventional_delta_18O_permil": conventional_delta18,
            },
            "observational_reference": {
                "source": "Pack (2021)",
                "Delta_prime_17O_permil": -0.432,
                "Delta_prime_17O_one_sigma_permil": 0.015,
                "conventional_delta_18O_permil": 23.9,
                "conventional_delta_18O_one_sigma_permil": 0.3,
            },
            "raw_model_reporting": "unadjusted mechanistic state",
        },
        "reporting_policy": {
            "primary_forward_output": "raw mechanistic model state",
            "observation_referenced_output_is_same_model": True,
            "observation_referenced_definition": (
                "Pack (2021) modern value plus the unchanged mechanistic "
                "scenario-minus-modern differential"
            ),
            "required_export_fields": [
                "raw_modern",
                "raw_scenario",
                "mechanistic_differential",
                "observation_referenced_scenario",
                "structural_baseline_residual",
            ],
        },
        "uncertainty": {
            "contract_id": uncertainty["uncertainty_contract_id"],
            "implementation": "code/updated_uncertainty_layers.py",
            "layers": list(uncertainty["layers"]),
            "layers_remain_separate": bool(
                uncertainty["policy"]["layers_must_remain_separate_in_exports"]
            ),
            "combined_public_confidence_interval_available": bool(
                uncertainty["policy"][
                    "combined_public_confidence_interval_available"
                ]
            ),
            "whole_domain_structural_Gaussian_sigma_available": bool(
                uncertainty["policy"][
                    "whole_domain_probabilistic_structural_discrepancy_available"
                ]
            ),
            "clima_role": (
                "non-probabilistic structural sensitivity end member used to "
                "quantify climate-architecture response"
            ),
        },
        "validation": {
            "evidence_matrix_id": evidence["matrix_id"],
            "evidence_bundle_id": "publication_validation_evidence_v1",
            "evidence_bundle_manifest": (
                "model_data/validation_evidence/manifest.json"
            ),
            "evidence": [
                {
                    "id": item["id"],
                    "evidence_type": item["evidence_type"],
                    "tested_quantity": item["tested_quantity"],
                    "uncertainty_use": item["uncertainty_use"],
                    "probabilistic_discrepancy_eligible": item[
                        "probabilistic_discrepancy_eligible"
                    ],
                    "source_report": item["source_report"],
                }
                for item in evidence["evidence"]
            ],
            "release_gate_generator": (
                "validation/audit_updated_molecular_release_scorecard.py"
            ),
            "integrated_acceptance_generator": (
                "validation/audit_publication_model_acceptance.py"
            ),
            "integrated_acceptance_report": (
                "docs/publication_model_acceptance.md"
            ),
        },
        "publication_scope": {
            "model": "one deterministic model and its declared inputs",
            "validation": "comparisons with observations and published models",
            "uncertainty": "separate measurement, parameter, numerical, and structural layers",
            "historical_reconstruction_role": "validation provenance",
        },
        "source_files": {
            name: {
                "path": _project_path(path),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--output-surface", type=Path, default=DEFAULT_OUTPUT_SURFACE)
    parser.add_argument("--uncertainty", type=Path, default=DEFAULT_UNCERTAINTY)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = run(
        args.response,
        args.output_surface,
        args.uncertainty,
        args.evidence,
        args.output,
    )
    print(
        f"Wrote {args.output} for {contract['publication_model_id']} "
        f"({len(contract['validation']['evidence'])} validation records)."
    )


if __name__ == "__main__":
    main()
