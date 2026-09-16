"""Export the versioned uncertainty-layer contract for the updated model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
EVIDENCE_BUNDLE = ROOT / "model_data" / "validation_evidence"


def _project_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _global_climate_rows(report: dict[str, object]) -> list[dict[str, float]]:
    return [
        {
            "pO2_PAL": float(row["pO2_PAL"]),
            "pCO2_ppm": float(row["pCO2_ppm"]),
            "GPP_percent_of_updated_modern": float(
                row["GPP_percent_of_updated_modern"]
            ),
            "GPP_PgC_per_year": float(row["GPP_PgC_per_year"]),
            "fixed_cap_delta17_prime_permil": float(
                row["fixed_cap_delta17_prime_permil"]
            ),
            "climate_cap_delta17_prime_permil": float(
                row["climate_cap_delta17_prime_permil"]
            ),
            "climate_minus_fixed_D17O_permil": float(
                row["climate_minus_fixed_D17O_permil"]
            ),
            "climate_minus_fixed_delta18_permil": float(
                row["climate_minus_fixed_delta18_permil"]
            ),
        }
        for row in report["rows"]
    ]


def _po2_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def run(
    evidence_path: Path,
    climate_report_path: Path,
    response_bundle_path: Path,
    output_path: Path,
    predictive_error_path: Path | None = None,
    fixed_total_climate_report_path: Path | None = None,
    pressure_convention_audit_path: Path | None = None,
    low_po2_climate_report_path: Path | None = None,
    high_po2_climate_report_path: Path | None = None,
    po2_cross_audit_path: Path | None = None,
) -> dict[str, object]:
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    climate = json.loads(Path(climate_report_path).read_text(encoding="utf-8"))
    response = json.loads(Path(response_bundle_path).read_text(encoding="utf-8"))
    predictive_error = (
        None
        if predictive_error_path is None
        else json.loads(Path(predictive_error_path).read_text(encoding="utf-8"))
    )
    fixed_total_climate = (
        None
        if fixed_total_climate_report_path is None
        else json.loads(
            Path(fixed_total_climate_report_path).read_text(encoding="utf-8")
        )
    )
    pressure_audit = (
        None
        if pressure_convention_audit_path is None
        else json.loads(
            Path(pressure_convention_audit_path).read_text(encoding="utf-8")
        )
    )
    low_po2_climate = (
        None
        if low_po2_climate_report_path is None
        else json.loads(Path(low_po2_climate_report_path).read_text(encoding="utf-8"))
    )
    high_po2_climate = (
        None
        if high_po2_climate_report_path is None
        else json.loads(Path(high_po2_climate_report_path).read_text(encoding="utf-8"))
    )
    po2_cross_audit = (
        None
        if po2_cross_audit_path is None
        else json.loads(Path(po2_cross_audit_path).read_text(encoding="utf-8"))
    )

    structural_ids = {
        "young_2014_fig7_fig8",
        "liu_2021_native_low_gpp_grid",
        "cao_bao_2013_high_pco2",
    }
    structural_sources = [
        {
            "id": item["id"],
            "evidence_type": item["evidence_type"],
            "domain": item["domain"],
            "uncertainty_use": item["uncertainty_use"],
            "probabilistic_discrepancy_eligible": item[
                "probabilistic_discrepancy_eligible"
            ],
            "source_report": item["source_report"],
        }
        for item in evidence["evidence"]
        if item["id"] in structural_ids
    ]
    climate_rows = _global_climate_rows(climate)
    fixed_total_climate_rows = (
        []
        if fixed_total_climate is None
        else _global_climate_rows(fixed_total_climate)
    )
    normalization = response["transfer_normalization"]
    contract: dict[str, object] = {
        "schema_version": 1,
        "uncertainty_contract_id": "updated_o2_uncertainty_layers_v1",
        "central_model_data_id": response["model_data_id"],
        "policy": {
            "combined_public_confidence_interval_available": False,
            "whole_domain_probabilistic_structural_discrepancy_available": False,
            "inter_model_spread_used_as_Gaussian_sigma": False,
            "numerical_guardrails_used_as_probability_distributions": False,
            "layers_must_remain_separate_in_exports": True,
        },
        "layers": {
            "measurement_proxy": {
                "probabilistic": True,
                "distribution": "Gaussian in measured air-O2 Delta-prime-17O when the reported analytical uncertainty is one sigma",
                "spherule_conversion": "Zahnow et al. (2025), Eq. 3; analytical delta18O and Delta-prime-17O uncertainties are propagated before inversion",
                "automatically_add_modern_reference_uncertainty": False,
            },
            "parameter": {
                "probabilistic_default": False,
                "components": {
                    "Adnew_source_isoflux": {
                        "source": normalization["observational_source"],
                        "mean_permil_PgC_per_year": normalization[
                            "observed_isoflux_permil_pgc_per_year"
                        ],
                        "one_sigma_permil_PgC_per_year": normalization[
                            "observed_isoflux_uncertainty_permil_pgc_per_year"
                        ],
                        "propagation": "minus-one-sigma, mean, and plus-one-sigma forward endpoints",
                        "probabilistic_input_definition_available": True,
                        "posterior_marginalization_implemented": False,
                    },
                    "biological_pathways": {
                        "source": "declared literature-corner ensemble in biological_o2_ensemble.py",
                        "propagation": "non-probabilistic corner envelope",
                        "probabilistic_input_definition_available": False,
                    },
                },
                "joint_parameter_interval": "full crossed Adnew endpoint and biological-corner envelope",
            },
            "numerical": {
                "probabilistic": False,
                "components": [
                    "native R7 response-surface crossed-holdout guardrail",
                    "accelerated global-output-surface interpolation guardrail",
                ],
                "interpretation": "deterministic numerical accuracy bounds",
            },
            "structural": {
                "probabilistic_default": False,
                "evidence_matrix_id": evidence["matrix_id"],
                "whole_domain_sigma_permil": None,
                "sources": structural_sources,
                "climate_endmember": {
                    "id": "clima_0p5p11_1pal_additive_co2_v1",
                    "pressure_convention": "additive_co2",
                    "role": "non-probabilistic structural sensitivity end member",
                    "central_model_changed": False,
                    "interpolation_permitted": False,
                    "domain": {
                        "pO2_PAL": [1.0, 1.0],
                        "pCO2_ppm": [300.0, 60000.0],
                        "GPP_percent_of_updated_modern": [25.0, 150.0],
                    },
                    "limitations": climate["promotion_decision"]["reason"],
                    "source_report": _project_path(Path(climate_report_path)),
                    "values": climate_rows,
                },
                "climate_pressure_alternatives": (
                    []
                    if fixed_total_climate is None
                    else [
                        {
                            "id": "clima_0p5p11_1pal_fixed_total_dry_major_v1",
                            "pressure_convention": "fixed_total_dry_major",
                            "role": "non-probabilistic structural sensitivity end member",
                            "central_model_changed": False,
                            "interpolation_permitted": False,
                            "domain": {
                                "pO2_PAL": [1.0, 1.0],
                                "pCO2_ppm": [300.0, 60000.0],
                                "GPP_percent_of_updated_modern": [25.0, 150.0],
                            },
                            "limitations": fixed_total_climate[
                                "promotion_decision"
                            ]["reason"],
                            "source_report": _project_path(
                                Path(fixed_total_climate_report_path)
                            ),
                            "values": fixed_total_climate_rows,
                        }
                    ]
                ),
                "climate_pressure_robustness": (
                    None
                    if pressure_audit is None
                    else {
                        "status": pressure_audit["status"],
                        "maximum_absolute_D17O_difference_through_30000ppm": pressure_audit[
                            "global_O2"
                        ]["maximum_absolute_D17O_difference_through_30000ppm"],
                        "maximum_absolute_D17O_difference_full_domain": pressure_audit[
                            "global_O2"
                        ]["maximum_absolute_D17O_difference_full_domain"],
                        "maximum_absolute_R7_forcing_ratio_difference_high_pCO2": pressure_audit[
                            "local_R7"
                        ]["maximum_absolute_forcing_ratio_difference_high_pCO2"],
                        "source_report": _project_path(
                            Path(pressure_convention_audit_path)
                        ),
                    }
                ),
                "climate_po2_endmembers": [
                    {
                        "id": f"clima_0p5p11_{_po2_tag(report['pO2_PAL'])}pal_additive_co2_v1",
                        "pressure_convention": "additive_co2",
                        "role": "non-probabilistic structural sensitivity end member",
                        "central_model_changed": False,
                        "interpolation_permitted": False,
                        "domain": {
                            "pO2_PAL": [float(report["pO2_PAL"])] * 2,
                            "pCO2_ppm": [300.0, 60000.0],
                            "GPP_percent_of_updated_modern": [25.0, 150.0],
                        },
                        "limitations": report["promotion_decision"]["reason"],
                        "source_report": _project_path(path),
                        "values": _global_climate_rows(report),
                    }
                    for report, path in (
                        (low_po2_climate, Path(low_po2_climate_report_path))
                        if low_po2_climate is not None
                        and low_po2_climate_report_path is not None
                        else (None, None),
                        (high_po2_climate, Path(high_po2_climate_report_path))
                        if high_po2_climate is not None
                        and high_po2_climate_report_path is not None
                        else (None, None),
                    )
                    if report is not None and path is not None
                ],
                "climate_po2_interaction": (
                    None
                    if po2_cross_audit is None
                    else {
                        "status": po2_cross_audit["status"],
                        "pO2_PAL": po2_cross_audit["pO2_PAL"],
                        "maximum_absolute_global_D17O_shift_permil": po2_cross_audit[
                            "maximum_absolute_global_D17O_shift_permil"
                        ],
                        "summary_by_pO2": po2_cross_audit["summary_by_pO2"],
                        "source_report": _project_path(Path(po2_cross_audit_path)),
                    }
                ),
            },
        },
        "domain_specific_discrepancy_candidates": [
            {
                "id": item["id"],
                "eligibility": item["probabilistic_discrepancy_eligible"],
                "limit": item.get("eligibility_limit"),
                "domain": item["domain"],
                "source_report": item["source_report"],
                **(
                    {
                        "candidate_excess_predictive_sigma_permil": predictive_error[
                            "variance_decomposition"
                        ]["candidate_excess_predictive_sigma_permil"],
                        "isotope_coordinate": predictive_error["domain"][
                            "isotope_coordinate"
                        ],
                        "promoted_to_public_default": predictive_error[
                            "probabilistic_scope"
                        ]["promoted_to_public_default"],
                        "estimate_source_report": _project_path(
                            Path(predictive_error_path)
                        ),
                    }
                    if item["id"] == "yang_banerjee_low_co2_tracking"
                    and predictive_error is not None
                    else {}
                ),
            }
            for item in evidence["evidence"]
            if item["probabilistic_discrepancy_eligible"]
            not in (False, None)
        ],
        "source_files": {
            "evidence_matrix": _project_path(evidence_path),
            "climate_global_o2_report": _project_path(climate_report_path),
            "response_bundle": _project_path(response_bundle_path),
            **(
                {
                    "low_co2_predictive_error": _project_path(
                        Path(predictive_error_path)
                    )
                }
                if predictive_error_path is not None
                else {}
            ),
            **(
                {
                    "low_po2_climate_global_o2_report": _project_path(
                        Path(low_po2_climate_report_path)
                    )
                }
                if low_po2_climate_report_path is not None
                else {}
            ),
            **(
                {
                    "high_po2_climate_global_o2_report": _project_path(
                        Path(high_po2_climate_report_path)
                    )
                }
                if high_po2_climate_report_path is not None
                else {}
            ),
            **(
                {
                    "climate_po2_cross_audit": _project_path(
                        Path(po2_cross_audit_path)
                    )
                }
                if po2_cross_audit_path is not None
                else {}
            ),
            **(
                {
                    "fixed_total_climate_global_o2_report": _project_path(
                        Path(fixed_total_climate_report_path)
                    )
                }
                if fixed_total_climate_report_path is not None
                else {}
            ),
            **(
                {
                    "climate_pressure_convention_audit": _project_path(
                        Path(pressure_convention_audit_path)
                    )
                }
                if pressure_convention_audit_path is not None
                else {}
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "model_data" / "literature" / "multimodel_evidence_matrix_v1.json",
    )
    parser.add_argument(
        "--climate-report",
        type=Path,
        default=EVIDENCE_BUNDLE / "clima_global_o2_response.json",
    )
    parser.add_argument(
        "--response-bundle",
        type=Path,
        default=ROOT / "model_data" / "updated_r7_response_surface_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "model_data"
            / "uncertainty"
            / "updated_o2_uncertainty_layers_v1.json"
        ),
    )
    parser.add_argument(
        "--predictive-error",
        type=Path,
        default=EVIDENCE_BUNDLE / "yang_lowco2_predictive_error.json",
    )
    parser.add_argument(
        "--fixed-total-climate-report",
        type=Path,
        default=EVIDENCE_BUNDLE / "clima_global_o2_response_fixed_total.json",
    )
    parser.add_argument(
        "--pressure-convention-audit",
        type=Path,
        default=EVIDENCE_BUNDLE / "clima_pressure_convention_audit.json",
    )
    parser.add_argument(
        "--low-po2-climate-report",
        type=Path,
        default=EVIDENCE_BUNDLE / "clima_global_o2_response_0p1pal.json",
    )
    parser.add_argument(
        "--high-po2-climate-report",
        type=Path,
        default=EVIDENCE_BUNDLE / "clima_global_o2_response_2pal.json",
    )
    parser.add_argument(
        "--po2-cross-audit",
        type=Path,
        default=EVIDENCE_BUNDLE / "clima_po2_cross_audit.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.evidence,
                args.climate_report,
                args.response_bundle,
                args.output,
                args.predictive_error,
                args.fixed_total_climate_report,
                args.pressure_convention_audit,
                args.low_po2_climate_report,
                args.high_po2_climate_report,
                args.po2_cross_audit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
