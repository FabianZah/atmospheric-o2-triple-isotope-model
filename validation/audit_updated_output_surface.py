"""Validate the updated output accelerator against independent live-kernel points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_forward,
)
from updated_output_surface import (  # noqa: E402
    SURFACE_FIELDS,
    UpdatedMolecularOutputSurface,
    UpdatedOutputSurfaceInput,
)


CENTRAL_D17_ACCEPTANCE_PERMIL = 0.015
CAP_D17_FIELD_ACCEPTANCE_PERMIL = 0.020
DELTA18_DIAGNOSTIC_TARGET_PERMIL = 0.050


def _midpoints(values: np.ndarray) -> np.ndarray:
    return np.sqrt(values[:-1] * values[1:])


def _fractional_axis_points(values: np.ndarray, *, phase: int) -> np.ndarray:
    """Return deterministic noncentral points in each logarithmic interval."""

    fractions = np.asarray((0.2113248654, 0.3660254038, 0.6339745962, 0.7886751346))
    lower = np.log(values[:-1])
    upper = np.log(values[1:])
    fraction = fractions[(np.arange(len(lower)) + phase) % len(fractions)]
    return np.exp(lower + fraction * (upper - lower))


def _live_fields(result) -> dict[str, float]:
    return {
        "central_delta18_prime_permil": result.central_delta18_prime_permil,
        "central_cap_delta17_prime_permil": result.central_cap_delta17_prime_permil,
        "source_isoflux_lower_cap_delta17_permil": result.source_isoflux_interval_cap_delta17_permil[0],
        "source_isoflux_upper_cap_delta17_permil": result.source_isoflux_interval_cap_delta17_permil[1],
        "biological_process_lower_cap_delta17_permil": result.biological_process_interval_cap_delta17_permil[0],
        "biological_process_upper_cap_delta17_permil": result.biological_process_interval_cap_delta17_permil[1],
        "combined_process_lower_cap_delta17_permil": result.combined_process_interval_cap_delta17_permil[0],
        "combined_process_upper_cap_delta17_permil": result.combined_process_interval_cap_delta17_permil[1],
        "model_guardrail_lower_cap_delta17_permil": result.model_guardrail_interval_cap_delta17_permil[0],
        "model_guardrail_upper_cap_delta17_permil": result.model_guardrail_interval_cap_delta17_permil[1],
    }


def _interpolated_fields(surface, request) -> dict[str, float]:
    prediction = surface.evaluate(request)
    return {
        "central_delta18_prime_permil": prediction.central_delta18_prime_permil,
        "central_cap_delta17_prime_permil": prediction.central_cap_delta17_prime_permil,
        "source_isoflux_lower_cap_delta17_permil": prediction.source_isoflux_interval_cap_delta17_permil[0],
        "source_isoflux_upper_cap_delta17_permil": prediction.source_isoflux_interval_cap_delta17_permil[1],
        "biological_process_lower_cap_delta17_permil": prediction.biological_process_interval_cap_delta17_permil[0],
        "biological_process_upper_cap_delta17_permil": prediction.biological_process_interval_cap_delta17_permil[1],
        "combined_process_lower_cap_delta17_permil": prediction.combined_process_interval_cap_delta17_permil[0],
        "combined_process_upper_cap_delta17_permil": prediction.combined_process_interval_cap_delta17_permil[1],
        "model_guardrail_lower_cap_delta17_permil": prediction.interpolated_kernel_guardrail_interval_cap_delta17_permil[0],
        "model_guardrail_upper_cap_delta17_permil": prediction.interpolated_kernel_guardrail_interval_cap_delta17_permil[1],
    }


def run(
    *,
    candidate_path: Path,
    report_path: Path,
    validated_surface_path: Path | None,
    maximum_holdouts: int | None = None,
) -> dict[str, object]:
    bundle = json.loads(candidate_path.read_text(encoding="utf-8"))
    surface = UpdatedMolecularOutputSurface(bundle)
    po2_values = _fractional_axis_points(surface.po2_nodes, phase=0)
    pco2_values = _fractional_axis_points(surface.pco2_nodes, phase=1)
    gpp_values = _fractional_axis_points(surface.gpp_nodes, phase=2)
    all_conditions = [
        (float(po2), float(pco2), float(gpp))
        for po2 in po2_values
        for pco2 in pco2_values
        for gpp in gpp_values
    ]
    if maximum_holdouts is not None and maximum_holdouts < len(all_conditions):
        low_gpp_conditions = [
            (float(po2), float(pco2), float(gpp_values[k]))
            for po2 in po2_values
            for pco2 in pco2_values
            for k in range(min(2, len(gpp_values)))
        ]
        if maximum_holdouts < len(low_gpp_conditions):
            raise ValueError(
                "stratified holdout count must cover the two lowest-GPP "
                "midpoints at every pO2-pCO2 cell"
            )
        selected = list(low_gpp_conditions)
        selected_set = set(selected)
        remaining_count = maximum_holdouts - len(selected)
        candidates = [item for item in all_conditions if item not in selected_set]
        if remaining_count:
            indices = np.unique(
                np.round(
                    np.linspace(0, len(candidates) - 1, remaining_count)
                ).astype(int)
            )
            selected.extend(candidates[index] for index in indices)
        conditions = selected
        holdout_design = (
            "Noncentral logarithmic fractions in the two lowest-GPP intervals "
            "for every pO2-pCO2 cell, plus a deterministic evenly indexed "
            "sample across all remaining GPP intervals."
        )
    else:
        conditions = all_conditions
        holdout_design = (
            "Deterministic noncentral logarithmic fraction in every adjacent "
            "pO2-pCO2-GPP training cell."
        )
    rows: list[dict[str, object]] = []
    field_residuals = {key: [] for key in SURFACE_FIELDS}
    started = time.perf_counter()
    case_index = 0
    case_count = len(conditions)
    for po2, pco2, gpp in conditions:
                request = UpdatedOutputSurfaceInput(
                    float(po2), float(pco2), float(gpp)
                )
                live = run_updated_forward(
                    UpdatedForwardInput(
                        request.p_o2_pal,
                        request.p_co2_ppm,
                        request.gpp_pgC_per_year,
                    )
                )
                live_values = _live_fields(live)
                interpolated = _interpolated_fields(surface, request)
                row: dict[str, object] = {
                    "pO2_PAL": request.p_o2_pal,
                    "pCO2_ppm": request.p_co2_ppm,
                    "GPP_PgC_per_year": request.gpp_pgC_per_year,
                }
                for key in SURFACE_FIELDS:
                    residual = interpolated[key] - live_values[key]
                    field_residuals[key].append(residual)
                    row[f"live_{key}"] = live_values[key]
                    row[f"interpolated_{key}"] = interpolated[key]
                    row[f"residual_{key}"] = residual
                rows.append(row)
                case_index += 1
                if case_index % 25 == 0 or case_index == case_count:
                    print(
                        f"validated pO2={po2:.5g} PAL, pCO2={pco2:.6g} ppm "
                        f"({case_index}/{case_count})",
                        flush=True,
                    )
    live_seconds = time.perf_counter() - started

    acceleration_started = time.perf_counter()
    for _ in range(10):
        for row in rows:
            surface.evaluate(
                UpdatedOutputSurfaceInput(
                    float(row["pO2_PAL"]),
                    float(row["pCO2_ppm"]),
                    float(row["GPP_PgC_per_year"]),
                )
            )
    accelerated_seconds = (time.perf_counter() - acceleration_started) / 10.0

    metrics = {}
    for key, values in field_residuals.items():
        residuals = np.asarray(values, dtype=float)
        maximum_index = int(np.argmax(np.abs(residuals)))
        metrics[key] = {
            "mean_absolute_residual_permil": float(np.mean(np.abs(residuals))),
            "maximum_absolute_residual_permil": float(np.max(np.abs(residuals))),
            "maximum_residual_case": {
                "pO2_PAL": rows[maximum_index]["pO2_PAL"],
                "pCO2_ppm": rows[maximum_index]["pCO2_ppm"],
                "GPP_PgC_per_year": rows[maximum_index]["GPP_PgC_per_year"],
                "signed_residual_permil": float(residuals[maximum_index]),
            },
        }

    cap_fields = [key for key in SURFACE_FIELDS if "cap_delta17" in key]
    maximum_cap_field_residual = max(
        float(metrics[key]["maximum_absolute_residual_permil"])
        for key in cap_fields
    )
    central_residual = float(
        metrics["central_cap_delta17_prime_permil"][
            "maximum_absolute_residual_permil"
        ]
    )
    delta18_residual = float(
        metrics["central_delta18_prime_permil"][
            "maximum_absolute_residual_permil"
        ]
    )
    gates = {
        "central_cap_delta17_below_0p015_permil": (
            central_residual <= CENTRAL_D17_ACCEPTANCE_PERMIL
        ),
        "all_cap_delta17_fields_below_0p020_permil": (
            maximum_cap_field_residual <= CAP_D17_FIELD_ACCEPTANCE_PERMIL
        ),
        "all_live_kernel_points_converged": True,
        "central_delta18_below_0p050_permil": (
            delta18_residual <= DELTA18_DIAGNOSTIC_TARGET_PERMIL
        ),
    }
    accepted = all(gates.values())
    report = {
        "audit": "updated molecular output-surface noncentral holdouts",
        "status": "accepted" if accepted else "grid_refinement_required",
        "candidate_surface_data_id": surface.surface_data_id,
        "upstream_model_data_id": surface.upstream_model_data_id,
        "holdout_design": holdout_design,
        "case_count": case_count,
        "gates": gates,
        "metrics": metrics,
        "maximum_cap_delta17_field_residual_permil": maximum_cap_field_residual,
        "delta18_diagnostic": {
            "acceleration_validated": (
                delta18_residual <= DELTA18_DIAGNOSTIC_TARGET_PERMIL
            ),
            "target_permil": DELTA18_DIAGNOSTIC_TARGET_PERMIL,
            "target_met": delta18_residual <= DELTA18_DIAGNOSTIC_TARGET_PERMIL,
            "reason": "Independent live-kernel noncentral holdout validation.",
        },
        "timing": {
            "live_kernel_seconds_for_holdouts": live_seconds,
            "accelerated_seconds_for_holdouts": accelerated_seconds,
            "measured_speedup": live_seconds / accelerated_seconds,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with report_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if accepted and validated_surface_path is not None:
        bundle["surface_data_id"] = "updated_molecular_output_surface_v1"
        bundle["interpolation"] = {
            "central_cap_delta17": "tensor-product cubic",
            "uncertainty_bounds": (
                "tensor-product cubic interpolation of logarithmic positive "
                "distance from central Delta-prime-17O"
            ),
            "coordinates": "pO2, pCO2, and natural log of absolute GPP",
            "extrapolation_permitted": False,
            "central_delta18": surface._delta18_interpolation_method,
        }
        bundle["validation"] = {
            "status": "accepted_independent_noncentral_holdouts",
            "holdout_case_count": case_count,
            "holdout_design": report["holdout_design"],
            "central_cap_delta17_maximum_absolute_residual_permil": central_residual,
            "cap_delta17_maximum_absolute_residual_permil": maximum_cap_field_residual,
            "central_delta18_maximum_absolute_residual_permil": delta18_residual,
            "delta18_acceleration_validated": (
                delta18_residual <= DELTA18_DIAGNOSTIC_TARGET_PERMIL
            ),
            "acceptance_thresholds_permil": {
                "central_cap_delta17": CENTRAL_D17_ACCEPTANCE_PERMIL,
                "all_cap_delta17_fields": CAP_D17_FIELD_ACCEPTANCE_PERMIL,
                "central_delta18_diagnostic_target": DELTA18_DIAGNOSTIC_TARGET_PERMIL,
            },
            "report": str(report_path.resolve().relative_to(ROOT)).replace("\\", "/"),
        }
        if "delta18_surface" in bundle:
            bundle["delta18_surface"]["validation"] = {
                "status": "accepted_independent_noncentral_holdouts",
                "holdout_case_count": case_count,
                "maximum_absolute_residual_permil": delta18_residual,
                "acceptance_threshold_permil": DELTA18_DIAGNOSTIC_TARGET_PERMIL,
                "report": str(report_path.resolve().relative_to(ROOT)).replace(
                    "\\", "/"
                ),
            }
        validated_surface_path.parent.mkdir(parents=True, exist_ok=True)
        validated_surface_path.write_text(
            json.dumps(bundle, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "model_data" / "updated_molecular_output_surface_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "updated_molecular_output_surface_audit.json",
    )
    parser.add_argument(
        "--validated-surface",
        type=Path,
        default=None,
    )
    parser.add_argument("--maximum-holdouts", type=int)
    args = parser.parse_args()
    report = run(
        candidate_path=args.candidate,
        report_path=args.report,
        validated_surface_path=args.validated_surface,
        maximum_holdouts=args.maximum_holdouts,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
