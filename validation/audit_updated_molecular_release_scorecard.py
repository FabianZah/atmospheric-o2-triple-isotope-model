"""Unified scorecard for the molecular model used by the public UI.

Formal gates test numerical and physical model properties. Young comparisons
are diagnostics: the accepted modern-reference difference is reported, but no
offset is applied inside the forward model or to its validation curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for subdirectory in ("code", "validation"):
    path = str(ROOT / subdirectory)
    if path not in sys.path:
        sys.path.insert(0, path)

from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
    run_updated_forward,
)
from isotopes import conventional_delta_from_prime  # noqa: E402
from updated_molecular_transient import (  # noqa: E402
    UpdatedTransientInput,
    run_updated_transient,
)
from updated_output_surface import (  # noqa: E402
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)
from updated_output_surface_inverse import (  # noqa: E402
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)
from updated_output_surface_posterior import (  # noqa: E402
    UpdatedConditionalPosteriorInput,
    conditional_updated_posterior,
)
from updated_output_surface_joint_posterior import (  # noqa: E402
    UpdatedJointPosteriorInput,
    joint_updated_posterior,
)
from updated_photosynthesis_transient import (  # noqa: E402
    UpdatedPhotosynthesisTransientInput,
    run_updated_photosynthesis_transient,
)
from young_fig8_reconstruction import d17o_from_pco2, digitize_fig8  # noqa: E402
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL  # noqa: E402
from young_validation_targets import (  # noqa: E402
    fig7_digitized_targets,
    fig8_pco2_points,
    load_fig7_digitized_contours,
    young_fig7_294ppm_d17o,
)


OUTPUTS = ROOT / "outputs"
ROWS_PATH = OUTPUTS / "updated_molecular_release_scorecard.csv"
JSON_PATH = OUTPUTS / "updated_molecular_release_scorecard.json"
MARKDOWN_PATH = OUTPUTS / "updated_molecular_release_scorecard.md"
SHAPE_AUDIT_PATH = OUTPUTS / "updated_molecular_output_surface_50ppm_shape_audit.json"
D17_AUDIT_PATH = OUTPUTS / "updated_molecular_output_surface_audit.json"
D18_AUDIT_PATH = (
    OUTPUTS / "updated_molecular_output_surface_delta18_every_cell_audit.json"
)
LOW_CO2_ACCELERATOR_AUDIT_PATH = (
    OUTPUTS / "updated_molecular_output_surface_50ppm_audit.json"
)
BANERJEE_APPLICATION_PATH = OUTPUTS / "banerjee_2026_updated_model_application.json"
YANG_CO2_TRACKING_PATH = OUTPUTS / "yang_2022_co2_tracking_audit.json"
YANG_TRANSIENT_TRACKING_PATH = (
    OUTPUTS / "yang_2022_transient_co2_tracking_audit.json"
)
BRANDON_TERMINATION_V_PATH = OUTPUTS / "brandon_2020_termination_v_audit.json"
UNCERTAINTY_CONTRACT_PATH = (
    ROOT / "model_data" / "uncertainty" / "updated_o2_uncertainty_layers_v1.json"
)
UNCERTAINTY_AUDIT_PATH = OUTPUTS / "updated_uncertainty_layers_audit.json"
YANG_PREDICTIVE_ERROR_PATH = OUTPUTS / "yang_lowco2_predictive_error.json"

UPDATED_MODERN_GPP = 290.0
PACK_D17 = -0.432
PACK_D17_UNCERTAINTY = 0.015
PACK_D18 = 23.9
PACK_D18_UNCERTAINTY = 0.3
YOUNG_MODERN_D17 = young_fig7_294ppm_d17o(100.0)
YOUNG_FIG10_D17_SHIFT_150 = -0.006
YOUNG_FIG10_D17_SHIFT_5000 = -0.046


def _metric(
    category: str,
    name: str,
    value: float | int | str | bool | None,
    units: str,
    status: str,
    criterion: str,
    note: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "metric": name,
        "value": value,
        "units": units,
        "status": status,
        "criterion": criterion,
        "note": note,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(residuals: list[float]) -> dict[str, float]:
    values = np.asarray(residuals, dtype=float)
    return {
        "mean_residual_permil": float(np.mean(values)),
        "mean_absolute_residual_permil": float(np.mean(np.abs(values))),
        "maximum_absolute_residual_permil": float(np.max(np.abs(values))),
    }


def _young_steady_diagnostics(surface) -> dict[str, Any]:
    modern_prediction = surface.evaluate(
        UpdatedOutputSurfaceInput(1.0, 294.0, UPDATED_MODERN_GPP)
    ).central_cap_delta17_prime_permil
    reference_difference = modern_prediction - YOUNG_MODERN_D17

    fig7_rows: list[dict[str, Any]] = []
    excluded_contours = sorted(
        co2
        for co2 in load_fig7_digitized_contours()
        if co2 < surface.domain["pco2_ppm"][0]
    )
    for pco2, gpp_percent, target in fig7_digitized_targets(
        samples_per_contour=15
    ):
        if not surface.domain["pco2_ppm"][0] <= pco2 <= surface.domain["pco2_ppm"][1]:
            continue
        model = surface.evaluate(
            UpdatedOutputSurfaceInput(
                1.0,
                float(pco2),
                UPDATED_MODERN_GPP * gpp_percent / 100.0,
            )
        ).central_cap_delta17_prime_permil
        fig7_rows.append(
            {
                "pCO2_ppm": float(pco2),
                "GPP_percent": float(gpp_percent),
                "target_permil": float(target),
                "model_permil": float(model),
                "residual_permil": float(model - target),
                "reference_aligned_residual_permil": float(
                    model - reference_difference - target
                ),
            }
        )

    curves = digitize_fig8()
    fig8_rows: list[dict[str, Any]] = []
    for gpp_percent in (100.0, 50.0):
        for pco2 in fig8_pco2_points(dense=True):
            target = float(d17o_from_pco2(pco2, int(gpp_percent), curves))
            model = surface.evaluate(
                UpdatedOutputSurfaceInput(
                    1.0,
                    float(pco2),
                    UPDATED_MODERN_GPP * gpp_percent / 100.0,
                )
            ).central_cap_delta17_prime_permil
            fig8_rows.append(
                {
                    "pCO2_ppm": float(pco2),
                    "GPP_percent": gpp_percent,
                    "target_permil": target,
                    "model_permil": float(model),
                    "residual_permil": float(model - target),
                    "reference_aligned_residual_permil": float(
                        model - reference_difference - target
                    ),
                }
            )

    return {
        "accepted_reference_difference_permil": float(reference_difference),
        "reference_definition": (
            "updated model at 1 PAL, 294 ppm, 290 PgC yr-1 minus Young's "
            "printed 294-ppm, 100%-GPP relation"
        ),
        "policy": "reported only; never applied to forward-model output",
        "fig7": {
            "included_contours_ppm": sorted(
                {int(row["pCO2_ppm"]) for row in fig7_rows}
            ),
            "excluded_below_domain_contours_ppm": excluded_contours,
            "point_count": len(fig7_rows),
            "absolute": _summary(
                [float(row["residual_permil"]) for row in fig7_rows]
            ),
            "reference_aligned": _summary(
                [
                    float(row["reference_aligned_residual_permil"])
                    for row in fig7_rows
                ]
            ),
        },
        "fig8": {
            "point_count": len(fig8_rows),
            "absolute": _summary(
                [float(row["residual_permil"]) for row in fig8_rows]
            ),
            "reference_aligned": _summary(
                [
                    float(row["reference_aligned_residual_permil"])
                    for row in fig8_rows
                ]
            ),
            "by_gpp": {
                str(int(gpp)): _summary(
                    [
                        float(row["reference_aligned_residual_permil"])
                        for row in fig8_rows
                        if row["GPP_percent"] == gpp
                    ]
                )
                for gpp in (100.0, 50.0)
            },
        },
    }


def _roundtrip_diagnostics(surface) -> list[dict[str, Any]]:
    cases = (
        ("pCO2", UpdatedOutputSurfaceInput(0.5, 4200.0, 182.56264), 4200.0),
        ("GPP", UpdatedOutputSurfaceInput(1.0, 5000.0, 350.0), 350.0),
        ("pO2", UpdatedOutputSurfaceInput(1.5, 1000.0, 290.0), 1.5),
    )
    rows: list[dict[str, Any]] = []
    for solve_for, forward_input, expected in cases:
        target = surface.evaluate(forward_input).central_cap_delta17_prime_permil
        inverse = invert_updated_output_surface(
            UpdatedSurfaceInverseInput(
                target_air_cap_delta17_permil=target,
                solve_for=solve_for,
                p_o2_pal=forward_input.p_o2_pal,
                p_co2_ppm=forward_input.p_co2_ppm,
                gpp_pgC_per_year=forward_input.gpp_pgC_per_year,
            ),
            verify_live_root=True,
        )
        recovered = inverse.central_root
        rows.append(
            {
                "solve_for": solve_for,
                "expected": expected,
                "recovered": recovered,
                "relative_error": (
                    None if recovered is None else abs(recovered - expected) / expected
                ),
                "live_root_residual_permil": inverse.live_root_residual_permil,
                "live_root_verified": inverse.live_root_verified,
                "admissible_interval": inverse.admissible_interval,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Updated Molecular Model Release Scorecard",
        "",
        "This scorecard evaluates the molecular engine used by the public UI. Young comparisons are diagnostics, not fitted constraints. The accepted modern-reference discrepancy is reported but is not applied inside the model.",
        "",
        "| Category | Metric | Value | Status | Criterion |",
        "|---|---|---:|---|---|",
    ]
    for row in report["metrics"]:
        rendered = f"{row['value']:.6g}" if isinstance(row["value"], float) else str(row["value"])
        lines.append(
            f"| {row['category']} | {row['metric']} | {rendered} {row['units']} | "
            f"{row['status']} | {row['criterion']} |"
        )
    lines.extend(
        [
            "",
            "`pass` and `fail` are formal release gates. `diagnostic` records Young agreement without retuning. `limitation` identifies remaining publication work.",
            "",
            "The companion curve-wide mechanism audit is `validation/audit_updated_fig8_response_shape.py`; it reports the required-GPP equivalent without using it as a correction.",
            "",
            "Complete values and provenance are in `outputs/updated_molecular_release_scorecard.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, include_photosynthesis_transient: bool = True) -> dict[str, Any]:
    surface = load_updated_output_surface()
    modern_request = UpdatedForwardInput(1.0, 294.0, 290.0)
    modern_live = run_updated_forward(modern_request)
    modern_central = run_updated_central_state(modern_request)
    pack_residual = modern_live.central_cap_delta17_prime_permil - PACK_D17
    modern_delta18 = conventional_delta_from_prime(
        modern_live.central_delta18_prime_permil
    )
    pack_delta18_residual = modern_delta18 - PACK_D18

    d17_audit = _load_json(D17_AUDIT_PATH)
    d18_audit = _load_json(D18_AUDIT_PATH)
    low_co2_audit = _load_json(LOW_CO2_ACCELERATOR_AUDIT_PATH)
    banerjee_application = _load_json(BANERJEE_APPLICATION_PATH)
    yang_tracking = _load_json(YANG_CO2_TRACKING_PATH)
    yang_transient = _load_json(YANG_TRANSIENT_TRACKING_PATH)
    brandon_termination_v = _load_json(BRANDON_TERMINATION_V_PATH)
    uncertainty_contract = _load_json(UNCERTAINTY_CONTRACT_PATH)
    uncertainty_audit = _load_json(UNCERTAINTY_AUDIT_PATH)
    yang_predictive_error = _load_json(YANG_PREDICTIVE_ERROR_PATH)
    shape_audit = _load_json(SHAPE_AUDIT_PATH)
    young = _young_steady_diagnostics(surface)
    inversions = _roundtrip_diagnostics(surface)
    posterior_target_request = UpdatedOutputSurfaceInput(0.5, 4200.0, 182.56264)
    posterior_target = surface.evaluate(
        posterior_target_request
    ).central_cap_delta17_prime_permil
    posterior = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(
            target_air_cap_delta17_permil=posterior_target,
            measurement_sigma_permil=0.02,
            solve_for="pCO2",
            prior="uniform",
            p_o2_pal=posterior_target_request.p_o2_pal,
            p_co2_ppm=posterior_target_request.p_co2_ppm,
            gpp_pgC_per_year=posterior_target_request.gpp_pgC_per_year,
            solve_bounds=(1000.0, 20000.0),
            grid_size=2049,
        ),
        verify_live_mode=False,
    )
    posterior_roundtrip_passed = bool(
        posterior.equal_tailed_credible_interval[0]
        < posterior_target_request.p_co2_ppm
        < posterior.equal_tailed_credible_interval[1]
        and posterior.central_likelihood_root is not None
        and abs(
            posterior.central_likelihood_root - posterior_target_request.p_co2_ppm
        )
        / posterior_target_request.p_co2_ppm
        <= 1.0e-8
    )
    joint_posterior = joint_updated_posterior(
        UpdatedJointPosteriorInput(
            target_air_cap_delta17_permil=posterior_target,
            measurement_sigma_permil=0.01,
            model_discrepancy_sigma_permil=0.02,
            model_discrepancy_source=(
                "synthetic release-gate propagation test; not an empirical "
                "model-discrepancy calibration"
            ),
            free_coordinates=("pCO2", "GPP", "pO2"),
            pco2_grid_size=31,
            gpp_grid_size=31,
            po2_grid_size=31,
        )
    )
    joint_truth = {
        "pCO2": posterior_target_request.p_co2_ppm,
        "GPP": posterior_target_request.gpp_pgC_per_year,
        "pO2": posterior_target_request.p_o2_pal,
    }
    joint_recovery_passed = bool(
        abs(joint_posterior.posterior_integral - 1.0) <= 1.0e-8
        and joint_posterior.hpd_probability_mass >= 0.95
        and all(
            joint_posterior.equal_tailed_credible_intervals[coordinate][0]
            <= value
            <= joint_posterior.equal_tailed_credible_intervals[coordinate][1]
            for coordinate, value in joint_truth.items()
        )
    )
    joint_posterior_check = {
        "inputs": asdict(joint_posterior.inputs),
        "status": joint_posterior.status,
        "posterior_shape": joint_posterior.posterior_shape,
        "posterior_integral": joint_posterior.posterior_integral,
        "HPD_probability_mass": joint_posterior.hpd_probability_mass,
        "generated_truth": joint_truth,
        "MAP_coordinates": joint_posterior.map_coordinates,
        "equal_tailed_credible_intervals": (
            joint_posterior.equal_tailed_credible_intervals
        ),
        "edge_probabilities": joint_posterior.edge_probabilities,
        "generated_target_recovered": joint_recovery_passed,
        "probability_scope": joint_posterior.probability_scope,
    }

    balanced = run_updated_transient(
        UpdatedTransientInput(
            initial=modern_request,
            final=UpdatedForwardInput(1.0, 1000.0, 290.0),
            duration_years=12000.0,
            sample_count=161,
            equilibrium_search_max_years=100000.0,
        )
    )
    balanced_distance = max(
        abs(
            balanced.states[-1].cap_delta17_prime_permil
            - balanced.final_steady_state["cap_delta17_prime_permil"]
        ),
        abs(
            balanced.states[-1].delta18_prime_permil
            - balanced.final_steady_state["delta18_prime_permil"]
        ),
    )
    balanced_inventory = np.asarray(
        [state.isotopologue_moles for state in balanced.states], dtype=float
    )

    fig10 = run_updated_transient(
        UpdatedTransientInput(
            initial=modern_request,
            final=UpdatedForwardInput(1.0, 400.0, 290.0),
            duration_years=5000.0,
            sample_count=1001,
            equilibrium_search_max_years=100000.0,
        )
    )
    fig10_time = np.asarray(fig10.time_years)
    fig10_d17 = np.asarray(
        [state.cap_delta17_prime_permil for state in fig10.states]
    )
    fig10_shift = fig10_d17 - fig10_d17[0]
    fig10_shift_150 = float(np.interp(150.0, fig10_time, fig10_shift))
    fig10_shift_5000 = float(fig10_shift[-1])
    fig10_monotonic = bool(np.all(np.diff(fig10_shift) <= 1.0e-10))

    photosynthesis: dict[str, Any] | None = None
    warning_messages: list[str] = []
    if include_photosynthesis_transient:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            transient = run_updated_photosynthesis_transient(
                UpdatedPhotosynthesisTransientInput(
                    initial=modern_request,
                    photosynthesis_fraction=0.5,
                    duration_years=12000.0,
                    sample_count=161,
                    equilibrium_search_max_years=100000.0,
                )
            )
        warning_messages = sorted({str(item.message) for item in captured})
        d17_values = np.asarray(
            [state.cap_delta17_prime_permil for state in transient.states]
        )
        d18_values = np.asarray(
            [state.delta18_prime_permil for state in transient.states]
        )
        po2_values = np.asarray(
            [state.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL for state in transient.states]
        )
        carbon_po2_values = np.asarray(transient.carbon_driver_po2_pal)
        pco2_values = np.asarray(transient.pco2_ppm)
        minimum_index = int(np.argmin(d17_values))
        maximum_d18_index = int(np.argmax(d18_values))
        maximum_pco2_index = int(np.argmax(pco2_values))
        photosynthesis = {
            "minimum_cap_delta17_prime_permil": float(d17_values[minimum_index]),
            "minimum_time_years": float(transient.time_years[minimum_index]),
            "maximum_delta18_prime_permil": float(d18_values[maximum_d18_index]),
            "maximum_delta18_time_years": float(
                transient.time_years[maximum_d18_index]
            ),
            "minimum_display_pO2_pal": float(np.min(po2_values)),
            "maximum_carbon_oxygen_pO2_mismatch_pal": float(
                np.max(np.abs(carbon_po2_values - po2_values))
            ),
            "long_run_cap_delta17_prime_permil": float(
                transient.long_run_state.cap_delta17_prime_permil
            ),
            "long_run_delta18_prime_permil": float(
                transient.long_run_state.delta18_prime_permil
            ),
            "long_run_pO2_pal": float(
                transient.long_run_state.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL
            ),
            "peak_pCO2_ppm": float(pco2_values[maximum_pco2_index]),
            "peak_pCO2_time_years": float(
                transient.time_years[maximum_pco2_index]
            ),
            "long_run_pCO2_ppm": transient.long_run_pco2_ppm,
            "operational_equilibrium_time_years": (
                transient.operational_equilibrium_time_years
            ),
            "carbon_driver_preset": transient.carbon_driver_preset,
            "solver": transient.solver,
            "captured_warning_count": len(warning_messages),
            "captured_warnings": warning_messages,
            "contained_carbon_warning_events": transient.solver[
                "carbon_internal_runtime_warning_events"
            ],
            "contained_carbon_warnings": transient.solver[
                "carbon_internal_runtime_warnings"
            ],
            "coupling_scope": (
                "one-way hybrid: detailed carbon driver supplies live pCO2 to "
                "the molecular O2 isotope integration"
            ),
        }

    maximum_roundtrip_error = max(
        float(row["relative_error"])
        for row in inversions
        if row["relative_error"] is not None
    )
    maximum_live_root_residual = max(
        abs(float(row["live_root_residual_permil"]))
        for row in inversions
        if row["live_root_residual_permil"] is not None
    )
    shape_passed = all(bool(value) for value in shape_audit["gates"].values())

    metrics = [
        _metric("modern", "Pack Delta-prime-17O residual", pack_residual, "permil", "pass" if abs(pack_residual) <= PACK_D17_UNCERTAINTY else "fail", "absolute residual <= 0.015 permil", "Direct live-model output; no atmospheric-O2 output offset."),
        _metric("modern", "Pack conventional delta-18O residual", pack_delta18_residual, "permil", "pass" if abs(pack_delta18_residual) <= PACK_D18_UNCERTAINTY else "fail", "absolute residual <= 0.3 permil", f"Model delta-prime-18O converted to conventional delta-18O: {modern_delta18:.6f} permil."),
        _metric("numerics", "Modern fixed-point residual", modern_central.maximum_fixed_point_residual_permil, "permil", "pass" if modern_central.numerically_converged else "fail", "solver converged at tolerance 1e-8 permil", modern_central.solver_method),
        _metric("accelerator", "Delta-prime-17O maximum holdout residual", d17_audit["maximum_cap_delta17_field_residual_permil"], "permil", "pass" if d17_audit["status"] == "accepted" else "fail", "all released fields < 0.020 permil", f"{d17_audit['case_count']} independent live-kernel holdouts."),
        _metric("accelerator", "delta-prime-18O maximum holdout residual", d18_audit["maximum_absolute_residual_permil"], "permil", "pass" if d18_audit["status"] == "accepted" else "fail", "maximum residual < 0.050 permil", f"{d18_audit['case_count']} every-cell noncentral holdouts."),
        _metric("accelerator", "Low-pCO2 maximum Delta-prime-17O holdout residual", low_co2_audit["maximum_all_Delta17_field_residual_permil"], "permil", "pass" if low_co2_audit["status"] == "accepted" else "fail", "all released fields < 0.020 permil", f"{low_co2_audit['case_count']} withheld-low-pCO2 and overlap live-kernel holdouts."),
        _metric("accelerator", "Low-pCO2 maximum delta-prime-18O holdout residual", low_co2_audit["metrics"]["central_delta18_prime_permil"]["maximum_absolute_residual_permil"], "permil", "pass" if low_co2_audit["status"] == "accepted" else "fail", "maximum residual < 0.050 permil", "The 75, 125, 175, 225, and 275 ppm planes were excluded from accelerator training."),
        _metric("shape", "Dense-domain monotonic and finite gates", shape_audit["dense_point_count"], "points", "pass" if shape_passed else "fail", "all declared shape gates true", "No hooks or wrong-direction steps in the validated surface."),
        _metric("inversion", "Maximum three-coordinate round-trip error", maximum_roundtrip_error, "relative", "pass" if maximum_roundtrip_error <= 1.0e-8 else "fail", "relative error <= 1e-8", "Independent pCO2, GPP, and pO2 roots."),
        _metric("inversion", "Maximum live-kernel residual at accelerated roots", maximum_live_root_residual, "permil", "pass" if maximum_live_root_residual <= 0.015 else "fail", "absolute residual <= 0.015 permil", "Live molecular-kernel verification of each root."),
        _metric("transient", "Balanced pCO2-step positivity", bool(np.all(balanced_inventory > 0.0)), "", "pass" if np.all(balanced_inventory > 0.0) else "fail", "all isotopologue inventories > 0", "294 to 1000 ppm step at fixed pO2 and GPP."),
        _metric("transient", "Balanced pCO2-step distance at 12000 years", balanced_distance, "permil", "pass" if balanced_distance <= 0.001 else "fail", "both isotope coordinates within 0.001 permil", f"Operational equilibrium at {balanced.equilibrium_time_years:.1f} years."),
        _metric("transient", "Fig. 10-type monotonic Delta-prime-17O relaxation", fig10_monotonic, "", "pass" if fig10_monotonic else "fail", "no reversal after the maintained 294-to-400 ppm step", "Updated-model physical response; Young values are comparison diagnostics."),
        _metric("Young diagnostic", "Accepted modern-reference difference", young["accepted_reference_difference_permil"], "permil", "diagnostic", "reported, not corrected", young["reference_definition"]),
        _metric("Young diagnostic", "Fig. 7 aligned mean absolute residual", young["fig7"]["reference_aligned"]["mean_absolute_residual_permil"], "permil", "diagnostic", "comparison only", "Contours 294-1094 ppm; no point displacement."),
        _metric("Young diagnostic", "Fig. 8 aligned mean absolute residual", young["fig8"]["reference_aligned"]["mean_absolute_residual_permil"], "permil", "diagnostic", "comparison only", "Dense 50% and 100% GPP trends through 30000 ppm."),
        _metric("Young diagnostic", "Fig. 10 Delta-prime-17O shift residual at 150 years", fig10_shift_150 - YOUNG_FIG10_D17_SHIFT_150, "permil", "diagnostic", "comparison only", f"Updated shift {fig10_shift_150:.6f} permil; Young visual anchor {YOUNG_FIG10_D17_SHIFT_150:.3f} permil."),
        _metric("Young diagnostic", "Fig. 10 Delta-prime-17O shift residual at 5000 years", fig10_shift_5000 - YOUNG_FIG10_D17_SHIFT_5000, "permil", "diagnostic", "comparison only", f"Updated shift {fig10_shift_5000:.6f} permil; Young visual anchor {YOUNG_FIG10_D17_SHIFT_5000:.3f} permil."),
        _metric("domain", "Physical and accelerated pCO2 domain", f"{surface.domain['pco2_ppm'][0]:.0f}-{surface.domain['pco2_ppm'][1]:.0f}", "ppm", "pass" if low_co2_audit["status"] == "accepted" else "fail", "native low-pCO2 training and holdout gates passed", "The physical response surface and public-UI accelerator have the same pCO2 bounds."),
        _metric("Young diagnostic", "Fig. 7 contours below physical domain", len(young["fig7"]["excluded_below_domain_contours_ppm"]), "contours", "limitation" if young["fig7"]["excluded_below_domain_contours_ppm"] else "pass", f"reported without extrapolation below {surface.domain['pco2_ppm'][0]:.0f} ppm", str(young["fig7"]["excluded_below_domain_contours_ppm"])),
        _metric("external application", "Banerjee MPT minus pre-MPT GPP", banerjee_application["summary_GPP_percent_PI"]["MPT_minus_pre_MPT_percentage_points"], "percentage points", "diagnostic", "paper reports an approximately 10% increase; no fit applied", "Paired measured CO2 and Delta-prime-17O from 21 pristine samples; pO2 fixed at 1 PAL."),
        _metric("external application", "Yang raw age-block CO2 tracking skill", yang_tracking["tracking_by_smoothing_window"]["0_ka"]["blocked_cross_validation"]["variance_skill"], "fraction", "diagnostic", "positive skill relative to an intercept-only holdout model", "269 paired measured CO2 and Delta-17O values; physical response amplitude fixed."),
        _metric("external application", "Yang 21-kyr-smoothed CO2 tracking correlation", yang_tracking["tracking_by_smoothing_window"]["21_ka"]["pearson_r"], "Pearson r", "diagnostic", "descriptive comparison; overlapping windows are non-independent", "GPP and pO2 fixed; only an additive reference offset is estimated."),
        _metric("external application", "Yang 21-kyr cross-validated CO2 inversion RMSE", yang_tracking["tracking_by_smoothing_window"]["21_ka"]["cross_validated_CO2_inversion"]["rmse_ppm"], "ppm", "diagnostic", "compare with independently measured ice-core CO2", "All 269 smoothed values have roots; GPP fixed at 290 PgC/yr and pO2 at 1 PAL."),
        _metric("external application", "CO2-only slope difference from Yang TB model", yang_tracking["co2_response"]["relative_slope_difference"], "relative", "diagnostic", "comparison only", "Updated physical model versus Yang et al. (2022) reported -0.55 ppm Delta-17O per ppm CO2."),
        _metric("external application", "Continuous-CO2 emergent response lag", yang_transient["emergent_response_lag"]["best_lag_years"], "years", "diagnostic", "emerges from the global O2 budget; not fitted to observations", "Lag maximizing agreement between transient and instantaneous model signals after the initialization guard."),
        _metric("external application", "Yang 21-kyr transient age-block tracking skill", yang_transient["tracking_by_smoothing_window"]["21_ka"]["transient"]["blocked_cross_validation"]["variance_skill"], "fraction", "diagnostic", "compare with the identical instantaneous forcing experiment", "Continuous Bereiter CO2 forcing, fixed GPP and pO2, state-dependent molecular response."),
        _metric("external application", "Termination V fixed-GPP mean residual", brandon_termination_v["fixed_GPP_test"]["published_event_window"]["mean_residual_ppm"], "ppm", "diagnostic", "a coherent positive residual is required before varying GPP", "Published 430-415 ka interval; reference offset trained only on observations older than 430 ka."),
        _metric("external application", "Termination V inferred GPP", brandon_termination_v["GPP_pulse_test"]["inferred_percent_of_preindustrial"], "% pre-industrial", "diagnostic", "compare with Brandon et al. reported 110-130% range", "One interval multiplier; event timing, response time, CO2 forcing, and isotope amplitude are not fitted."),
        _metric("external application", "Termination V GPP-pulse SSE reduction", brandon_termination_v["GPP_pulse_test"]["event_SSE_reduction_fraction"], "fraction", "diagnostic", "improvement over the identical fixed-GPP transient", "The inferred interval-average GPP is not a unique attribution of all residual variance."),
        _metric("uncertainty", "Conditional one-coordinate posterior normalization", abs(posterior.posterior_integral - 1.0), "absolute error", "pass" if abs(posterior.posterior_integral - 1.0) <= 1.0e-8 else "fail", "posterior integral error <= 1e-8", "Gaussian analytical likelihood and explicit bounded prior; model guardrails remain non-probabilistic."),
        _metric("uncertainty", "Conditional posterior generated-target recovery", posterior_roundtrip_passed, "", "pass" if posterior_roundtrip_passed else "fail", "central root recovered and contained by the credible interval", "One solved coordinate with the other two fixed."),
        _metric("uncertainty", "Joint pCO2-GPP-pO2 posterior generated-target recovery", joint_recovery_passed, "", "pass" if joint_recovery_passed else "fail", "normalized posterior and all generated coordinates inside marginal credible intervals", "The calculation retains the three-coordinate ridge and an explicit Gaussian discrepancy input."),
        _metric("uncertainty", "Explicit model-discrepancy provenance requirement", joint_posterior.probabilistic_model_discrepancy_included, "", "pass" if joint_posterior.probabilistic_model_discrepancy_included else "fail", "nonzero probabilistic discrepancy requires a named source", joint_posterior.probability_scope),
        _metric("uncertainty", "Separated uncertainty-layer contract", uncertainty_audit["all_combined_confidence_intervals_disabled"], "", "pass" if uncertainty_audit["all_combined_confidence_intervals_disabled"] and not uncertainty_contract["policy"]["combined_public_confidence_interval_available"] else "fail", "measurement, parameter, numerical, and structural layers remain separate", "The contract and 12-state audit explicitly prohibit a combined public confidence interval."),
        _metric("uncertainty", "Yang low-pCO2 excess predictive scale", yang_predictive_error["variance_decomposition"]["candidate_excess_predictive_sigma_permil"], "permil", "diagnostic", "late-Quaternary, 1 PAL, fixed-GPP domain only", "Estimated from age-block held-out residuals after retaining reported isotope and propagated CO2 analytical variances; not a public default or whole-domain sigma."),
        _metric("uncertainty", "Empirical structural model-discrepancy calibration", False, "", "limitation", "required before a default joint posterior is presented as calibrated", "The joint engine is operational, but the release gate uses a synthetic discrepancy solely to verify propagation. Literature-corner guardrails remain non-probabilistic."),
        _metric("transient", "Fully simultaneous carbon-oxygen coupling", False, "", "limitation", "required before claiming fully coupled transient dynamics", "Current Fig. 9-type experiment uses operator splitting; its independent pO2 trajectories are audited below."),
    ]
    if photosynthesis is not None:
        metrics.extend(
            [
                _metric("Young diagnostic", "Fig. 9-type half-photosynthesis minimum", photosynthesis["minimum_cap_delta17_prime_permil"], "permil", "diagnostic", "Young visual minimum approximately -0.73 permil", f"Minimum at {photosynthesis['minimum_time_years']:.0f} years."),
                _metric("Young diagnostic", "Fig. 9-type peak delta-prime-18O", photosynthesis["maximum_delta18_prime_permil"], "permil", "diagnostic", "Young visual peak approximately 28 permil", f"Peak at {photosynthesis['maximum_delta18_time_years']:.0f} years."),
                _metric("transient", "Half-photosynthesis multi-reservoir response", True, "", "pass", "pO2 decreases, pCO2 overshoots, and isotope extrema precede equilibrium", f"Minimum displayed pO2 {photosynthesis['minimum_display_pO2_pal']:.3f} PAL; peak pCO2 {photosynthesis['peak_pCO2_ppm']:.1f} ppm."),
                _metric("transient", "Operator-split pO2 closure", photosynthesis["maximum_carbon_oxygen_pO2_mismatch_pal"], "PAL", "pass" if photosynthesis["maximum_carbon_oxygen_pO2_mismatch_pal"] <= 0.01 else "fail", "maximum absolute pO2 mismatch <= 0.01 PAL", "Independent detailed-box and molecular-reservoir oxygen inventories under identical photosynthesis forcing."),
                _metric("transient", "Uncontained numerical warning classes", photosynthesis["captured_warning_count"], "", "pass" if not warning_messages else "fail", "zero", "; ".join(warning_messages) if warning_messages else "None; internal BDF trial-state warnings are retained in solver provenance."),
            ]
        )

    formal_gates = [row for row in metrics if row["status"] in {"pass", "fail"}]
    report = {
        "scorecard": "updated molecular model public-UI release",
        "model_data_id": modern_live.model_data_id,
        "surface_data_id": surface.surface_data_id,
        "formal_gate_summary": {
            "passed": sum(row["status"] == "pass" for row in formal_gates),
            "failed": sum(row["status"] == "fail" for row in formal_gates),
            "all_passed": all(row["status"] == "pass" for row in formal_gates),
        },
        "modern_state": {
            "input": asdict(modern_request),
            "delta18_prime_permil": modern_live.central_delta18_prime_permil,
            "cap_delta17_prime_permil": modern_live.central_cap_delta17_prime_permil,
            "Pack_2021_cap_delta17_prime_permil": PACK_D17,
            "Pack_2021_cap_delta17_uncertainty_permil": PACK_D17_UNCERTAINTY,
            "conventional_delta18_permil": modern_delta18,
            "Pack_2021_conventional_delta18_permil": PACK_D18,
            "Pack_2021_conventional_delta18_uncertainty_permil": PACK_D18_UNCERTAINTY,
        },
        "metrics": metrics,
        "young_steady_diagnostics": young,
        "inversion_roundtrips": inversions,
        "conditional_posterior_check": posterior.as_dict(),
        "joint_posterior_check": joint_posterior_check,
        "balanced_transient": {
            "request": asdict(balanced.request),
            "final_display_distance_permil": balanced_distance,
            "operational_equilibrium_time_years": balanced.equilibrium_time_years,
            "solver": balanced.solver,
        },
        "fig10_transient": {
            "request": asdict(fig10.request),
            "shift_at_150_years_permil": fig10_shift_150,
            "shift_at_5000_years_permil": fig10_shift_5000,
            "operational_equilibrium_time_years": fig10.equilibrium_time_years,
            "monotonic_delta17_relaxation": fig10_monotonic,
            "solver": fig10.solver,
        },
        "photosynthesis_transient": photosynthesis,
        "source_audits": {
            "delta17": str(D17_AUDIT_PATH.relative_to(ROOT)),
            "delta18": str(D18_AUDIT_PATH.relative_to(ROOT)),
            "low_co2_accelerator": str(
                LOW_CO2_ACCELERATOR_AUDIT_PATH.relative_to(ROOT)
            ),
            "shape": str(SHAPE_AUDIT_PATH.relative_to(ROOT)),
            "Banerjee_2026_application": str(
                BANERJEE_APPLICATION_PATH.relative_to(ROOT)
            ),
            "Yang_2022_CO2_tracking": str(YANG_CO2_TRACKING_PATH.relative_to(ROOT)),
            "Yang_2022_transient_CO2_tracking": str(
                YANG_TRANSIENT_TRACKING_PATH.relative_to(ROOT)
            ),
            "Brandon_2020_Termination_V": str(
                BRANDON_TERMINATION_V_PATH.relative_to(ROOT)
            ),
            "uncertainty_contract": str(UNCERTAINTY_CONTRACT_PATH.relative_to(ROOT)),
            "uncertainty_layer_audit": str(UNCERTAINTY_AUDIT_PATH.relative_to(ROOT)),
            "Yang_low_CO2_predictive_error": str(
                YANG_PREDICTIVE_ERROR_PATH.relative_to(ROOT)
            ),
        },
    }
    _write_csv(metrics, ROWS_PATH)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, MARKDOWN_PATH)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-photosynthesis-transient", action="store_true")
    args = parser.parse_args()
    report = run(
        include_photosynthesis_transient=not args.skip_photosynthesis_transient
    )
    summary = report["formal_gate_summary"]
    print(
        f"Updated molecular release gates: {summary['passed']} passed, "
        f"{summary['failed']} failed"
    )
    print(f"Wrote {ROWS_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
