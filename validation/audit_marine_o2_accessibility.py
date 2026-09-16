"""Audit a source-backed marine-O2 accessibility term in the updated model.

The candidate changes no photochemistry and is not fitted to isotope output.
It applies the Bender/Liu atmosphere-accessible fraction only to the marine
part of total GPP at the three pO2 nodes explicitly represented in the Liu
archive.  Results are diagnostic until the promotion decision is documented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for subdirectory in ("code", "validation"):
    path = str(ROOT / subdirectory)
    if path not in sys.path:
        sys.path.insert(0, path)

from biological_o2_ensemble import central_biological_member  # noqa: E402
from isotope_reference_frames import (  # noqa: E402
    MODEL_LAMBDA,
    PrimeIsotopeComposition,
    relative_cap_delta17_ppm,
)
from liu_2021_exchange_limited_gpp import (  # noqa: E402
    exchange_limited_gpp_at_supported_po2,
)
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
)
from young_fig8_reconstruction import d17o_from_pco2, digitize_fig8  # noqa: E402
from young_validation_targets import (  # noqa: E402
    fig7_digitized_targets,
    fig8_pco2_points,
    young_fig7_294ppm_d17o,
)
from audit_yang_2022_co2_tracking import (  # noqa: E402
    INPUT_PATH as YANG_INPUT_PATH,
    YANG_RELATIVE_LAMBDA,
    _blocked_cross_validation,
    _load_rows as load_yang_rows,
    _moving_window_mean,
    _tracking_metrics,
)


DEFAULT_BENCHMARK = ROOT / "outputs" / "liu_2021_low_gpp_multimodel_benchmark.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "marine_o2_accessibility_audit.json"
DEFAULT_RELEASE_SCORECARD = (
    ROOT / "outputs" / "updated_molecular_release_scorecard.json"
)
PACK_D17 = -0.432
PACK_D17_SIGMA = 0.015
PACK_D18 = 23.9
PACK_D18_SIGMA = 0.3
UPDATED_MODERN_GPP = 290.0


def _candidate_state(po2: float, pco2: float, gpp: float):
    """Evaluate the marine-access candidate at a Liu-supported pO2 node."""

    exchange = exchange_limited_gpp_at_supported_po2(
        po2_pal=po2,
        gpp_percent_of_liu_modern=100.0 * gpp / UPDATED_MODERN_GPP,
    )
    return run_updated_central_state(
        UpdatedForwardInput(po2, pco2, gpp),
        marine_accessible_fraction=exchange.accessible_fraction,
        marine_accessibility_source=(
            "Bender et al. (1994) marine recycling architecture and "
            f"Liu et al. (2021) archived {po2:g}-PAL GPPOXY node"
        ),
    )


def _residual_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean_residual_permil": float(np.mean(array)),
        "mean_absolute_residual_permil": float(np.mean(np.abs(array))),
        "maximum_absolute_residual_permil": float(np.max(np.abs(array))),
    }


def _young_guardrail() -> dict[str, Any]:
    """Compare candidate curve shapes with digitized Young Fig. 7 and Fig. 8."""

    young_modern = float(young_fig7_294ppm_d17o(100.0))
    candidate_modern = _candidate_state(
        1.0, 294.0, UPDATED_MODERN_GPP
    ).cap_delta17_prime_permil
    reference_difference = candidate_modern - young_modern

    fig7_residuals: list[float] = []
    fig7_aligned: list[float] = []
    for pco2, gpp_percent, target in fig7_digitized_targets(
        samples_per_contour=15
    ):
        model = _candidate_state(
            1.0,
            float(pco2),
            UPDATED_MODERN_GPP * float(gpp_percent) / 100.0,
        ).cap_delta17_prime_permil
        residual = float(model - target)
        fig7_residuals.append(residual)
        fig7_aligned.append(residual - reference_difference)

    curves = digitize_fig8()
    fig8_rows: list[dict[str, float]] = []
    for gpp_percent in (100.0, 50.0):
        for pco2 in fig8_pco2_points(dense=True):
            target = float(d17o_from_pco2(pco2, int(gpp_percent), curves))
            model = _candidate_state(
                1.0,
                float(pco2),
                UPDATED_MODERN_GPP * gpp_percent / 100.0,
            ).cap_delta17_prime_permil
            fig8_rows.append(
                {
                    "gpp_percent": gpp_percent,
                    "residual": float(model - target),
                    "aligned_residual": float(model - target - reference_difference),
                }
            )

    return {
        "reference_difference_permil": float(reference_difference),
        "reference_alignment_is_diagnostic_only": True,
        "fig7": {
            "point_count": len(fig7_residuals),
            "absolute": _residual_summary(fig7_residuals),
            "reference_aligned": _residual_summary(fig7_aligned),
        },
        "fig8": {
            "point_count": len(fig8_rows),
            "absolute": _residual_summary(
                [row["residual"] for row in fig8_rows]
            ),
            "reference_aligned": _residual_summary(
                [row["aligned_residual"] for row in fig8_rows]
            ),
            "by_gpp": {
                str(int(gpp)): _residual_summary(
                    [
                        row["aligned_residual"]
                        for row in fig8_rows
                        if row["gpp_percent"] == gpp
                    ]
                )
                for gpp in (100.0, 50.0)
            },
        },
    }


def _candidate_relative_delta17_ppm(co2_values: np.ndarray) -> np.ndarray:
    reference_state = _candidate_state(1.0, 280.0, UPDATED_MODERN_GPP)
    reference = PrimeIsotopeComposition(
        delta18_prime_permil=reference_state.delta18_prime_permil,
        cap_delta17_prime_permil=reference_state.cap_delta17_prime_permil,
        lambda_ref=MODEL_LAMBDA,
    )
    predictions = []
    cache: dict[float, float] = {}
    for value in np.asarray(co2_values, dtype=float):
        key = float(value)
        if key not in cache:
            state = _candidate_state(1.0, key, UPDATED_MODERN_GPP)
            sample = PrimeIsotopeComposition(
                delta18_prime_permil=state.delta18_prime_permil,
                cap_delta17_prime_permil=state.cap_delta17_prime_permil,
                lambda_ref=MODEL_LAMBDA,
            )
            cache[key] = relative_cap_delta17_ppm(
                sample,
                reference,
                relative_lambda=YANG_RELATIVE_LAMBDA,
            )
        predictions.append(cache[key])
    return np.asarray(predictions, dtype=float)


def _yang_guardrail() -> dict[str, Any]:
    """Repeat the non-fitted Yang CO2-tracking test for the candidate."""

    rows = load_yang_rows(YANG_INPUT_PATH)
    age = np.asarray([row["gas_age_ka_bp"] for row in rows], dtype=float)
    co2 = np.asarray([row["co2_ppm"] for row in rows], dtype=float)
    observed = np.asarray(
        [row["cap_delta17_modern_air_relative_ppm"] for row in rows], dtype=float
    )
    predicted = _candidate_relative_delta17_ppm(co2)
    response_co2 = np.linspace(float(np.min(co2)), float(np.max(co2)), 101)
    response_delta17 = _candidate_relative_delta17_ppm(response_co2)
    slope, intercept = np.polyfit(response_co2, response_delta17, 1)
    nonlinear = response_delta17 - (slope * response_co2 + intercept)

    tracking: dict[str, Any] = {}
    for width in (0.0, 21.0):
        observed_smooth = _moving_window_mean(age, observed, width)
        predicted_smooth = _moving_window_mean(age, predicted, width)
        key = f"{width:g}_ka"
        tracking[key] = {
            **_tracking_metrics(observed_smooth, predicted_smooth),
            "blocked_cross_validation": _blocked_cross_validation(
                observed_smooth, predicted_smooth
            ),
        }
    return {
        "sample_count": len(rows),
        "amplitude_fitted_to_isotope_data": False,
        "additive_reference_offset_only": True,
        "co2_response": {
            "candidate_slope_ppm_Delta17_per_ppm_CO2": float(slope),
            "maximum_nonlinearity_about_linear_fit_ppm": float(
                np.max(np.abs(nonlinear))
            ),
        },
        "tracking_by_smoothing_window": tracking,
    }


def _metrics(rows: list[dict[str, Any]], difference_key: str) -> dict[str, float | int]:
    valid = [
        row
        for row in rows
        if row["candidate_status"] == "solved"
        and math.isfinite(float(row[difference_key]))
    ]
    differences = np.asarray([float(row[difference_key]) for row in valid])
    liu = np.asarray([float(row["Liu_response_permil"]) for row in valid])
    updated_key = (
        "candidate_response_permil"
        if difference_key == "candidate_minus_Liu_response_permil"
        else "baseline_response_permil"
    )
    updated = np.asarray([float(row[updated_key]) for row in valid])
    slope = float(np.dot(liu, updated) / np.dot(liu, liu))
    return {
        "paired_state_count": len(valid),
        "rmse_permil": float(np.sqrt(np.mean(differences**2))),
        "mean_bias_permil": float(np.mean(differences)),
        "maximum_absolute_difference_permil": float(np.max(np.abs(differences))),
        "origin_constrained_slope_relative_to_Liu": slope,
        "pearson_response_correlation": float(np.corrcoef(liu, updated)[0, 1]),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    valid = [row for row in rows if row["candidate_status"] == "solved"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    liu = np.asarray([row["Liu_response_permil"] for row in valid], dtype=float)
    baseline = np.asarray(
        [row["baseline_response_permil"] for row in valid], dtype=float
    )
    candidate = np.asarray(
        [row["candidate_response_permil"] for row in valid], dtype=float
    )
    extent = (float(min(liu.min(), baseline.min(), candidate.min())), 0.0)
    axes[0].plot(extent, extent, color="0.4", linewidth=1.0)
    axes[0].scatter(liu, baseline, s=16, alpha=0.55, label="Current")
    axes[0].scatter(liu, candidate, s=16, alpha=0.55, label="Marine access")
    axes[0].set_xlabel("Liu response (‰)")
    axes[0].set_ylabel("Updated-model response (‰)")
    axes[0].legend(frameon=False)

    po2_nodes = sorted({float(row["pO2_PAL"]) for row in valid})
    positions = np.arange(len(po2_nodes), dtype=float)
    width = 0.36
    baseline_rmse = []
    candidate_rmse = []
    for po2 in po2_nodes:
        subset = [row for row in valid if float(row["pO2_PAL"]) == po2]
        baseline_rmse.append(
            np.sqrt(
                np.mean(
                    np.square(
                        [row["baseline_minus_Liu_response_permil"] for row in subset]
                    )
                )
            )
        )
        candidate_rmse.append(
            np.sqrt(
                np.mean(
                    np.square(
                        [row["candidate_minus_Liu_response_permil"] for row in subset]
                    )
                )
            )
        )
    axes[1].bar(positions - width / 2, baseline_rmse, width, label="Current")
    axes[1].bar(positions + width / 2, candidate_rmse, width, label="Marine access")
    axes[1].set_xticks(positions, [f"{value:g}" for value in po2_nodes])
    axes[1].set_xlabel("pO$_2$ (PAL)")
    axes[1].set_ylabel("Response RMSE to Liu (‰)")
    axes[1].legend(frameon=False)

    colors = {0.1: "#007c91", 0.3: "#7b5ea7", 1.0: "#c4512d"}
    selected_gpp = 100.0
    for po2 in po2_nodes:
        subset = sorted(
            (
                row
                for row in valid
                if float(row["pO2_PAL"]) == po2
                and float(row["GPP_percent"]) == selected_gpp
            ),
            key=lambda row: float(row["pCO2_ppm"]),
        )
        axes[2].semilogx(
            [row["pCO2_ppm"] for row in subset],
            [row["candidate_minus_baseline_D17O_permil"] for row in subset],
            marker="o",
            color=colors[po2],
            label=f"{po2:g} PAL",
        )
    axes[2].axhline(0.0, color="0.4", linewidth=0.9)
    axes[2].set_xlabel("pCO$_2$ (ppm)")
    axes[2].set_ylabel("Marine-access minus current Δ′$^{17}$O (‰)")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(benchmark_path: Path, output_path: Path) -> dict[str, Any]:
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    source_rows = benchmark["scenarios"]
    member = central_biological_member()
    land_fraction = float(member.production_partition.land_fraction)

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        po2 = float(source["pO2_PAL"])
        pco2 = float(source["pCO2_ppm"])
        gpp_percent = float(source["GPP_percent_of_own_reference"])
        gpp = float(source["updated_global_GPP_PgC_per_year"])
        exchange = exchange_limited_gpp_at_supported_po2(
            po2_pal=po2,
            gpp_percent_of_liu_modern=gpp_percent,
        )
        accessible_fraction = float(exchange.accessible_fraction)
        effective_fraction = land_fraction + (1.0 - land_fraction) * accessible_fraction
        row: dict[str, Any] = {
            "pO2_PAL": po2,
            "pCO2_ppm": pco2,
            "GPP_percent": gpp_percent,
            "total_global_GPP_PgC_per_year": gpp,
            "marine_accessible_fraction": accessible_fraction,
            "effective_global_accessible_fraction": effective_fraction,
            "effective_global_accessible_GPP_PgC_per_year": gpp * effective_fraction,
            "Liu_response_permil": float(source["Liu_response_from_300ppm_100pct_permil"]),
            "baseline_D17O_permil": float(source["updated_cap_delta17_prime_permil"]),
            "candidate_status": "failed",
            "candidate_reason": "",
            "candidate_D17O_permil": float("nan"),
            "candidate_delta18_prime_permil": float("nan"),
        }
        try:
            candidate = run_updated_central_state(
                UpdatedForwardInput(po2, pco2, gpp),
                marine_accessible_fraction=accessible_fraction,
                marine_accessibility_source=(
                    "Bender et al. (1994) marine recycling architecture and "
                    f"Liu et al. (2021) archived {po2:g}-PAL GPPOXY node"
                ),
            )
            row.update(
                {
                    "candidate_status": "solved",
                    "candidate_D17O_permil": candidate.cap_delta17_prime_permil,
                    "candidate_delta18_prime_permil": candidate.delta18_prime_permil,
                }
            )
        except (RuntimeError, ValueError) as exc:
            row["candidate_reason"] = str(exc)
        rows.append(row)

    references: dict[float, dict[str, Any]] = {}
    for po2 in sorted({float(row["pO2_PAL"]) for row in rows}):
        matches = [
            row
            for row in rows
            if row["pO2_PAL"] == po2
            and row["pCO2_ppm"] == 300.0
            and row["GPP_percent"] == 100.0
        ]
        if len(matches) != 1 or matches[0]["candidate_status"] != "solved":
            raise RuntimeError(f"missing solved reference state at {po2:g} PAL")
        references[po2] = matches[0]

    for row in rows:
        reference = references[float(row["pO2_PAL"])]
        row["baseline_response_permil"] = (
            row["baseline_D17O_permil"] - reference["baseline_D17O_permil"]
        )
        row["baseline_minus_Liu_response_permil"] = (
            row["baseline_response_permil"] - row["Liu_response_permil"]
        )
        if row["candidate_status"] == "solved":
            row["candidate_response_permil"] = (
                row["candidate_D17O_permil"] - reference["candidate_D17O_permil"]
            )
            row["candidate_minus_Liu_response_permil"] = (
                row["candidate_response_permil"] - row["Liu_response_permil"]
            )
            row["candidate_minus_baseline_D17O_permil"] = (
                row["candidate_D17O_permil"] - row["baseline_D17O_permil"]
            )
        else:
            row["candidate_response_permil"] = float("nan")
            row["candidate_minus_Liu_response_permil"] = float("nan")
            row["candidate_minus_baseline_D17O_permil"] = float("nan")

    modern_exchange = exchange_limited_gpp_at_supported_po2(
        po2_pal=1.0,
        gpp_percent_of_liu_modern=100.0,
    )
    modern_candidate = run_updated_central_state(
        UpdatedForwardInput(1.0, 294.0, 290.0),
        marine_accessible_fraction=modern_exchange.accessible_fraction,
        marine_accessibility_source=(
            "Bender et al. (1994) marine recycling architecture and "
            "Liu et al. (2021) archived 1-PAL GPPOXY node"
        ),
    )
    conventional_d18 = 1000.0 * math.expm1(
        modern_candidate.delta18_prime_permil / 1000.0
    )
    baseline_metrics = _metrics(rows, "baseline_minus_Liu_response_permil")
    candidate_metrics = _metrics(rows, "candidate_minus_Liu_response_permil")
    young_guardrail = _young_guardrail()
    yang_guardrail = _yang_guardrail()
    release_scorecard = json.loads(
        DEFAULT_RELEASE_SCORECARD.read_text(encoding="utf-8")
    )
    baseline_young = release_scorecard["young_steady_diagnostics"]
    baseline_yang_path = ROOT / release_scorecard["source_audits"][
        "Yang_2022_CO2_tracking"
    ]
    baseline_yang = json.loads(baseline_yang_path.read_text(encoding="utf-8"))

    report = {
        "schema_version": 1,
        "audit": "marine O2 accessibility candidate",
        "candidate_is_central_model": False,
        "isotope_output_fitted": False,
        "physical_change": (
            "Apply Bender/Liu atmosphere-accessible recycling only to the marine "
            "share of total GPP; terrestrial photosynthetic O2 remains fully accessible."
        ),
        "pO2_policy": (
            "Evaluate only exact 0.1, 0.3, and 1 PAL Liu archive nodes; no pO2 "
            "interpolation or extrapolation."
        ),
        "central_partition": {
            "member": member.key,
            "land_fraction": land_fraction,
            "marine_fraction": 1.0 - land_fraction,
        },
        "modern_candidate": {
            "marine_accessible_fraction": modern_exchange.accessible_fraction,
            "Delta_prime_17O_permil": modern_candidate.cap_delta17_prime_permil,
            "residual_to_Pack_permil": modern_candidate.cap_delta17_prime_permil
            - PACK_D17,
            "Pack_one_sigma_overlap": abs(
                modern_candidate.cap_delta17_prime_permil - PACK_D17
            )
            <= PACK_D17_SIGMA,
            "delta_prime_18O_permil": modern_candidate.delta18_prime_permil,
            "conventional_delta_18O_permil": conventional_d18,
            "residual_to_Pack_delta18_permil": conventional_d18 - PACK_D18,
            "Pack_delta18_one_sigma_overlap": abs(conventional_d18 - PACK_D18)
            <= PACK_D18_SIGMA,
        },
        "shared_grid": {
            "scenario_count": len(rows),
            "solved_candidate_count": sum(
                row["candidate_status"] == "solved" for row in rows
            ),
            "failed_candidate_count": sum(
                row["candidate_status"] != "solved" for row in rows
            ),
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_minus_baseline_RMSE_change_permil": (
                float(candidate_metrics["rmse_permil"])
                - float(baseline_metrics["rmse_permil"])
            ),
        },
        "young_shape_guardrail": {
            "baseline": baseline_young,
            "candidate": young_guardrail,
            "candidate_minus_baseline_aligned_MAE_permil": {
                "fig7": (
                    young_guardrail["fig7"]["reference_aligned"]
                    ["mean_absolute_residual_permil"]
                    - baseline_young["fig7"]["reference_aligned"]
                    ["mean_absolute_residual_permil"]
                ),
                "fig8": (
                    young_guardrail["fig8"]["reference_aligned"]
                    ["mean_absolute_residual_permil"]
                    - baseline_young["fig8"]["reference_aligned"]
                    ["mean_absolute_residual_permil"]
                ),
            },
        },
        "yang_low_CO2_guardrail": {
            "baseline": {
                "co2_response": baseline_yang["co2_response"],
                "tracking_by_smoothing_window": {
                    key: baseline_yang["tracking_by_smoothing_window"][key]
                    for key in ("0_ka", "21_ka")
                },
            },
            "candidate": yang_guardrail,
        },
        "promotion_gate": {
            "status": "not_promoted_structural_sensitivity",
            "decision": (
                "Retain atmosphere-accessible marine production as a structural "
                "uncertainty experiment. It improves the modern Pack closure and "
                "the Liu shared-grid RMSE, but degrades the Young Fig. 7/Fig. 8 "
                "guardrails and the Yang low-CO2 tracking score."
            ),
            "requirements": [
                "preserve modern Pack Delta-prime-17O and delta18O gates",
                "preserve exact major-O2 closure",
                "improve or physically explain independent response behavior",
                "define a source-backed pO2 mapping over the full 0.1-2 PAL public domain",
                "regenerate and revalidate the publication output surface before use",
            ],
        },
        "rows": rows,
        "source_benchmark": str(Path(benchmark_path).resolve()),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    _write_csv(rows, csv_path)
    _plot(rows, figure_path)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.benchmark, args.output)


if __name__ == "__main__":
    main()
