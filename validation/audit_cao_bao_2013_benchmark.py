"""Audit the updated model against Cao and Bao's independent 2013 model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cao_bao_2013_reduced_model import (  # noqa: E402
    CaoBaoRateExperiment,
    PRESENT_FOP_MOL_O2_PER_YEAR,
    PRESENT_PO2_BAR,
    run_rate_experiment,
    steady_delta17_linear_052_permil,
)
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
)


SOURCE_PATH = ROOT / "model_data" / "literature" / "cao_bao_2013_benchmark_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "cao_bao_2013_multimodel_benchmark.json"
BASELINE_PCO2_PPM = 375.0
PCO2_AXIS_PPM = (375.0, 3200.0, 10000.0, 30000.0, 60000.0)
PO2_AXIS_PAL = (0.1, 0.3, 1.0)
CAO_FOP_EQUIVALENT_GPP_PGC_PER_YEAR = (
    PRESENT_FOP_MOL_O2_PER_YEAR * 12.011 / 1.0e15
)


def _source_reproduction(source: dict[str, object]) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    passed = True
    for anchor in source["fig_s6_visual_validation_anchors"]:
        result = run_rate_experiment(
            CaoBaoRateExperiment(
                o2_rise_mol_per_year=float(anchor["o2_rise_mol_per_year"]),
                co2_drawdown_mol_per_year=float(
                    anchor["co2_drawdown_mol_per_year"]
                ),
            )
        )
        minimum_residual = (
            result.minimum_delta17_linear_052_permil
            - float(anchor["minimum_permil"])
        )
        time_residual = result.minimum_time_years - float(
            anchor["minimum_time_years"]
        )
        row_passed = (
            abs(minimum_residual) <= float(anchor["minimum_uncertainty_permil"])
            and abs(time_residual) <= float(anchor["time_uncertainty_years"])
        )
        passed = passed and row_passed
        rows.append(
            {
                "panel": anchor["panel"],
                "o2_rise_mol_per_year": float(anchor["o2_rise_mol_per_year"]),
                "co2_drawdown_mol_per_year": float(
                    anchor["co2_drawdown_mol_per_year"]
                ),
                "figure_minimum_permil": float(anchor["minimum_permil"]),
                "calculated_minimum_permil": (
                    result.minimum_delta17_linear_052_permil
                ),
                "minimum_residual_permil": minimum_residual,
                "figure_minimum_time_years": float(anchor["minimum_time_years"]),
                "calculated_minimum_time_years": result.minimum_time_years,
                "time_residual_years": time_residual,
                "within_visual_uncertainty": row_passed,
            }
        )
    return rows, passed


def _common_domain_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for po2_pal in PO2_AXIS_PAL:
        for pco2_ppm in PCO2_AXIS_PPM:
            cao = steady_delta17_linear_052_permil(
                p_o2_bar=po2_pal * PRESENT_PO2_BAR,
                p_co2_bar=pco2_ppm * 1.0e-6,
            )
            updated_default = run_updated_central_state(
                UpdatedForwardInput(po2_pal, pco2_ppm, 290.0)
            )
            updated_exchange_matched = run_updated_central_state(
                UpdatedForwardInput(
                    po2_pal,
                    pco2_ppm,
                    CAO_FOP_EQUIVALENT_GPP_PGC_PER_YEAR,
                )
            )
            rows.append(
                {
                    "po2_pal": po2_pal,
                    "pco2_ppm": pco2_ppm,
                    "cao_delta17_linear_052_permil": cao,
                    "updated_delta17_prime_permil": (
                        updated_default.cap_delta17_prime_permil
                    ),
                    "updated_cao_fop_matched_delta17_prime_permil": (
                        updated_exchange_matched.cap_delta17_prime_permil
                    ),
                }
            )
    for po2_pal in PO2_AXIS_PAL:
        subset = [row for row in rows if row["po2_pal"] == po2_pal]
        baseline = subset[0]
        for row in subset:
            row["cao_response_from_375ppm_permil"] = (
                row["cao_delta17_linear_052_permil"]
                - baseline["cao_delta17_linear_052_permil"]
            )
            row["updated_response_from_375ppm_permil"] = (
                row["updated_delta17_prime_permil"]
                - baseline["updated_delta17_prime_permil"]
            )
            row["updated_cao_fop_matched_response_from_375ppm_permil"] = (
                row["updated_cao_fop_matched_delta17_prime_permil"]
                - baseline["updated_cao_fop_matched_delta17_prime_permil"]
            )
    return rows


def _shape_metrics(rows: list[dict[str, float]]) -> dict[str, object]:
    by_po2: dict[str, object] = {}
    for po2_pal in PO2_AXIS_PAL:
        subset = [row for row in rows if row["po2_pal"] == po2_pal]
        cao = np.asarray([row["cao_response_from_375ppm_permil"] for row in subset])
        updated = np.asarray(
            [row["updated_response_from_375ppm_permil"] for row in subset]
        )
        matched = np.asarray(
            [
                row["updated_cao_fop_matched_response_from_375ppm_permil"]
                for row in subset
            ]
        )
        core_slice = slice(0, 4)
        by_po2[f"{po2_pal:g}"] = {
            "cao_full_domain_monotonically_decreasing": bool(
                np.all(np.diff(cao) < 0.0)
            ),
            "updated_full_domain_monotonically_decreasing": bool(
                np.all(np.diff(updated) < 0.0)
            ),
            "cao_fop_matched_full_domain_monotonically_decreasing": bool(
                np.all(np.diff(matched) < 0.0)
            ),
            "all_monotonically_decreasing_through_30000ppm": bool(
                np.all(np.diff(cao[core_slice]) < 0.0)
                and np.all(np.diff(updated[core_slice]) < 0.0)
                and np.all(np.diff(matched[core_slice]) < 0.0)
            ),
            "updated_rank_correlation": float(spearmanr(cao, updated).statistic),
            "cao_fop_matched_rank_correlation": float(
                spearmanr(cao, matched).statistic
            ),
            "high_pco2_response_ratio_updated_over_cao": float(
                updated[-1] / cao[-1]
            ),
            "high_pco2_response_ratio_cao_fop_matched_over_cao": float(
                matched[-1] / cao[-1]
            ),
            "cao_30000_to_60000ppm_change_permil": float(cao[-1] - cao[-2]),
            "updated_30000_to_60000ppm_change_permil": float(
                updated[-1] - updated[-2]
            ),
            "cao_fop_matched_30000_to_60000ppm_change_permil": float(
                matched[-1] - matched[-2]
            ),
        }
    return {"by_po2": by_po2}


def _plot(output: Path, rows: list[dict[str, float]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    colors = {5.0e12: "#1b9e77", 1.0e13: "#3b5b92", 1.5e13: "#c44e52"}
    for rise in (5.0e12, 1.0e13, 1.5e13):
        result = run_rate_experiment(CaoBaoRateExperiment(o2_rise_mol_per_year=rise))
        axes[0].plot(
            result.time_years / 1.0e6,
            result.delta17_linear_052_permil,
            color=colors[rise],
            label=f"For = {rise / 1.0e12:g} x 10$^{{12}}$ mol yr$^{{-1}}$",
        )
    axes[0].set_title("Cao-Bao Fig. S6A reconstruction")
    axes[0].set_xlabel("Time (Myr)")
    axes[0].set_ylabel(r"Native $\Delta^{17}$O$_{O_2}$ (‰; $\lambda=0.52$)")
    axes[0].legend(frameon=False, fontsize=7.5)

    drawdown_colors = {1.7e13: "#3b5b92", 2.0e13: "#dd8452", 2.5e13: "#8172b3"}
    for drawdown in (1.7e13, 2.0e13, 2.5e13):
        result = run_rate_experiment(
            CaoBaoRateExperiment(co2_drawdown_mol_per_year=drawdown)
        )
        axes[1].plot(
            result.time_years / 1.0e6,
            result.delta17_linear_052_permil,
            color=drawdown_colors[drawdown],
            label=f"Fcd = {drawdown / 1.0e12:g} x 10$^{{12}}$ mol yr$^{{-1}}$",
        )
    axes[1].set_title("Cao-Bao Fig. S6B reconstruction")
    axes[1].set_xlabel("Time (Myr)")
    axes[1].legend(frameon=False, fontsize=7.5)

    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(PO2_AXIS_PAL)))
    for color, po2_pal in zip(palette, PO2_AXIS_PAL, strict=True):
        subset = [row for row in rows if row["po2_pal"] == po2_pal]
        pco2 = [row["pco2_ppm"] for row in subset]
        axes[2].plot(
            pco2,
            [row["cao_response_from_375ppm_permil"] for row in subset],
            "o-",
            color=color,
            label=f"{po2_pal:g} PAL O$_2$",
        )
        axes[2].plot(
            pco2,
            [
                row["updated_cao_fop_matched_response_from_375ppm_permil"]
                for row in subset
            ],
            "s--",
            color=color,
        )
    axes[2].set_xscale("log")
    axes[2].set_title("Common-domain CO$_2$ response")
    axes[2].set_xlabel("pCO$_2$ (ppm)")
    axes[2].set_ylabel("Change from 375 ppm (‰)")
    axes[2].legend(frameon=False, fontsize=7.5)
    axes[2].text(
        0.03,
        0.04,
        "circles: Cao-Bao native 0.52 coordinate\n"
        "squares: updated model at Cao Fop-equivalent turnover",
        transform=axes[2].transAxes,
        fontsize=7.2,
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run(output_path: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    reproduction, reproduction_passed = _source_reproduction(source)
    rows = _common_domain_rows()
    shape = _shape_metrics(rows)
    core_shape_passed = all(
        item["all_monotonically_decreasing_through_30000ppm"]
        for item in shape["by_po2"].values()
    )
    report = {
        "schema_version": 1,
        "benchmark_id": "cao_bao_2013_vs_updated_molecular_v1",
        "status": (
            "passed_source_and_core_shape_with_high_co2_difference"
            if reproduction_passed and core_shape_passed
            else "failed_benchmark"
        ),
        "comparison_role": (
            "Independent high-pCO2 dynamic and response-shape benchmark; no Cao-Bao "
            "output is used to calibrate the updated model."
        ),
        "source_reproduction": {
            "passed": reproduction_passed,
            "figure": "Supporting Information Figure S6",
            "rows": reproduction,
        },
        "comparison_policy": {
            "common_updated_model_domain_only": True,
            "baseline_pco2_ppm": BASELINE_PCO2_PPM,
            "absolute_coordinates_not_subtracted": True,
            "native_coordinate_mismatch": (
                "Cao-Bao uses linear delta17O - 0.52 delta18O; the updated model uses "
                "logarithmic Delta-prime-17O with lambda 0.528. Primary comparison is "
                "therefore baseline-relative response shape."
            ),
            "cao_fop_equivalent_gpp_pgC_per_year": (
                CAO_FOP_EQUIVALENT_GPP_PGC_PER_YEAR
            ),
        },
        "shape": shape,
        "source_inconsistency": source["source_inconsistency"],
        "interpretation": {
            "agreement": (
                "The reduced source model reproduces Fig. S6 without fitted factors. "
                "Both Cao-Bao and the updated model predict monotonically more negative "
                "atmospheric O2 anomaly with increasing pCO2 through 30,000 ppm across "
                "the tested pO2 range."
            ),
            "structural_difference": (
                "At 0.1 PAL O2, the Cao-Bao steady equation has a small rebound from "
                "30,000 to 60,000 ppm because both Phi(rho) and the residence-time term "
                "enter its closed-form solution. The updated model remains monotonic. "
                "Cao-Bao's moving-boundary Fig. S2 nevertheless orders higher initial "
                "pCO2 experiments toward larger transient depletion."
            ),
            "scope_limit": (
                "Cao-Bao starts at zero O2 and 0.1 bar CO2, outside the currently validated "
                "updated-model domain of at least 0.1 PAL O2 and at most 60,000 ppm CO2. "
                "A full moving-boundary transient comparison is therefore not extrapolated."
            ),
        },
        "common_domain_rows": rows,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(output_path.with_suffix(".png"), rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = run(arguments.output)
    ratios = [
        item["high_pco2_response_ratio_cao_fop_matched_over_cao"]
        for item in report["shape"]["by_po2"].values()
    ]
    print(
        f"{report['status']}: Fig. S6 source reproduction="
        f"{report['source_reproduction']['passed']}; high-CO2 response ratio "
        f"range={min(ratios):.3f}-{max(ratios):.3f}"
    )


if __name__ == "__main__":
    main()
