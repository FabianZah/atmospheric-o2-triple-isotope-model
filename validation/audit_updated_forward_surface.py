"""Audit shape, uncertainty, and inverse-root behavior of the updated engine."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from gpp_normalization import YOUNG_MODERN_GPP_PGC_PER_YEAR  # noqa: E402
from model_scenarios import ScenarioInput, run_scenario  # noqa: E402
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_forward,
)


PO2_VALUES = (0.1, 1.0, 2.0)
GPP_PERCENT_VALUES = (5.0, 25.0, 50.0, 100.0, 232.0)
PCO2_VALUES = np.geomspace(294.0, 60000.0, 25)


def _sign_change_count(values: np.ndarray, target: float) -> int:
    residual = values - target
    exact = int(np.count_nonzero(np.isclose(residual, 0.0, atol=1.0e-12)))
    changes = int(np.count_nonzero(residual[:-1] * residual[1:] < 0.0))
    return exact + changes


def _plot(rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.92, len(GPP_PERCENT_VALUES)))
    for axis, po2 in zip(axes, PO2_VALUES, strict=True):
        for color, gpp in zip(colors, GPP_PERCENT_VALUES, strict=True):
            selected = [
                row
                for row in rows
                if row["pO2_PAL"] == po2 and row["GPP_percent_of_Young"] == gpp
            ]
            x = np.asarray([row["pCO2_ppm"] for row in selected], dtype=float)
            y = np.asarray([row["central_D17O_permil"] for row in selected], dtype=float)
            lower = np.asarray([row["model_guardrail_lower_permil"] for row in selected])
            upper = np.asarray([row["model_guardrail_upper_permil"] for row in selected])
            axis.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0.0)
            axis.plot(x, y, color=color, linewidth=1.7, label=f"{gpp:g}%")
        axis.set_xscale("log")
        axis.set_title(f"{po2:g} PAL O$_2$")
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"Atmospheric O$_2$ $\Delta'^{17}$O (N{PER MILLE SIGN})")
    axes[-1].legend(title="GPP", frameon=False, fontsize=8)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(*, output_path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for po2 in PO2_VALUES:
        for gpp_percent in GPP_PERCENT_VALUES:
            gpp = YOUNG_MODERN_GPP_PGC_PER_YEAR * gpp_percent / 100.0
            for pco2 in PCO2_VALUES:
                result = run_updated_forward(
                    UpdatedForwardInput(
                        p_o2_pal=po2,
                        p_co2_ppm=float(pco2),
                        gpp_pgC_per_year=gpp,
                    )
                )
                rows.append(
                    {
                        "pO2_PAL": po2,
                        "pCO2_ppm": float(pco2),
                        "GPP_percent_of_Young": gpp_percent,
                        "GPP_PgC_per_year": gpp,
                        "central_D17O_permil": result.central_cap_delta17_prime_permil,
                        "model_guardrail_lower_permil": result.model_guardrail_interval_cap_delta17_permil[0],
                        "model_guardrail_upper_permil": result.model_guardrail_interval_cap_delta17_permil[1],
                        "maximum_fixed_point_residual_permil": result.maximum_fixed_point_residual_permil,
                    }
                )

    curve_checks: list[dict[str, object]] = []
    for po2 in PO2_VALUES:
        for gpp in GPP_PERCENT_VALUES:
            selected = [
                row
                for row in rows
                if row["pO2_PAL"] == po2 and row["GPP_percent_of_Young"] == gpp
            ]
            values = np.asarray([row["central_D17O_permil"] for row in selected])
            targets = np.linspace(values[-1], values[0], 31)[1:-1]
            root_counts = [_sign_change_count(values, float(target)) for target in targets]
            curve_checks.append(
                {
                    "pO2_PAL": po2,
                    "GPP_percent_of_Young": gpp,
                    "strictly_decreasing_with_pCO2": bool(np.all(np.diff(values) < 0.0)),
                    "maximum_tested_inverse_root_count": max(root_counts),
                    "minimum_tested_inverse_root_count": min(root_counts),
                }
            )

    gpp_monotonic = True
    po2_monotonic = True
    for pco2 in PCO2_VALUES:
        for po2 in PO2_VALUES:
            values = np.asarray(
                [
                    next(
                        row["central_D17O_permil"]
                        for row in rows
                        if row["pO2_PAL"] == po2
                        and row["GPP_percent_of_Young"] == gpp
                        and row["pCO2_ppm"] == float(pco2)
                    )
                    for gpp in GPP_PERCENT_VALUES
                ]
            )
            gpp_monotonic &= bool(np.all(np.diff(values) > 0.0))
        for gpp in GPP_PERCENT_VALUES:
            values = np.asarray(
                [
                    next(
                        row["central_D17O_permil"]
                        for row in rows
                        if row["pO2_PAL"] == po2
                        and row["GPP_percent_of_Young"] == gpp
                        and row["pCO2_ppm"] == float(pco2)
                    )
                    for po2 in PO2_VALUES
                ]
            )
            po2_monotonic &= bool(np.all(np.diff(values) > 0.0))

    comparison_inputs = (
        (1.0, 294.0, 100.0),
        (1.0, 3000.0, 50.0),
        (1.0, 30000.0, 50.0),
        (0.5, 10000.0, 25.0),
        (2.0, 60000.0, 232.0),
    )
    comparisons = []
    for po2, pco2, gpp in comparison_inputs:
        updated = run_updated_forward(
            UpdatedForwardInput(
                po2,
                pco2,
                YOUNG_MODERN_GPP_PGC_PER_YEAR * gpp / 100.0,
            )
        )
        legacy = run_scenario(
            ScenarioInput(
                preset="physical_extrapolation",
                p_o2_pal=po2,
                p_co2_ppm=pco2,
                gpp_scale=gpp / 100.0,
            )
        )
        legacy_value = float(legacy.outputs["O2_trop_D17O_permil"])
        comparisons.append(
            {
                "pO2_PAL": po2,
                "pCO2_ppm": pco2,
                "GPP_percent_of_Young": gpp,
                "updated_molecular_D17O_permil": updated.central_cap_delta17_prime_permil,
                "current_public_D17O_permil": legacy_value,
                "updated_minus_current_public_permil": updated.central_cap_delta17_prime_permil - legacy_value,
            }
        )

    gates = {
        "all_pCO2_curves_strictly_decreasing": all(
            row["strictly_decreasing_with_pCO2"] for row in curve_checks
        ),
        "all_tested_inverse_targets_have_one_root": all(
            row["minimum_tested_inverse_root_count"] == 1
            and row["maximum_tested_inverse_root_count"] == 1
            for row in curve_checks
        ),
        "D17O_strictly_increases_with_GPP": bool(gpp_monotonic),
        "D17O_strictly_increases_with_pO2": bool(po2_monotonic),
        "fixed_point_residual_below_1e-8_permil": max(
            float(row["maximum_fixed_point_residual_permil"]) for row in rows
        )
        <= 1.0e-8,
    }
    report = {
        "audit": "updated molecular forward surface",
        "status": "shape_and_inverse_gates_passed" if all(gates.values()) else "gate_failed",
        "candidate_promoted_to_public_default": False,
        "gates": gates,
        "grid": {
            "pO2_PAL": list(PO2_VALUES),
            "pCO2_ppm": PCO2_VALUES.tolist(),
            "GPP_percent_of_Young": list(GPP_PERCENT_VALUES),
            "case_count": len(rows),
        },
        "maximum_fixed_point_residual_permil": max(
            float(row["maximum_fixed_point_residual_permil"]) for row in rows
        ),
        "curve_checks": curve_checks,
        "comparison_with_current_public_engine": comparisons,
        "remaining_promotion_gates": [
            "propagate relative-modern GPP normalization uncertainty when that input convention is selected",
            "precompute a versioned updated output surface for responsive UI heatmaps",
            "verify UI heatmap and exports consume the same API and metadata",
            "rerun Young-anchor validation separately; do not require the updated model to equal Young",
        ],
    }

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output.with_suffix(".png"))
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "updated_molecular_forward_surface.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(output_path=args.output), indent=2))


if __name__ == "__main__":
    main()
