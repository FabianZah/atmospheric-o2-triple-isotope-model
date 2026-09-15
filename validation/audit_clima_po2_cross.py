"""Audit the pO2 dependence of the Clima high-pCO2 structural end member."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
PO2_LEVELS = (0.1, 1.0, 2.0)
PCO2_NODES = (300.0, 10000.0, 30000.0, 60000.0)


def _load(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plot(
    local_rows: list[dict[str, float]],
    global_rows: list[dict[str, float]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    colors = {0.1: "#28728f", 1.0: "#2a9d76", 2.0: "#d06b32"}
    for po2 in PO2_LEVELS:
        local = [row for row in local_rows if row["pO2_PAL"] == po2]
        pco2 = [row["pCO2_ppm"] for row in local]
        label = f"{po2:g} PAL O$_2$"
        axes[0, 0].plot(
            pco2,
            [row["climate_over_fixed_D17O_forcing"] for row in local],
            "o-",
            color=colors[po2],
            label=label,
        )
        axes[0, 1].plot(
            pco2,
            [row["climate_over_fixed_O3_column"] for row in local],
            "o-",
            color=colors[po2],
            label=label,
        )
        axes[1, 0].plot(
            pco2,
            [row["climate_over_fixed_O3_to_O1D_source"] for row in local],
            "o-",
            color=colors[po2],
            label=label,
        )
        for percent, marker, linestyle in ((50.0, "o", "-"), (100.0, "s", "--")):
            selected = [
                row
                for row in global_rows
                if row["pO2_PAL"] == po2 and row["GPP_percent"] == percent
            ]
            axes[1, 1].plot(
                [row["pCO2_ppm"] for row in selected],
                [row["climate_minus_fixed_D17O_permil"] for row in selected],
                marker=marker,
                linestyle=linestyle,
                color=colors[po2],
                label=f"{po2:g} PAL, {percent:g}% GPP",
            )

    axes[0, 0].axhline(1.0, color="0.45", linewidth=0.9)
    axes[0, 1].axhline(1.0, color="0.45", linewidth=0.9)
    axes[1, 0].axhline(1.0, color="0.45", linewidth=0.9)
    axes[1, 1].axhline(0.0, color="0.45", linewidth=0.9)
    axes[0, 0].set_ylabel("Clima / fixed R7 Δ′$^{17}$O forcing")
    axes[0, 1].set_ylabel("Clima / fixed O$_3$ column")
    axes[1, 0].set_ylabel("Clima / fixed O$_3$ to O($^1$D) source")
    axes[1, 1].set_ylabel("Clima − fixed atmospheric O$_2$ Δ′$^{17}$O (‰)")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(
    attribution_paths: tuple[Path, Path, Path],
    global_paths: tuple[Path, Path, Path],
    output_path: Path,
) -> dict[str, object]:
    attributions = [_load(path) for path in attribution_paths]
    globals_ = [_load(path) for path in global_paths]
    observed_po2 = tuple(float(report["pO2_PAL"]) for report in attributions)
    if observed_po2 != PO2_LEVELS:
        raise ValueError(f"expected pO2 reports {PO2_LEVELS}, found {observed_po2}")
    if tuple(float(report["pO2_PAL"]) for report in globals_) != PO2_LEVELS:
        raise ValueError("global and local pO2 reports are not aligned")
    if not all(report["all_scenarios_converged"] for report in globals_):
        raise ValueError("at least one global pO2-cross scenario did not converge")

    local_rows: list[dict[str, float]] = []
    for po2, report in zip(PO2_LEVELS, attributions):
        rows = {float(row["pCO2_ppm"]): row for row in report["rows"]}
        if not set(PCO2_NODES).issubset(rows):
            raise ValueError(f"missing pCO2 nodes in {po2:g} PAL attribution")
        for pco2 in PCO2_NODES:
            row = rows[pco2]
            local_rows.append(
                {
                    "pO2_PAL": po2,
                    "pCO2_ppm": pco2,
                    "climate_over_fixed_D17O_forcing": float(
                        row["climate_over_fixed_D17O_forcing"]
                    ),
                    "climate_over_fixed_O3_column": float(
                        row["climate_over_fixed_O3_column_molecules_per_cm2"]
                    ),
                    "climate_over_fixed_O3_to_O1D_source": float(
                        row[
                            "climate_over_fixed_O3_to_O1D_column_photolysis_molecules_per_cm2_s"
                        ]
                    ),
                    "maximum_temperature_anomaly_K": float(
                        row["maximum_temperature_anomaly_K"]
                    ),
                }
            )

    global_rows: list[dict[str, float]] = []
    for po2, report in zip(PO2_LEVELS, globals_):
        for row in report["rows"]:
            pco2 = float(row["pCO2_ppm"])
            percent = float(row["GPP_percent_of_updated_modern"])
            if pco2 not in PCO2_NODES:
                continue
            global_rows.append(
                {
                    "pO2_PAL": po2,
                    "pCO2_ppm": pco2,
                    "GPP_percent": percent,
                    "central_D17O_permil": float(
                        row["fixed_cap_delta17_prime_permil"]
                    ),
                    "climate_D17O_permil": float(
                        row["climate_cap_delta17_prime_permil"]
                    ),
                    "climate_minus_fixed_D17O_permil": float(
                        row["climate_minus_fixed_D17O_permil"]
                    ),
                    "climate_minus_fixed_delta18_permil": float(
                        row["climate_minus_fixed_delta18_permil"]
                    ),
                }
            )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    local_csv = output.with_name(f"{output.stem}_local.csv")
    global_csv = output.with_name(f"{output.stem}_global.csv")
    figure_path = output.with_suffix(".png")
    for rows, path in ((local_rows, local_csv), (global_rows, global_csv)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _plot(local_rows, global_rows, figure_path)

    high_local = [row for row in local_rows if row["pCO2_ppm"] >= 30000.0]
    high_global = [row for row in global_rows if row["pCO2_ppm"] >= 30000.0]
    by_po2 = {
        f"{po2:g}": {
            "minimum_high_pCO2_R7_forcing_ratio": min(
                row["climate_over_fixed_D17O_forcing"]
                for row in high_local
                if row["pO2_PAL"] == po2
            ),
            "maximum_absolute_high_pCO2_global_D17O_shift_permil": max(
                abs(row["climate_minus_fixed_D17O_permil"])
                for row in high_global
                if row["pO2_PAL"] == po2
            ),
        }
        for po2 in PO2_LEVELS
    }
    report: dict[str, object] = {
        "audit": "Clima pO2 by high-pCO2 structural interaction",
        "status": "po2_interaction_material_structural_endmembers_not_promoted",
        "central_model_changed": False,
        "pO2_PAL": list(PO2_LEVELS),
        "pCO2_ppm": list(PCO2_NODES),
        "all_global_scenarios_converged": True,
        "summary_by_pO2": by_po2,
        "maximum_absolute_global_D17O_shift_permil": max(
            abs(row["climate_minus_fixed_D17O_permil"]) for row in global_rows
        ),
        "interpretation": {
            "supported": (
                "pO2 materially modulates the high-pCO2 climate-ozone-R7 response; "
                "low pO2 strengthens the forcing reduction and high pO2 partly buffers it."
            ),
            "remaining_limit": (
                "All Clima cases retain an isothermal stratosphere without explicit "
                "ozone-heated stratospheric radiative equilibrium."
            ),
            "promotion_decision": False,
            "next_test": "ozone-heated or independently bracketed stratospheric profiles",
        },
        "local_rows": local_rows,
        "global_rows": global_rows,
        "sources": {
            "attribution_reports": [str(Path(path).resolve()) for path in attribution_paths],
            "global_reports": [str(Path(path).resolve()) for path in global_paths],
        },
        "local_csv": str(local_csv),
        "global_csv": str(global_csv),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-0p1", type=Path, required=True)
    parser.add_argument(
        "--attribution-1",
        type=Path,
        default=ROOT / "outputs" / "clima_high_pco2_attribution.json",
    )
    parser.add_argument("--attribution-2", type=Path, required=True)
    parser.add_argument("--global-0p1", type=Path, required=True)
    parser.add_argument(
        "--global-1",
        type=Path,
        default=ROOT / "outputs" / "clima_global_o2_response.json",
    )
    parser.add_argument("--global-2", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "clima_po2_cross_audit.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                (args.attribution_0p1, args.attribution_1, args.attribution_2),
                (args.global_0p1, args.global_1, args.global_2),
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
