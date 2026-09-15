"""Audit separated uncertainty layers across representative model states."""

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

from updated_output_surface import UpdatedOutputSurfaceInput  # noqa: E402
from updated_uncertainty_layers import decompose_updated_uncertainty  # noqa: E402


PCO2_NODES = (300.0, 1000.0, 3000.0, 10000.0, 30000.0, 60000.0)
GPP_PERCENT = (50.0, 100.0)
UPDATED_MODERN_GPP_PGC_PER_YEAR = 290.0


def _maximum_width(layer) -> float:
    return max(layer.lower_distance_permil, layer.upper_distance_permil)


def _plot(rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    colors = {50.0: "#28728f", 100.0: "#2a9d76"}
    for percent in GPP_PERCENT:
        selected = [row for row in rows if row["GPP_percent"] == percent]
        pco2 = np.asarray([float(row["pCO2_ppm"]) for row in selected])
        central = np.asarray([float(row["central_D17O_permil"]) for row in selected])
        additive = np.asarray(
            [float(row["additive_climate_D17O_permil"]) for row in selected]
        )
        fixed_total = np.asarray(
            [float(row["fixed_total_climate_D17O_permil"]) for row in selected]
        )
        axes[0].plot(
            pco2,
            central,
            "o-",
            color=colors[percent],
            label=f"{percent:g}% central",
        )
        axes[0].plot(
            pco2,
            additive,
            "s--",
            color=colors[percent],
            label=f"{percent:g}% Clima, additive CO$_2$",
        )
        axes[0].plot(
            pco2,
            fixed_total,
            "^:",
            color=colors[percent],
            label=f"{percent:g}% Clima, fixed total dry gas",
        )

        axis = axes[1 if percent == 50.0 else 2]
        for key, label, marker in (
            ("source_isoflux_width_permil", "Adnew source isoflux", "o"),
            ("biological_width_permil", "Biological corners", "s"),
            ("numerical_width_permil", "Numerical guardrail", "^"),
            ("structural_envelope_magnitude_permil", "Clima structural envelope", "D"),
        ):
            axis.plot(
                pco2,
                [float(row[key]) for row in selected],
                marker=marker,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_title(f"{percent:g}% modern GPP")
        axis.set_ylabel("Magnitude in atmospheric O$_2$ Δ′$^{17}$O (‰)")
        axis.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("Atmospheric O$_2$ Δ′$^{17}$O (‰)")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(output_path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for percent in GPP_PERCENT:
        absolute_gpp = UPDATED_MODERN_GPP_PGC_PER_YEAR * percent / 100.0
        for pco2 in PCO2_NODES:
            result = decompose_updated_uncertainty(
                UpdatedOutputSurfaceInput(1.0, pco2, absolute_gpp)
            )
            if len(result.structural_endmember_points) != 2:
                raise ValueError(
                    f"expected two exact Clima pressure end members at {pco2:g} ppm, "
                    f"{percent:g}% GPP"
                )
            structural = {
                point.pressure_convention: point
                for point in result.structural_endmember_points
            }
            if set(structural) != {"additive_co2", "fixed_total_dry_major"}:
                raise ValueError(
                    f"unexpected Clima pressure conventions at {pco2:g} ppm, "
                    f"{percent:g}% GPP: {sorted(structural)}"
                )
            additive = structural["additive_co2"]
            fixed_total = structural["fixed_total_dry_major"]
            source_width = _maximum_width(result.source_isoflux_parameter)
            biological_width = _maximum_width(result.biological_parameter)
            numerical_width = max(
                result.numerical.total_lower_margin_permil,
                result.numerical.total_upper_margin_permil,
            )
            structural_width = max(
                abs(additive.offset_from_central_permil),
                abs(fixed_total.offset_from_central_permil),
            )
            magnitudes = {
                "source_isoflux_parameter": source_width,
                "biological_parameter": biological_width,
                "numerical": numerical_width,
                "Clima_structural_envelope": structural_width,
            }
            rows.append(
                {
                    "pO2_PAL": 1.0,
                    "pCO2_ppm": pco2,
                    "GPP_percent": percent,
                    "GPP_PgC_per_year": absolute_gpp,
                    "central_D17O_permil": result.central_cap_delta17_prime_permil,
                    "source_isoflux_width_permil": source_width,
                    "biological_width_permil": biological_width,
                    "crossed_parameter_width_permil": _maximum_width(
                        result.crossed_parameter_envelope
                    ),
                    "numerical_width_permil": numerical_width,
                    "additive_climate_D17O_permil": additive.cap_delta17_prime_permil,
                    "fixed_total_climate_D17O_permil": (
                        fixed_total.cap_delta17_prime_permil
                    ),
                    "pressure_convention_difference_permil": (
                        fixed_total.cap_delta17_prime_permil
                        - additive.cap_delta17_prime_permil
                    ),
                    "structural_envelope_magnitude_permil": structural_width,
                    "largest_nonmeasurement_component": max(
                        magnitudes, key=magnitudes.get
                    ),
                    "combined_confidence_interval_available": (
                        result.combined_public_confidence_interval_available
                    ),
                }
            )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, figure_path)
    report: dict[str, object] = {
        "audit": "separated updated-model uncertainty layers",
        "status": "layers_separated_no_combined_confidence_interval",
        "measurement_layer": (
            "Not plotted: analytical uncertainty is observation-specific and no "
            "hypothetical value is assigned."
        ),
        "parameter_layer": (
            "Adnew one-sigma source-isoflux endpoints and biological literature "
            "corners are reported independently; neither is treated as a posterior."
        ),
        "numerical_layer": "deterministic response-surface and accelerator margins",
        "structural_layer": (
            "exact, non-interpolated 1 PAL Clima end-member values for additive-CO2 "
            "and fixed-total-dry-gas pressure conventions at the calculated nodes"
        ),
        "scenario_count": len(rows),
        "all_combined_confidence_intervals_disabled": all(
            not bool(row["combined_confidence_interval_available"]) for row in rows
        ),
        "rows": rows,
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "updated_uncertainty_layers_audit.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
