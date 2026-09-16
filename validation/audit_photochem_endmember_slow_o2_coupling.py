"""Couple sparse Photochem isotope end members to the slow global O2 box."""

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
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from global_o2_isotope_reservoir import (  # noqa: E402
    IsotopologueTendency,
    YoungBiologicalO2Budget,
    frozen_photochemical_steady_state,
)
from young_global_o2_budget import (  # noqa: E402
    GLOBAL_MAJOR_O2_MOLES_1PAL,
    fixed_po2_young_biological_budget,
)
GPP_PERCENT_VALUES = (5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 232.0)


def _photochemical_tendency(row: dict[str, object]) -> IsotopologueTendency:
    return IsotopologueTendency(
        float(row["R7_O16O16_tendency_mol_per_year"]),
        float(row["R7_O16O17_tendency_mol_per_year"]),
        float(row["R7_O16O18_tendency_mol_per_year"]),
        source=(
            f"native Photochem isotope kernel at {row['pO2_PAL']:g} PAL and "
            f"{row['pCO2_ppm']:g} ppm"
        ),
    )


def _fixed_po2_budget(
    *,
    po2_pal: float,
    gpp_percent: float,
    photochemical: IsotopologueTendency,
) -> YoungBiologicalO2Budget:
    return fixed_po2_young_biological_budget(
        po2_pal=po2_pal,
        gpp_percent=gpp_percent,
        photochemical=photochemical,
    )


def _rows(forcing_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for forcing in forcing_rows:
        po2 = float(forcing["pO2_PAL"])
        pco2 = float(forcing["pCO2_ppm"])
        photo = _photochemical_tendency(forcing)
        target_major = GLOBAL_MAJOR_O2_MOLES_1PAL * po2
        for gpp_percent in GPP_PERCENT_VALUES:
            try:
                budget = _fixed_po2_budget(
                    po2_pal=po2,
                    gpp_percent=gpp_percent,
                    photochemical=photo,
                )
                equilibrium = frozen_photochemical_steady_state(
                    budget,
                    photo,
                    source="sparse native-parent slow global O2 equilibrium",
                )
                cap = equilibrium.cap_delta17_prime_permil
                delta18 = equilibrium.delta18_prime_permil
                major_relative_residual = equilibrium.o16o16 / target_major - 1.0
                physical = True
                reason = ""
                respiration_rate = budget.respiration_rate_per_year
            except ValueError as error:
                cap = float("nan")
                delta18 = float("nan")
                major_relative_residual = float("nan")
                physical = False
                reason = str(error)
                respiration_rate = float("nan")
            results.append(
                {
                    "pO2_PAL": po2,
                    "pCO2_ppm": pco2,
                    "GPP_percent_of_Young": gpp_percent,
                    "physical_affine_domain": physical,
                    "domain_failure_reason": reason,
                    "equilibrium_cap_delta17_prime_permil": cap,
                    "equilibrium_delta18_prime_permil": delta18,
                    "diagnosed_respiration_rate_per_year": respiration_rate,
                    "major_O2_relative_steady_residual": major_relative_residual,
                }
            )
    return results


def _shape_checks(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if bool(row["physical_affine_domain"])]
    co2_checks = []
    for po2 in sorted({float(row["pO2_PAL"]) for row in valid}):
        for gpp in GPP_PERCENT_VALUES:
            selected = sorted(
                (
                    row
                    for row in valid
                    if float(row["pO2_PAL"]) == po2
                    and float(row["GPP_percent_of_Young"]) == gpp
                ),
                key=lambda row: float(row["pCO2_ppm"]),
            )
            values = np.asarray(
                [float(row["equilibrium_cap_delta17_prime_permil"]) for row in selected]
            )
            co2_checks.append(
                {
                    "pO2_PAL": po2,
                    "GPP_percent_of_Young": gpp,
                    "strictly_more_negative_with_pCO2": bool(
                        len(values) >= 2 and np.all(np.diff(values) < 0.0)
                    ),
                }
            )
    gpp_checks = []
    for po2 in sorted({float(row["pO2_PAL"]) for row in valid}):
        for pco2 in sorted({float(row["pCO2_ppm"]) for row in valid if float(row["pO2_PAL"]) == po2}):
            selected = sorted(
                (
                    row
                    for row in valid
                    if float(row["pO2_PAL"]) == po2
                    and float(row["pCO2_ppm"]) == pco2
                ),
                key=lambda row: float(row["GPP_percent_of_Young"]),
            )
            values = np.asarray(
                [float(row["equilibrium_cap_delta17_prime_permil"]) for row in selected]
            )
            gpp_checks.append(
                {
                    "pO2_PAL": po2,
                    "pCO2_ppm": pco2,
                    "strictly_less_negative_with_GPP": bool(
                        len(values) >= 2 and np.all(np.diff(values) > 0.0)
                    ),
                }
            )
    return {
        "pCO2_direction_checks": co2_checks,
        "GPP_direction_checks": gpp_checks,
        "all_pCO2_directions_pass": all(
            row["strictly_more_negative_with_pCO2"] for row in co2_checks
        ),
        "all_GPP_directions_pass": all(
            row["strictly_less_negative_with_GPP"] for row in gpp_checks
        ),
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], path: Path) -> None:
    po2_values = sorted({float(row["pO2_PAL"]) for row in rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.06, 0.92, len(GPP_PERCENT_VALUES)))
    figure, axes = plt.subplots(
        1, len(po2_values), figsize=(13.2, 4.7), sharey=False, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, po2 in zip(axes, po2_values):
        for gpp, color in zip(GPP_PERCENT_VALUES, colors):
            selected = sorted(
                (
                    row
                    for row in rows
                    if bool(row["physical_affine_domain"])
                    and float(row["pO2_PAL"]) == po2
                    and float(row["GPP_percent_of_Young"]) == gpp
                ),
                key=lambda row: float(row["pCO2_ppm"]),
            )
            axis.plot(
                [float(row["pCO2_ppm"]) for row in selected],
                [float(row["equilibrium_cap_delta17_prime_permil"]) for row in selected],
                "o-",
                color=color,
                label=f"{gpp:g}%" if axis is axes[0] else None,
            )
        axis.set_xscale("log")
        axis.set_title(f"{po2:g} PAL O$_2$")
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel(r"Raw equilibrium O$_2$ $\Delta'^{17}$O (‰)")
    axes[0].legend(title="GPP", frameon=False, fontsize=8, title_fontsize=9)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(input_path: Path, output_path: Path) -> dict[str, object]:
    forcing_report = json.loads(Path(input_path).read_text(encoding="utf-8"))
    rows = _rows(forcing_report["scenarios"])
    checks = _shape_checks(rows)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    _write_csv(rows, csv_path)
    _plot(rows, figure_path)
    maximum_major_residual = max(
        abs(float(row["major_O2_relative_steady_residual"]))
        for row in rows
        if bool(row["physical_affine_domain"])
    )
    report = {
        "audit": "sparse native-parent forcing coupled to slow global O2",
        "status": (
            "gate_passed"
            if checks["all_pCO2_directions_pass"]
            and checks["all_GPP_directions_pass"]
            and maximum_major_residual <= 1.0e-12
            else "gate_failed"
        ),
        "forcing_source": str(Path(input_path).resolve()),
        "biological_policy": (
            "Young et al. (2014) isotope fractionation laws are retained for "
            "this compatibility gate. GPP scales gross production; the major-"
            "O2 respiration coefficient is diagnosed separately in each case "
            "to preserve prescribed pO2."
        ),
        "equilibrium_policy": (
            "One-way affine equilibrium using R7 forcing evaluated at the Pack "
            "modern isotope boundary. These values test shape and domain only; "
            "extreme states require a self-consistent fast-column/slow-box iteration."
        ),
        "gpp_percent_values": list(GPP_PERCENT_VALUES),
        "scenario_count": len(rows),
        "physical_scenario_count": sum(
            bool(row["physical_affine_domain"]) for row in rows
        ),
        "maximum_major_O2_relative_steady_residual": maximum_major_residual,
        "shape_checks": checks,
        "scenarios": rows,
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "photochem_endmember_isotope_kernel.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "photochem_endmember_slow_o2_coupling.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
