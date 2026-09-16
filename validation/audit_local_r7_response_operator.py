"""Validate a fast isotope-ratio R7 operator against direct column fixed points."""

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

from audit_photochem_endmember_self_consistent_o2 import _target_evaluator  # noqa: E402
from global_o2_isotope_reservoir import frozen_photochemical_steady_state  # noqa: E402
from local_r7_response_operator import (  # noqa: E402
    LocalR7ResponseOperator,
    fit_local_r7_response_operator,
)
from modern_isotope_reference import modern_reference_isotope_compositions  # noqa: E402
from self_consistent_isotope_fixed_point import solve_mechanistic_fixed_point  # noqa: E402
from young_global_o2_budget import fixed_po2_young_biological_budget  # noqa: E402


PCO2_VALUES = (294.0, 3000.0, 30000.0)
GPP_VALUES = (5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 232.0)


def _load_scenarios(path: Path) -> list[dict[str, object]]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = report.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No scenarios found in {path}.")
    return rows


def _direct_lookup(paths: tuple[Path, ...]) -> dict[tuple[float, float, float], dict[str, object]]:
    lookup: dict[tuple[float, float, float], dict[str, object]] = {}
    for path in paths:
        for row in _load_scenarios(path):
            key = (
                float(row["pO2_PAL"]),
                float(row["pCO2_ppm"]),
                float(row["GPP_percent_of_Young"]),
            )
            if key in lookup:
                old = float(lookup[key]["self_consistent_cap_delta17_prime_permil"])
                new = float(row["self_consistent_cap_delta17_prime_permil"])
                if abs(old - new) > 1.0e-8:
                    raise ValueError(f"Conflicting direct fixed point: {key}")
            lookup[key] = row
    return lookup


def _fit_operator(forcing: dict[str, object]) -> LocalR7ResponseOperator:
    o2_reference, co2_composition = modern_reference_isotope_compositions()
    evaluate_target, records = _target_evaluator(
        artifact=Path(str(forcing["artifact"])),
        po2_pal=float(forcing["pO2_PAL"]),
        pco2_ppm=float(forcing["pCO2_ppm"]),
        gpp_percent=25.0,
        co2_composition=co2_composition,
    )

    def evaluate_tendency(state: np.ndarray) -> np.ndarray:
        evaluate_target(state)
        return records[-1]["photochemical"].values

    return fit_local_r7_response_operator(
        evaluate_tendency,
        np.asarray(
            [
                o2_reference.delta18_prime_permil,
                o2_reference.cap_delta17_prime_permil,
            ]
        ),
        finite_difference_step_permil=np.asarray([0.5, 0.5]),
        source=(
            f"native Photochem parent {float(forcing['pO2_PAL']):g} PAL, "
            f"{float(forcing['pCO2_ppm']):g} ppm"
        ),
    )


def _solve_fast(
    operator: LocalR7ResponseOperator,
    *,
    po2_pal: float,
    gpp_percent: float,
) -> tuple[np.ndarray, float, int]:
    o2_reference, _co2 = modern_reference_isotope_compositions()

    def evaluate_target(state: np.ndarray) -> np.ndarray:
        photochemical = operator.evaluate(state)
        biological = fixed_po2_young_biological_budget(
            po2_pal=po2_pal,
            gpp_percent=gpp_percent,
            photochemical=photochemical,
        )
        target = frozen_photochemical_steady_state(
            biological,
            photochemical,
            source="local R7 operator/global O2 fixed-point target",
        )
        return np.asarray(
            [target.delta18_prime_permil, target.cap_delta17_prime_permil]
        )

    result = solve_mechanistic_fixed_point(
        evaluate_target,
        np.asarray(
            [
                o2_reference.delta18_prime_permil,
                o2_reference.cap_delta17_prime_permil,
            ]
        ),
        finite_difference_step=np.asarray([0.5, 0.5]),
        tolerance=1.0e-8,
        maximum_iterations=8,
    )
    if not result.converged:
        raise RuntimeError(
            "local R7 response fixed point did not converge: "
            f"residual={result.maximum_absolute_residual:.6g}"
        )
    return result.state, result.maximum_absolute_residual, result.target_evaluations


def _plot(rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(PCO2_VALUES)))
    for pco2, color in zip(PCO2_VALUES, colors, strict=True):
        selected = sorted(
            (row for row in rows if float(row["pCO2_ppm"]) == pco2),
            key=lambda row: float(row["GPP_percent_of_Young"]),
        )
        axes[0].plot(
            [float(row["GPP_percent_of_Young"]) for row in selected],
            [float(row["direct_cap_delta17_prime_permil"]) for row in selected],
            "o",
            color=color,
            label=f"{pco2:g} ppm direct",
        )
        axes[0].plot(
            [float(row["GPP_percent_of_Young"]) for row in selected],
            [float(row["operator_cap_delta17_prime_permil"]) for row in selected],
            "x--",
            color=color,
            label=f"{pco2:g} ppm operator",
        )
    axes[0].set(
        xscale="log",
        xlabel="GPP (% of Young et al., 2014)",
        ylabel="O$_2$ $\\Delta'^{17}$O (\N{PER MILLE SIGN})",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0.0, color="0.45", linewidth=0.8)
    for pco2, color in zip(PCO2_VALUES, colors, strict=True):
        selected = sorted(
            (row for row in rows if float(row["pCO2_ppm"]) == pco2),
            key=lambda row: float(row["GPP_percent_of_Young"]),
        )
        axes[1].plot(
            [float(row["GPP_percent_of_Young"]) for row in selected],
            [float(row["cap_delta17_prime_residual_permil"]) for row in selected],
            "o-",
            color=color,
            label=f"{pco2:g} ppm",
        )
    axes[1].set(
        xscale="log",
        xlabel="GPP (% of Young et al., 2014)",
        ylabel="Operator minus direct (\N{PER MILLE SIGN})",
        title="Local R7 response residual",
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(
    forcing_path: Path,
    direct_paths: tuple[Path, ...],
    output_path: Path,
) -> dict[str, object]:
    forcing_rows = {
        (float(row["pO2_PAL"]), float(row["pCO2_ppm"])): row
        for row in _load_scenarios(forcing_path)
    }
    direct = _direct_lookup(direct_paths)
    operators: dict[float, LocalR7ResponseOperator] = {}
    rows: list[dict[str, object]] = []
    for pco2 in PCO2_VALUES:
        forcing = forcing_rows[(1.0, pco2)]
        operator = _fit_operator(forcing)
        operators[pco2] = operator
        for gpp in GPP_VALUES:
            direct_row = direct[(1.0, pco2, gpp)]
            state, fixed_point_residual, evaluations = _solve_fast(
                operator, po2_pal=1.0, gpp_percent=gpp
            )
            direct_delta18 = float(
                direct_row["self_consistent_delta18_prime_permil"]
            )
            direct_cap = float(
                direct_row["self_consistent_cap_delta17_prime_permil"]
            )
            rows.append(
                {
                    "pO2_PAL": 1.0,
                    "pCO2_ppm": pco2,
                    "GPP_percent_of_Young": gpp,
                    "direct_delta18_prime_permil": direct_delta18,
                    "operator_delta18_prime_permil": float(state[0]),
                    "delta18_prime_residual_permil": float(state[0] - direct_delta18),
                    "direct_cap_delta17_prime_permil": direct_cap,
                    "operator_cap_delta17_prime_permil": float(state[1]),
                    "cap_delta17_prime_residual_permil": float(state[1] - direct_cap),
                    "operator_fixed_point_residual_permil": fixed_point_residual,
                    "operator_target_evaluations": evaluations,
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
    cap_residuals = np.asarray(
        [float(row["cap_delta17_prime_residual_permil"]) for row in rows]
    )
    delta18_residuals = np.asarray(
        [float(row["delta18_prime_residual_permil"]) for row in rows]
    )
    report = {
        "audit": "local isotope-ratio R7 response operator",
        "status": "audit_complete",
        "policy": (
            "Five native isotope-column evaluations fit each operator in "
            "relative 17O/16O and 18O/16O coordinates. The global biological "
            "balance and fixed-point equations are unchanged. No damping, flux "
            "factor, output offset, or fit to the direct fixed-point targets is used."
        ),
        "operator_count": len(operators),
        "scenario_count": len(rows),
        "cap_delta17_prime_metrics_permil": {
            "mean_residual": float(np.mean(cap_residuals)),
            "mean_absolute_residual": float(np.mean(np.abs(cap_residuals))),
            "root_mean_square_residual": float(
                np.sqrt(np.mean(cap_residuals**2))
            ),
            "maximum_absolute_residual": float(np.max(np.abs(cap_residuals))),
        },
        "delta18_prime_metrics_permil": {
            "mean_residual": float(np.mean(delta18_residuals)),
            "mean_absolute_residual": float(np.mean(np.abs(delta18_residuals))),
            "root_mean_square_residual": float(
                np.sqrt(np.mean(delta18_residuals**2))
            ),
            "maximum_absolute_residual": float(np.max(np.abs(delta18_residuals))),
        },
        "operators": {
            f"1PAL_{pco2:g}ppm": operator.as_dict()
            for pco2, operator in operators.items()
        },
        "scenarios": rows,
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forcing",
        type=Path,
        default=ROOT / "outputs" / "photochem_gpp_holdout_isotope_kernel.json",
    )
    parser.add_argument(
        "--direct",
        nargs="+",
        type=Path,
        default=[
            ROOT / "outputs" / "photochem_self_consistent_o2_reference_grid_complete.json",
            ROOT / "outputs" / "photochem_gpp_holdout_self_consistent_o2.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "photochem_local_r7_response_operator.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.forcing, tuple(args.direct), args.output), indent=2))


if __name__ == "__main__":
    main()
