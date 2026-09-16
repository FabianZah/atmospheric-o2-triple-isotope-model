"""Audit constrained-posterior convergence for a low Delta-prime-17O case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for subdirectory in ("code", "validation"):
    module_path = str(ROOT / subdirectory)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from updated_constrained_pco2_posterior import (  # noqa: E402
    ConstrainedCoordinateInput,
    CoordinateConstraint,
    constrained_coordinate_posterior,
)
DEFAULT_OUTPUT = ROOT / "outputs" / "constrained_posterior_convergence.json"
DEFAULT_FIGURE = ROOT / "outputs" / "constrained_posterior_convergence.png"


def _display_density(axis: np.ndarray, density: np.ndarray) -> np.ndarray:
    transformed = density * np.log(10.0) * axis
    return transformed / float(np.max(transformed))


def _peak_count(values: np.ndarray, minimum_relative_height: float = 0.05) -> int:
    interior = (
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] >= minimum_relative_height)
    )
    return int(np.sum(interior))


def _run_case(
    *,
    name: str,
    pco2_grid_size: int,
    gpp_grid_size: int,
    po2_grid_size: int,
    uncertain_po2: bool,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    constraints = {
        "GPP": CoordinateConstraint("range", lower=145.0, upper=435.0),
        "pO2": (
            CoordinateConstraint("normal", center=1.0, sigma=0.2)
            if uncertain_po2
            else CoordinateConstraint("fixed", center=1.0)
        ),
    }
    started = perf_counter()
    result = constrained_coordinate_posterior(
        ConstrainedCoordinateInput(
            solve_for="pCO2",
            target_air_cap_delta17_permil=-10.0,
            measurement_sigma_permil=0.015,
            target_air_delta18_conventional_permil=23.9,
            delta18_measurement_sigma_permil=0.3,
            constraints=constraints,
            pco2_grid_size=pco2_grid_size,
            gpp_grid_size=gpp_grid_size,
            po2_grid_size=po2_grid_size,
            enforce_public_resolution=False,
        )
    )
    elapsed = perf_counter() - started
    axis = np.asarray(result.solve_axis, dtype=float)
    density = np.asarray(result.solve_marginal_density, dtype=float)
    displayed = _display_density(axis, density)
    second_difference = np.diff(displayed, n=2)
    record: dict[str, object] = {
        "name": name,
        "requested_grid": {
            "pCO2": pco2_grid_size,
            "GPP": gpp_grid_size,
            "pO2": po2_grid_size,
        },
        "uncertain_pO2": uncertain_po2,
        "elapsed_seconds": elapsed,
        "adaptive_pCO2_refinement": result.numerical_refinement_applied,
        "final_pCO2_bounds_ppm": result.final_solve_bounds,
        "posterior_median_ppm": result.posterior_median,
        "credible_interval_ppm": result.equal_tailed_credible_interval,
        "display_profile_peak_count_above_5pct": _peak_count(displayed),
        "display_profile_second_difference_L1": float(
            np.sum(np.abs(second_difference))
        ),
    }
    return record, axis, displayed


def run(*, output: Path = DEFAULT_OUTPUT, figure: Path = DEFAULT_FIGURE) -> dict:
    specifications = (
        ("fixed_pO2_public", 181, 81, 17, False),
        ("fixed_pO2_targeted", 181, 201, 17, False),
        ("fixed_pO2_dense_GPP", 181, 321, 17, False),
        ("fixed_pO2_dense_both", 401, 321, 17, False),
        ("uncertain_pO2_public", 181, 81, 17, True),
        ("uncertain_pO2_targeted_GPP", 181, 201, 17, True),
        ("uncertain_pO2_balanced", 181, 121, 33, True),
        ("uncertain_pO2_dense", 181, 161, 65, True),
    )
    rows: list[dict[str, object]] = []
    profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, pco2_size, gpp_size, po2_size, uncertain_po2 in specifications:
        row, axis, displayed = _run_case(
            name=name,
            pco2_grid_size=pco2_size,
            gpp_grid_size=gpp_size,
            po2_grid_size=po2_size,
            uncertain_po2=uncertain_po2,
        )
        rows.append(row)
        profiles.append((name, axis, displayed))

    report = {
        "audit": "constrained posterior convergence at Delta-prime-17O = -10 per mil",
        "inputs": {
            "air_Delta_prime_17O_permil": -10.0,
            "air_Delta_prime_17O_1sigma_permil": 0.015,
            "air_delta_18O_VSMOW_permil": 23.9,
            "air_delta_18O_1sigma_permil": 0.3,
            "GPP_range_PgC_per_year": [145.0, 435.0],
            "fixed_pO2_PAL": 1.0,
            "uncertain_pO2_PAL": {"mean": 1.0, "one_sigma": 0.2},
        },
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for panel, uncertain in zip(axes, (False, True), strict=True):
        for name, axis, displayed in profiles:
            row = next(item for item in rows if item["name"] == name)
            if bool(row["uncertain_pO2"]) != uncertain:
                continue
            panel.plot(axis, displayed, label=name)
        panel.set_xscale("log")
        panel.set_xlabel("pCO2 (ppm)")
        panel.set_ylabel("Relative density per log10(pCO2)")
        panel.set_title("Uncertain GPP, " + ("uncertain pO2" if uncertain else "fixed pO2"))
        panel.legend(fontsize=8)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    report = run(output=args.output, figure=args.figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
