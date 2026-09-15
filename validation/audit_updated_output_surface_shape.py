"""Audit dense-domain shape and interval safety of the D17O accelerator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_output_surface import UpdatedMolecularOutputSurface  # noqa: E402


def run(*, surface_path: Path, report_path: Path) -> dict[str, object]:
    bundle = json.loads(surface_path.read_text(encoding="utf-8"))
    surface = UpdatedMolecularOutputSurface(bundle)
    po2 = np.geomspace(*surface.domain["po2_pal"], 41)
    pco2 = np.geomspace(*surface.domain["pco2_ppm"], 121)
    gpp = np.geomspace(*surface.domain["gpp_pgC_per_year"], 81)
    po2_grid, pco2_grid, gpp_grid = np.meshgrid(
        po2, pco2, gpp, indexing="ij"
    )
    points = np.column_stack(
        (po2_grid.ravel(), pco2_grid.ravel(), np.log(gpp_grid.ravel()))
    )
    central = surface._central_d17_interpolator(points).reshape(po2_grid.shape)
    node_central = surface.values["central_cap_delta17_prime_permil"]
    central_steps = {
        "pO2": np.diff(central, axis=0),
        "pCO2": np.diff(central, axis=1),
        "GPP": np.diff(central, axis=2),
    }

    widths: dict[str, dict[str, float]] = {}
    interval_order_valid = True
    interval_monotonicity_valid = True
    all_finite = bool(np.all(np.isfinite(central)))
    for lower_key, _upper_key in surface._interval_pairs():
        lower_interpolator, upper_interpolator = (
            surface._interval_width_interpolators[lower_key]
        )
        lower_width = np.exp(lower_interpolator(points))
        upper_width = np.exp(upper_interpolator(points))
        all_finite &= bool(
            np.all(np.isfinite(lower_width)) and np.all(np.isfinite(upper_width))
        )
        interval_order_valid &= bool(
            np.all(lower_width > 0.0) and np.all(upper_width > 0.0)
        )
        lower_surface = (central.ravel() - lower_width).reshape(central.shape)
        upper_surface = (central.ravel() + upper_width).reshape(central.shape)
        for bound_surface in (lower_surface, upper_surface):
            interval_monotonicity_valid &= bool(
                np.all(np.diff(bound_surface, axis=0) >= -1.0e-10)
                and np.all(np.diff(bound_surface, axis=1) <= 1.0e-10)
                and np.all(np.diff(bound_surface, axis=2) >= -1.0e-10)
            )
        widths[lower_key] = {
            "minimum_lower_width_permil": float(np.min(lower_width)),
            "maximum_lower_width_permil": float(np.max(lower_width)),
            "minimum_upper_width_permil": float(np.min(upper_width)),
            "maximum_upper_width_permil": float(np.max(upper_width)),
        }

    lower_overshoot = max(0.0, float(np.min(node_central) - np.min(central)))
    upper_overshoot = max(0.0, float(np.max(central) - np.max(node_central)))
    gates = {
        "all_dense_values_finite": all_finite,
        "all_uncertainty_widths_positive": interval_order_valid,
        "central_increases_with_pO2": bool(
            np.all(central_steps["pO2"] >= -1.0e-10)
        ),
        "central_decreases_with_pCO2": bool(
            np.all(central_steps["pCO2"] <= 1.0e-10)
        ),
        "central_increases_with_GPP": bool(
            np.all(central_steps["GPP"] >= -1.0e-10)
        ),
        "all_uncertainty_bounds_follow_same_monotonic_directions": (
            interval_monotonicity_valid
        ),
        "spline_overshoot_below_0p001_permil": max(
            lower_overshoot, upper_overshoot
        )
        <= 0.001,
    }
    report = {
        "audit": "updated molecular output-surface dense shape",
        "status": "accepted" if all(gates.values()) else "failed",
        "surface_data_id": surface.surface_data_id,
        "dense_point_count": int(central.size),
        "domain": surface.domain,
        "gates": gates,
        "central_cap_delta17_range_permil": [
            float(np.min(central)),
            float(np.max(central)),
        ],
        "training_node_cap_delta17_range_permil": [
            float(np.min(node_central)),
            float(np.max(node_central)),
        ],
        "spline_overshoot_permil": {
            "below_training_range": lower_overshoot,
            "above_training_range": upper_overshoot,
        },
        "maximum_wrong_direction_step_permil": {
            "pO2": float(max(0.0, -np.min(central_steps["pO2"]))),
            "pCO2": float(max(0.0, np.max(central_steps["pCO2"]))),
            "GPP": float(max(0.0, -np.min(central_steps["GPP"]))),
        },
        "uncertainty_width_ranges": widths,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "updated_molecular_output_surface_shape_audit.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(surface_path=args.surface, report_path=args.report), indent=2))


if __name__ == "__main__":
    main()
