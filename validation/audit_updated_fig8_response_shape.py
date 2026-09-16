"""Diagnose the updated molecular model's Young Fig. 8 response shape.

The required-GPP calculation is diagnostic only. It asks which GPP value in
the unchanged updated model would reproduce each digitized Young point; it
does not alter the model or define a correction function.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_output_surface import (  # noqa: E402
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)
from updated_output_surface_inverse import (  # noqa: E402
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)
from young_fig8_reconstruction import d17o_from_pco2, digitize_fig8  # noqa: E402
from young_validation_targets import fig8_pco2_points  # noqa: E402


OUTPUTS = ROOT / "outputs"
CSV_PATH = OUTPUTS / "updated_molecular_fig8_response_shape.csv"
JSON_PATH = OUTPUTS / "updated_molecular_fig8_response_shape.json"
FIGURE_PATH = OUTPUTS / "updated_molecular_fig8_response_shape"
UPDATED_MODERN_GPP = 290.0


def _write_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, object]:
    surface = load_updated_output_surface()
    curves = digitize_fig8()
    rows: list[dict[str, float]] = []

    for gpp_percent in (100.0, 50.0):
        nominal_gpp = UPDATED_MODERN_GPP * gpp_percent / 100.0
        group: list[dict[str, float]] = []
        for pco2 in fig8_pco2_points(dense=True):
            target = float(d17o_from_pco2(pco2, int(gpp_percent), curves))
            model = surface.evaluate(
                UpdatedOutputSurfaceInput(1.0, pco2, nominal_gpp)
            ).central_cap_delta17_prime_permil
            inverse = invert_updated_output_surface(
                UpdatedSurfaceInverseInput(
                    target_air_cap_delta17_permil=target,
                    solve_for="GPP",
                    p_o2_pal=1.0,
                    p_co2_ppm=pco2,
                    gpp_pgC_per_year=nominal_gpp,
                ),
                verify_live_root=False,
            )
            if inverse.central_root is None:
                raise RuntimeError(
                    f"Young Fig. 8 target has no GPP root at {pco2:g} ppm, "
                    f"{gpp_percent:g}% nominal GPP"
                )
            group.append(
                {
                    "GPP_percent_nominal": gpp_percent,
                    "pCO2_ppm": float(pco2),
                    "Young_target_permil": target,
                    "updated_model_permil": float(model),
                    "residual_permil": float(model - target),
                    "required_GPP_PgC_per_year": float(inverse.central_root),
                    "required_GPP_percent_updated_modern": float(
                        100.0 * inverse.central_root / UPDATED_MODERN_GPP
                    ),
                    "required_to_nominal_GPP_ratio": float(
                        inverse.central_root / nominal_gpp
                    ),
                }
            )

        origin_residual = group[0]["residual_permil"]
        log_co2 = np.log([row["pCO2_ppm"] for row in group])
        target_values = np.asarray([row["Young_target_permil"] for row in group])
        model_values = np.asarray([row["updated_model_permil"] for row in group])
        target_slopes = np.gradient(target_values, log_co2)
        model_slopes = np.gradient(model_values, log_co2)
        for row, target_slope, model_slope in zip(
            group, target_slopes, model_slopes, strict=True
        ):
            row["curve_origin_aligned_residual_permil"] = float(
                row["residual_permil"] - origin_residual
            )
            row["Young_slope_per_ln_pCO2_permil"] = float(target_slope)
            row["updated_slope_per_ln_pCO2_permil"] = float(model_slope)
            row["slope_difference_per_ln_pCO2_permil"] = float(
                model_slope - target_slope
            )
        rows.extend(group)

    summary: dict[str, object] = {
        "audit": "updated molecular response against digitized Young Fig. 8",
        "surface_data_id": surface.surface_data_id,
        "policy": (
            "diagnostic only; required GPP is not used to tune or correct the model"
        ),
        "updated_modern_GPP_PgC_per_year": UPDATED_MODERN_GPP,
        "by_nominal_GPP_percent": {},
    }
    for gpp_percent in (100.0, 50.0):
        group = [row for row in rows if row["GPP_percent_nominal"] == gpp_percent]
        aligned = np.asarray(
            [row["curve_origin_aligned_residual_permil"] for row in group]
        )
        ratios = np.asarray([row["required_to_nominal_GPP_ratio"] for row in group])
        summary["by_nominal_GPP_percent"][str(int(gpp_percent))] = {
            "point_count": len(group),
            "curve_origin_aligned_mean_absolute_residual_permil": float(
                np.mean(np.abs(aligned))
            ),
            "curve_origin_aligned_maximum_absolute_residual_permil": float(
                np.max(np.abs(aligned))
            ),
            "required_to_nominal_GPP_ratio_range": [
                float(np.min(ratios)),
                float(np.max(ratios)),
            ],
            "required_to_nominal_GPP_ratio_at_30000_ppm": float(ratios[-1]),
            "residual_at_30000_ppm_permil": float(group[-1]["residual_permil"]),
        }

    _write_csv(rows, CSV_PATH)
    JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    colors = {100.0: "#2368a2", 50.0: "#c44e52"}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    for gpp_percent in (100.0, 50.0):
        group = [row for row in rows if row["GPP_percent_nominal"] == gpp_percent]
        x = np.asarray([row["pCO2_ppm"] for row in group])
        young = np.asarray([row["Young_target_permil"] for row in group])
        model = np.asarray([row["updated_model_permil"] for row in group])
        aligned = np.asarray(
            [row["curve_origin_aligned_residual_permil"] for row in group]
        )
        required = np.asarray(
            [row["required_GPP_percent_updated_modern"] for row in group]
        )
        color = colors[gpp_percent]
        axes[0].plot(x, young, color=color, lw=2.2, label=f"Young {gpp_percent:g}%")
        axes[0].plot(x, model, color=color, lw=1.7, ls="--", label=f"Updated {gpp_percent:g}%")
        axes[1].plot(x, aligned, color=color, lw=2.0, marker="o", ms=3.2, label=f"{gpp_percent:g}%")
        axes[2].plot(x, required, color=color, lw=2.0, marker="o", ms=3.2, label=f"{gpp_percent:g}%")
        axes[2].axhline(gpp_percent, color=color, lw=0.9, ls=":")

    axes[0].set_ylabel(r"Atmospheric O$_2$ $\Delta'^{17}$O (‰)")
    axes[0].set_title("Digitized Young and updated model")
    axes[1].axhline(0.0, color="#333333", lw=0.8)
    axes[1].set_ylabel("Curve-origin residual (‰)")
    axes[1].set_title("Response-shape residual")
    axes[2].set_ylabel("Required GPP (% of updated modern)")
    axes[2].set_title("Equivalent GPP diagnostic")
    for axis in axes:
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(color="#dddddd", lw=0.6)
        axis.legend(frameon=False, fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURE_PATH.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {FIGURE_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
