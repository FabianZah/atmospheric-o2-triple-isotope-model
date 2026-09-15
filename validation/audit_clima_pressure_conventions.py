"""Compare additive-CO2 and fixed-total-pressure Clima end members."""

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


def _load(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key(row: dict[str, object]) -> tuple[float, float]:
    return (
        float(row["pCO2_ppm"]),
        float(row["GPP_percent_of_updated_modern"]),
    )


def _plot(
    local_rows: list[dict[str, float]],
    global_rows: list[dict[str, float]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    pco2 = np.asarray([row["pCO2_ppm"] for row in local_rows])
    axes[0].plot(
        pco2,
        [row["additive_climate_over_fixed_D17O_forcing"] for row in local_rows],
        "o-",
        label="Additive CO$_2$ pressure",
    )
    axes[0].plot(
        pco2,
        [row["fixed_total_climate_over_fixed_D17O_forcing"] for row in local_rows],
        "s--",
        label="Fixed total dry pressure",
    )
    axes[0].axhline(1.0, color="0.45", linewidth=0.9)
    axes[0].set_ylabel("Clima / fixed R7 Δ′$^{17}$O forcing")
    axes[0].legend(frameon=False, fontsize=8)

    colors = {50.0: "#28728f", 100.0: "#2a9d76"}
    for percent in (50.0, 100.0):
        selected = [row for row in global_rows if row["GPP_percent"] == percent]
        x = [row["pCO2_ppm"] for row in selected]
        axes[1].plot(
            x,
            [row["central_D17O_permil"] for row in selected],
            "o-",
            color=colors[percent],
            label=f"{percent:g}% central",
        )
        axes[1].plot(
            x,
            [row["additive_climate_D17O_permil"] for row in selected],
            "s--",
            color=colors[percent],
            label=f"{percent:g}% additive",
        )
        axes[1].plot(
            x,
            [row["fixed_total_climate_D17O_permil"] for row in selected],
            "^:",
            color=colors[percent],
            label=f"{percent:g}% fixed total",
        )
        axes[2].plot(
            x,
            [row["fixed_total_minus_additive_D17O_permil"] for row in selected],
            "o-",
            color=colors[percent],
            label=f"{percent:g}% GPP",
        )
    axes[1].set_ylabel("Atmospheric O$_2$ Δ′$^{17}$O (‰)")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[2].axhline(0.0, color="0.45", linewidth=0.9)
    axes[2].set_ylabel("Fixed total − additive Δ′$^{17}$O (‰)")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(
    additive_attribution_path: Path,
    fixed_total_attribution_path: Path,
    additive_global_path: Path,
    fixed_total_global_path: Path,
    output_path: Path,
) -> dict[str, object]:
    additive_attribution = _load(additive_attribution_path)
    fixed_attribution = _load(fixed_total_attribution_path)
    additive_global = _load(additive_global_path)
    fixed_global = _load(fixed_total_global_path)
    if additive_attribution.get("pressure_convention") not in (
        None,
        "additive_co2",
    ):
        raise ValueError("first attribution report is not additive CO2")
    if fixed_attribution.get("pressure_convention") != "fixed_total_dry_major":
        raise ValueError("second attribution report is not fixed total dry major")

    additive_local = {
        float(row["pCO2_ppm"]): row for row in additive_attribution["rows"]
    }
    fixed_local = {
        float(row["pCO2_ppm"]): row for row in fixed_attribution["rows"]
    }
    if set(additive_local) != set(fixed_local):
        raise ValueError("local pressure-convention nodes differ")
    local_rows = [
        {
            "pCO2_ppm": pco2,
            "additive_climate_over_fixed_D17O_forcing": float(
                additive_local[pco2]["climate_over_fixed_D17O_forcing"]
            ),
            "fixed_total_climate_over_fixed_D17O_forcing": float(
                fixed_local[pco2]["climate_over_fixed_D17O_forcing"]
            ),
            "fixed_total_minus_additive_forcing_ratio": float(
                fixed_local[pco2]["climate_over_fixed_D17O_forcing"]
                - additive_local[pco2]["climate_over_fixed_D17O_forcing"]
            ),
        }
        for pco2 in sorted(additive_local)
    ]

    additive_states = {_key(row): row for row in additive_global["rows"]}
    fixed_states = {_key(row): row for row in fixed_global["rows"]}
    if set(additive_states) != set(fixed_states):
        raise ValueError("global pressure-convention nodes differ")
    global_rows: list[dict[str, float]] = []
    for key in sorted(additive_states):
        additive = additive_states[key]
        fixed = fixed_states[key]
        additive_value = float(additive["climate_cap_delta17_prime_permil"])
        fixed_value = float(fixed["climate_cap_delta17_prime_permil"])
        central = float(additive["fixed_cap_delta17_prime_permil"])
        if not np.isclose(
            central,
            float(fixed["fixed_cap_delta17_prime_permil"]),
            atol=1.0e-12,
        ):
            raise ValueError(f"central models differ at {key}")
        global_rows.append(
            {
                "pCO2_ppm": key[0],
                "GPP_percent": key[1],
                "central_D17O_permil": central,
                "additive_climate_D17O_permil": additive_value,
                "fixed_total_climate_D17O_permil": fixed_value,
                "fixed_total_minus_additive_D17O_permil": (
                    fixed_value - additive_value
                ),
                "additive_climate_shift_permil": additive_value - central,
                "fixed_total_climate_shift_permil": fixed_value - central,
            }
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(global_rows[0]))
        writer.writeheader()
        writer.writerows(global_rows)
    _plot(local_rows, global_rows, figure_path)

    convention_differences = np.asarray(
        [row["fixed_total_minus_additive_D17O_permil"] for row in global_rows]
    )
    through_30k = np.asarray(
        [
            row["fixed_total_minus_additive_D17O_permil"]
            for row in global_rows
            if row["pCO2_ppm"] <= 30000.0
        ]
    )
    high_local = [row for row in local_rows if row["pCO2_ppm"] >= 30000.0]
    report: dict[str, object] = {
        "audit": "Clima pressure-convention robustness",
        "status": "pressure_robust_structural_endmember_not_promoted",
        "central_model_changed": False,
        "pO2_PAL": 1.0,
        "pressure_conventions": ["additive_co2", "fixed_total_dry_major"],
        "local_R7": {
            "maximum_absolute_forcing_ratio_difference_high_pCO2": float(
                max(
                    abs(row["fixed_total_minus_additive_forcing_ratio"])
                    for row in high_local
                )
            ),
            "rows": local_rows,
        },
        "global_O2": {
            "maximum_absolute_D17O_difference_through_30000ppm": float(
                np.max(np.abs(through_30k))
            ),
            "maximum_absolute_D17O_difference_full_domain": float(
                np.max(np.abs(convention_differences))
            ),
            "RMSE_between_pressure_conventions_permil": float(
                np.sqrt(np.mean(convention_differences**2))
            ),
            "rows": global_rows,
        },
        "interpretation": {
            "supported": (
                "The large high-pCO2 climate-ozone response persists under both "
                "surface-pressure conventions and is therefore not an artifact of "
                "adding CO2 to total dry pressure."
            ),
            "remaining_limit": (
                "Both calculations retain the same simplified isothermal "
                "stratosphere and have been tested only at 1 PAL O2."
            ),
            "promotion_decision": False,
        },
        "sources": {
            "additive_attribution": str(Path(additive_attribution_path).resolve()),
            "fixed_total_attribution": str(Path(fixed_total_attribution_path).resolve()),
            "additive_global": str(Path(additive_global_path).resolve()),
            "fixed_total_global": str(Path(fixed_total_global_path).resolve()),
        },
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--additive-attribution",
        type=Path,
        default=ROOT / "outputs" / "clima_high_pco2_attribution.json",
    )
    parser.add_argument("--fixed-total-attribution", type=Path, required=True)
    parser.add_argument(
        "--additive-global",
        type=Path,
        default=ROOT / "outputs" / "clima_global_o2_response.json",
    )
    parser.add_argument("--fixed-total-global", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "clima_pressure_convention_audit.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.additive_attribution,
                args.fixed_total_attribution,
                args.additive_global,
                args.fixed_total_global,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
