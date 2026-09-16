"""Merge the native Liu low-GPP grid and compare it with the updated model."""

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
    path
    for path in Path(__file__).resolve().parents
    if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
)


DEFAULT_INPUT = ROOT / "model_data/literature/liu_2021_reference_grid.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "liu_2021_low_gpp_multimodel_benchmark.json"
UPDATED_REFERENCE_GPP_PGC_PER_YEAR = 290.0
TMOL_O2_TO_PGC_AT_ONE_TO_ONE = 0.012


def _updated_state(
    *, p_o2_pal: float, p_co2_ppm: float, gpp_pgC_per_year: float
) -> tuple[float, str, str]:
    try:
        updated = run_updated_central_state(
            UpdatedForwardInput(
                p_o2_pal=p_o2_pal,
                p_co2_ppm=p_co2_ppm,
                gpp_pgC_per_year=gpp_pgC_per_year,
            )
        )
        return float(updated.cap_delta17_prime_permil), "solved", ""
    except ValueError as error:
        return float("nan"), "outside_physical_domain", str(error)


def load_columns(input_directory: Path) -> list[dict[str, object]]:
    if Path(input_directory).is_file():
        columns = json.loads(Path(input_directory).read_text(encoding="utf-8"))["columns"]
    else:
        paths = sorted(Path(input_directory).rglob("column.json"))
        columns = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not columns:
        raise FileNotFoundError(f"no Liu reference columns in {input_directory}")
    keys = [(float(row["pO2_PAL"]), float(row["pCO2_ppm"])) for row in columns]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate Liu pO2-pCO2 columns in downloaded artifacts")
    if any(bool(row.get("dry_run")) for row in columns):
        raise ValueError("native Liu benchmark cannot merge dry-run columns")
    return columns


def build_comparison_rows(columns: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in columns:
        for native in column["scenarios"]:
            gpp_percent = float(native["GPP_percent_of_Liu_modern_marine"])
            po2 = float(native["pO2_PAL"])
            pco2 = float(native["pCO2_ppm"])
            accessible_o2 = float(native["accessible_O2_exchange_Tmol_per_year"])
            own_reference_gpp = (
                UPDATED_REFERENCE_GPP_PGC_PER_YEAR * gpp_percent / 100.0
            )
            accessible_turnover_gpp = accessible_o2 * TMOL_O2_TO_PGC_AT_ONE_TO_ONE
            updated_cap, updated_domain_status, updated_domain_reason = _updated_state(
                p_o2_pal=po2,
                p_co2_ppm=pco2,
                gpp_pgC_per_year=own_reference_gpp,
            )
            (
                turnover_cap,
                turnover_domain_status,
                turnover_domain_reason,
            ) = _updated_state(
                p_o2_pal=po2,
                p_co2_ppm=pco2,
                gpp_pgC_per_year=accessible_turnover_gpp,
            )
            rows.append(
                {
                    "pO2_PAL": po2,
                    "pCO2_ppm": pco2,
                    "GPP_percent_of_own_reference": gpp_percent,
                    "Liu_total_marine_GPP_Tmol_O2_per_year": float(
                        native["total_marine_GPP_Tmol_O2_per_year"]
                    ),
                    "Liu_accessible_O2_exchange_Tmol_per_year": accessible_o2,
                    "Liu_accessible_fraction": float(native["accessible_fraction"]),
                    "Liu_cap_delta17_prime_permil": float(
                        native["surface_cap_delta17_prime_permil"]
                    ),
                    "updated_global_GPP_PgC_per_year": own_reference_gpp,
                    "updated_cap_delta17_prime_permil": updated_cap,
                    "updated_domain_status": updated_domain_status,
                    "updated_domain_reason": updated_domain_reason,
                    "updated_accessible_turnover_GPP_PgC_per_year": (
                        accessible_turnover_gpp
                    ),
                    "updated_accessible_turnover_cap_delta17_prime_permil": (
                        turnover_cap
                    ),
                    "updated_accessible_turnover_domain_status": (
                        turnover_domain_status
                    ),
                    "updated_accessible_turnover_domain_reason": (
                        turnover_domain_reason
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["pO2_PAL"]),
            float(row["pCO2_ppm"]),
            float(row["GPP_percent_of_own_reference"]),
        )
    )
    for po2 in sorted({float(row["pO2_PAL"]) for row in rows}):
        subset = [row for row in rows if float(row["pO2_PAL"]) == po2]
        reference = next(
            row
            for row in subset
            if float(row["pCO2_ppm"]) == 300.0
            and float(row["GPP_percent_of_own_reference"]) == 100.0
        )
        for row in subset:
            row["Liu_response_from_300ppm_100pct_permil"] = float(
                row["Liu_cap_delta17_prime_permil"]
                - reference["Liu_cap_delta17_prime_permil"]
            )
            row["updated_response_from_300ppm_100pct_permil"] = float(
                row["updated_cap_delta17_prime_permil"]
                - reference["updated_cap_delta17_prime_permil"]
            )
            row["response_difference_updated_minus_Liu_permil"] = float(
                row["updated_response_from_300ppm_100pct_permil"]
                - row["Liu_response_from_300ppm_100pct_permil"]
            )
            row["updated_accessible_turnover_response_permil"] = float(
                row["updated_accessible_turnover_cap_delta17_prime_permil"]
                - reference["updated_accessible_turnover_cap_delta17_prime_permil"]
            )
            row["accessible_turnover_difference_updated_minus_Liu_permil"] = float(
                row["updated_accessible_turnover_response_permil"]
                - row["Liu_response_from_300ppm_100pct_permil"]
            )
    return rows


def shape_metrics(
    rows: list[dict[str, object]],
    *,
    updated_cap_field: str = "updated_cap_delta17_prime_permil",
    updated_response_field: str = "updated_response_from_300ppm_100pct_permil",
    updated_label: str = "updated",
) -> dict[str, object]:
    pco2_checks: list[dict[str, object]] = []
    gpp_checks: list[dict[str, object]] = []
    for po2 in sorted({float(row["pO2_PAL"]) for row in rows}):
        for gpp in sorted({float(row["GPP_percent_of_own_reference"]) for row in rows}):
            selected = sorted(
                (
                    row
                    for row in rows
                    if float(row["pO2_PAL"]) == po2
                    and float(row["GPP_percent_of_own_reference"]) == gpp
                ),
                key=lambda row: float(row["pCO2_ppm"]),
            )
            for model, field in (
                ("Liu", "Liu_cap_delta17_prime_permil"),
                (updated_label, updated_cap_field),
            ):
                values = np.asarray(
                    [float(row[field]) for row in selected]
                )
                values = values[np.isfinite(values)]
                pco2_checks.append(
                    {
                        "model": model,
                        "pO2_PAL": po2,
                        "GPP_percent": gpp,
                        "available_state_count": int(len(values)),
                        "more_negative_with_pCO2": bool(
                            len(values) >= 2 and np.all(np.diff(values) < 0.0)
                        ),
                    }
                )
        for pco2 in sorted({float(row["pCO2_ppm"]) for row in rows}):
            selected = sorted(
                (
                    row
                    for row in rows
                    if float(row["pO2_PAL"]) == po2 and float(row["pCO2_ppm"]) == pco2
                ),
                key=lambda row: float(row["GPP_percent_of_own_reference"]),
            )
            for model, field in (
                ("Liu", "Liu_cap_delta17_prime_permil"),
                (updated_label, updated_cap_field),
            ):
                values = np.asarray(
                    [float(row[field]) for row in selected]
                )
                values = values[np.isfinite(values)]
                gpp_checks.append(
                    {
                        "model": model,
                        "pO2_PAL": po2,
                        "pCO2_ppm": pco2,
                        "available_state_count": int(len(values)),
                        "less_negative_with_GPP": bool(
                            len(values) >= 2 and np.all(np.diff(values) > 0.0)
                        ),
                    }
                )
    liu_response = np.asarray(
        [float(row["Liu_response_from_300ppm_100pct_permil"]) for row in rows]
    )
    updated_response = np.asarray(
        [float(row[updated_response_field]) for row in rows]
    )
    finite = np.isfinite(liu_response) & np.isfinite(updated_response)
    difference = updated_response[finite] - liu_response[finite]
    return {
        "response_rank_correlation": float(
            spearmanr(liu_response[finite], updated_response[finite]).statistic
        ),
        "paired_finite_state_count": int(np.sum(finite)),
        "updated_unavailable_state_count": int(np.sum(~np.isfinite(updated_response))),
        "response_RMSE_permil": float(np.sqrt(np.mean(difference**2))),
        "response_mean_bias_permil": float(np.mean(difference)),
        "response_max_abs_difference_permil": float(np.max(np.abs(difference))),
        "all_pCO2_direction_checks_pass": all(
            bool(row["more_negative_with_pCO2"]) for row in pco2_checks
        ),
        "all_GPP_direction_checks_pass": all(
            bool(row["less_negative_with_GPP"]) for row in gpp_checks
        ),
        "pCO2_direction_checks": pco2_checks,
        "GPP_direction_checks": gpp_checks,
    }


def _grid(
    rows: list[dict[str, object]], po2: float, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if float(row["pO2_PAL"]) == po2]
    x = np.asarray(sorted({float(row["pCO2_ppm"]) for row in selected}))
    y = np.asarray(
        sorted({float(row["GPP_percent_of_own_reference"]) for row in selected})
    )
    z = np.empty((len(y), len(x)))
    for yi, gpp in enumerate(y):
        for xi, pco2 in enumerate(x):
            row = next(
                row
                for row in selected
                if float(row["pCO2_ppm"]) == pco2
                and float(row["GPP_percent_of_own_reference"]) == gpp
            )
            z[yi, xi] = float(row[field])
    return x, y, z


def plot(
    rows: list[dict[str, object]],
    path: Path,
    *,
    updated_response_field: str = "updated_response_from_300ppm_100pct_permil",
    updated_title: str = "Updated model",
    difference_title: str = "Updated minus Liu",
) -> None:
    po2_axis = sorted({float(row["pO2_PAL"]) for row in rows})
    fields = (
        "Liu_response_from_300ppm_100pct_permil",
        updated_response_field,
        (
            "accessible_turnover_difference_updated_minus_Liu_permil"
            if updated_response_field == "updated_accessible_turnover_response_permil"
            else "response_difference_updated_minus_Liu_permil"
        ),
    )
    titles = ("Liu 2021", updated_title, difference_title)
    response_values = np.asarray(
        [float(row[field]) for row in rows for field in fields[:2]]
    )
    response_values = response_values[np.isfinite(response_values)]
    response_limit = max(abs(response_values.min()), abs(response_values.max()))
    difference_values = np.asarray([float(row[fields[2]]) for row in rows])
    difference_values = difference_values[np.isfinite(difference_values)]
    difference_limit = max(abs(difference_values.min()), abs(difference_values.max()))
    figure, axes = plt.subplots(
        len(po2_axis), 3, figsize=(13.2, 10.6), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    response_mappable = None
    difference_mappable = None
    for row_index, po2 in enumerate(po2_axis):
        for column_index, (field, title) in enumerate(zip(fields, titles)):
            x, y, z = _grid(rows, po2, field)
            levels = np.linspace(
                -difference_limit if column_index == 2 else -response_limit,
                difference_limit if column_index == 2 else response_limit,
                25,
            )
            mappable = axes[row_index, column_index].contourf(
                x, y, z, levels=levels, cmap="RdBu_r", extend="both"
            )
            axes[row_index, column_index].contour(
                x, y, z, levels=levels[::4], colors="k", linewidths=0.35, alpha=0.5
            )
            axes[row_index, column_index].set_xscale("log")
            axes[row_index, column_index].set_title(
                f"{title}; {po2:g} PAL O$_2$", fontsize=10
            )
            axes[row_index, column_index].set_xlabel("pCO$_2$ (ppm)")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel("Marine GPP (% own reference)")
            if column_index == 2:
                difference_mappable = mappable
            else:
                response_mappable = mappable
    figure.colorbar(
        response_mappable,
        ax=axes[:, :2],
        label="Response from 300 ppm, 100% GPP (per mil)",
        shrink=0.82,
    )
    figure.colorbar(
        difference_mappable,
        ax=axes[:, 2],
        label=f"{difference_title} response (per mil)",
        shrink=0.82,
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(input_directory: Path, output_path: Path) -> dict[str, object]:
    columns = load_columns(input_directory)
    rows = build_comparison_rows(columns)
    metrics = shape_metrics(rows)
    accessible_turnover_metrics = shape_metrics(
        rows,
        updated_cap_field="updated_accessible_turnover_cap_delta17_prime_permil",
        updated_response_field="updated_accessible_turnover_response_permil",
        updated_label="updated_accessible_turnover",
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    figure_path = output_path.with_suffix(".png")
    turnover_figure_path = output_path.with_name(
        f"{output_path.stem}_accessible_turnover.png"
    )
    _write_csv(rows, csv_path)
    plot(rows, figure_path)
    plot(
        rows,
        turnover_figure_path,
        updated_response_field="updated_accessible_turnover_response_permil",
        updated_title="Updated (turnover matched)",
        difference_title="Difference (turnover matched)",
    )
    report = {
        "schema_version": 2,
        "benchmark": "Liu 2021 native high-pCO2 low-GPP multimodel comparison",
        "comparison_policy": (
            "Both own-reference-percent and atmosphere-accessible-turnover "
            "coordinates are reported. Responses are differenced from 300 ppm "
            "and 100% GPP at each pO2. Absolute isotope offsets, total global "
            "GPP, and marine atmosphere-accessible O2 exchange are not conflated."
        ),
        "column_count": len(columns),
        "scenario_count": len(rows),
        "pO2_PAL_axis": sorted({float(row["pO2_PAL"]) for row in rows}),
        "pCO2_ppm_axis": sorted({float(row["pCO2_ppm"]) for row in rows}),
        "GPP_percent_axis": sorted(
            {float(row["GPP_percent_of_own_reference"]) for row in rows}
        ),
        "metrics": metrics,
        "own_reference_percent_metrics": metrics,
        "accessible_turnover_matched_metrics": accessible_turnover_metrics,
        "scenarios": rows,
        "csv": str(csv_path),
        "own_reference_percent_figure": str(figure_path),
        "accessible_turnover_matched_figure": str(turnover_figure_path),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.input_directory, arguments.output), indent=2))


if __name__ == "__main__":
    main()
