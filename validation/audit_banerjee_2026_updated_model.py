"""Apply the updated model to pristine Banerjee et al. (2026) air samples."""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from audit_banerjee_2026 import load_rows  # noqa: E402
from isotope_reference_frames import (  # noqa: E402
    BANERJEE_LAMBDA,
    MODEL_LAMBDA,
    PrimeIsotopeComposition,
    relative_cap_delta17_ppm,
)
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
)


PO2_PAL = 1.0
PI_GPP_PGC_PER_YEAR = 290.0
OBSERVED_ONE_SIGMA_PPM = 4.0
GPP_BOUNDS_PGC_PER_YEAR = (1.0, 850.0)
PACK_MODERN_CAP_DELTA17_PRIME_PERMIL = -0.432
PACK_MODERN_DELTA18_CONVENTIONAL_PERMIL = 23.9


def _composition(state) -> PrimeIsotopeComposition:
    return PrimeIsotopeComposition(
        delta18_prime_permil=float(state.delta18_prime_permil),
        cap_delta17_prime_permil=float(state.cap_delta17_prime_permil),
        lambda_ref=MODEL_LAMBDA,
    )


def _pack_modern_air_reference() -> PrimeIsotopeComposition:
    delta18_prime = 1000.0 * np.log1p(
        PACK_MODERN_DELTA18_CONVENTIONAL_PERMIL / 1000.0
    )
    return PrimeIsotopeComposition(
        delta18_prime_permil=float(delta18_prime),
        cap_delta17_prime_permil=PACK_MODERN_CAP_DELTA17_PRIME_PERMIL,
        lambda_ref=MODEL_LAMBDA,
    )


def _roots_for_target(
    pco2_ppm: float,
    target_ppm: float,
    reference: PrimeIsotopeComposition,
) -> list[float]:
    @lru_cache(maxsize=None)
    def residual(gpp: float) -> float:
        state = run_updated_central_state(
            UpdatedForwardInput(PO2_PAL, pco2_ppm, float(gpp))
        )
        predicted = relative_cap_delta17_ppm(
            _composition(state),
            reference,
            relative_lambda=BANERJEE_LAMBDA,
        )
        return predicted - target_ppm

    grid = np.geomspace(*GPP_BOUNDS_PGC_PER_YEAR, 65)
    values = np.asarray([residual(float(gpp)) for gpp in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:], strict=True
    ):
        if f_left == 0.0:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(
                float(brentq(residual, float(left), float(right), xtol=1.0e-8))
            )
    if values[-1] == 0.0:
        roots.append(float(grid[-1]))
    return roots


def _single_root(roots: list[float]) -> float | None:
    return roots[0] if len(roots) == 1 else None


def _mean_percent(rows: list[dict[str, object]], predicate) -> float | None:
    values = [
        float(row["inferred_GPP_percent_PI"])
        for row in rows
        if predicate(float(row["age_ka"])) and row["inferred_GPP_percent_PI"] is not None
    ]
    return None if not values else float(np.mean(values))


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    solved = [row for row in rows if row["inferred_GPP_percent_PI"] is not None]
    age_ma = np.asarray([float(row["age_ka"]) / 1000.0 for row in solved])
    gpp = np.asarray([float(row["inferred_GPP_percent_PI"]) for row in solved])
    lower = np.asarray([float(row["GPP_lower_percent_PI"]) for row in solved])
    upper = np.asarray([float(row["GPP_upper_percent_PI"]) for row in solved])
    co2 = np.asarray([float(row["measured_CO2_ppm"]) for row in solved])
    yerr = np.vstack((gpp - lower, upper - gpp))

    figure, axis = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    axis.errorbar(
        age_ma,
        gpp,
        yerr=yerr,
        fmt="none",
        ecolor="0.45",
        elinewidth=0.9,
        capsize=2.0,
        zorder=1,
    )
    points = axis.scatter(
        age_ma,
        gpp,
        c=co2,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.5,
        s=48,
        zorder=2,
    )
    axis.axhline(100.0, color="0.25", linewidth=1.0)
    axis.axvspan(0.8, 1.2, color="0.9", zorder=-2)
    axis.set(
        xlabel="Age (Ma)",
        ylabel="Inferred GPP (% of preindustrial)",
    )
    axis.grid(alpha=0.2)
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("Measured CO$_2$ (ppm)")
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run(output_path: Path) -> dict[str, object]:
    reference = _pack_modern_air_reference()
    pristine = [
        row
        for row in load_rows()
        if row["excluded_from_pristine_co2_dataset"] == "no"
        and row["measured_co2_ppm"] is not None
        and row["cap_delta17_average_ppm"] is not None
    ]

    rows: list[dict[str, object]] = []
    for source in pristine:
        pco2 = float(source["measured_co2_ppm"])
        observed = float(source["cap_delta17_average_ppm"])
        central_roots = _roots_for_target(pco2, observed, reference)
        minus_roots = _roots_for_target(
            pco2, observed - OBSERVED_ONE_SIGMA_PPM, reference
        )
        plus_roots = _roots_for_target(
            pco2, observed + OBSERVED_ONE_SIGMA_PPM, reference
        )
        central = _single_root(central_roots)
        uncertainty_roots = [
            value
            for value in (*minus_roots, *plus_roots)
            if np.isfinite(value)
        ]
        lower = min(uncertainty_roots) if central is not None and uncertainty_roots else None
        upper = max(uncertainty_roots) if central is not None and uncertainty_roots else None
        rows.append(
            {
                "table": source["table"],
                "source_row": source["row"],
                "core_id": source["core_id"],
                "age_ka": float(source["ar_age_ka"]),
                "measured_CO2_ppm": pco2,
                "observed_modern_air_relative_Delta17_ppm": observed,
                "inferred_GPP_PgC_per_year": central,
                "inferred_GPP_percent_PI": (
                    None if central is None else 100.0 * central / PI_GPP_PGC_PER_YEAR
                ),
                "GPP_lower_PgC_per_year": lower,
                "GPP_upper_PgC_per_year": upper,
                "GPP_lower_percent_PI": (
                    None if lower is None else 100.0 * lower / PI_GPP_PGC_PER_YEAR
                ),
                "GPP_upper_percent_PI": (
                    None if upper is None else 100.0 * upper / PI_GPP_PGC_PER_YEAR
                ),
                "central_root_count": len(central_roots),
                "minus_1sigma_root_count": len(minus_roots),
                "plus_1sigma_root_count": len(plus_roots),
            }
        )

    solved = [row for row in rows if row["inferred_GPP_percent_PI"] is not None]
    pre_mpt = _mean_percent(rows, lambda age: age > 1200.0)
    mpt = _mean_percent(rows, lambda age: 800.0 <= age <= 1200.0)
    post_mpt = _mean_percent(rows, lambda age: age < 800.0)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, figure_path)

    values = np.asarray(
        [float(row["inferred_GPP_percent_PI"]) for row in solved], dtype=float
    )
    report = {
        "audit": "Banerjee et al. (2026) updated-model GPP application",
        "status": "application_complete" if len(solved) == len(rows) else "partial_roots",
        "role": (
            "External paired-isotope/CO2 application and model-to-model benchmark; "
            "not a strict forward holdout because sample-specific GPP is inferred, "
            "not independently observed."
        ),
        "model_fitted_to_Banerjee": False,
        "assumptions": {
            "pO2_PAL": PO2_PAL,
            "preindustrial_GPP_PgC_per_year": PI_GPP_PGC_PER_YEAR,
            "observed_Delta17_one_sigma_ppm": OBSERVED_ONE_SIGMA_PPM,
            "reported_relative_lambda": BANERJEE_LAMBDA,
            "model_absolute_lambda": MODEL_LAMBDA,
            "modern_air_reference": {
                "source": "Pack (2021)",
                "cap_Delta17_prime_permil": PACK_MODERN_CAP_DELTA17_PRIME_PERMIL,
                "delta18_conventional_permil": PACK_MODERN_DELTA18_CONVENTIONAL_PERMIL,
            },
            "reference_policy": (
                "Each modeled ice-core sample is differenced from measured modern "
                "atmospheric O2 in Banerjee's lambda=0.518 coordinate. The modern "
                "reference is not represented by a preindustrial model state."
            ),
        },
        "coverage": {
            "pristine_rows": len(rows),
            "single_root_rows": len(solved),
            "multiple_or_missing_root_rows": len(rows) - len(solved),
        },
        "summary_GPP_percent_PI": {
            "mean_all": float(np.mean(values)),
            "median_all": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "post_MPT_younger_than_0p8Ma_mean": post_mpt,
            "MPT_0p8_to_1p2Ma_mean": mpt,
            "pre_MPT_older_than_1p2Ma_mean": pre_mpt,
            "MPT_minus_pre_MPT_percentage_points": (
                None if mpt is None or pre_mpt is None else mpt - pre_mpt
            ),
        },
        "comparison_to_paper_text": {
            "Banerjee_result": (
                "AL and Y-F inversions indicate early-Pleistocene GPP comparable "
                "to the last-0.8-Ma mean and a tentative approximately 10% increase "
                "across the MPT. Their point estimates are not archived in the "
                "provided workbook and are therefore not used as numerical targets."
            ),
            "score_against_digitized_Figure_5": False,
        },
        "rows": rows,
        "outputs": {"csv": str(csv_path), "figure": str(figure_path)},
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "banerjee_2026_updated_model_application.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({
        "status": report["status"],
        "coverage": report["coverage"],
        "summary_GPP_percent_PI": report["summary_GPP_percent_PI"],
    }, indent=2))


if __name__ == "__main__":
    main()
