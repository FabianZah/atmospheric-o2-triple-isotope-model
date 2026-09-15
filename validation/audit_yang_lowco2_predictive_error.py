"""Estimate a domain-limited Yang low-CO2 predictive residual scale.

The estimate is based on age-contiguous held-out residuals. Reported isotope
and CO2 analytical variances are kept explicit. Any remaining Gaussian scale
is a candidate for this validation domain only; it is not a whole-model error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "validation") not in sys.path:
    sys.path.insert(0, str(ROOT / "validation"))

from audit_yang_2022_co2_tracking import (  # noqa: E402
    BLOCK_COUNT,
    FIXED_GPP_PGC_PER_YEAR,
    FIXED_PO2_PAL,
    INPUT_PATH,
    _load_rows,
    _model_co2_only_prediction,
)


OUTPUT_PATH = ROOT / "outputs" / "yang_lowco2_predictive_error.json"


def _blocked_predictions(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    estimates = np.empty_like(observed)
    indices = np.arange(len(observed))
    for held_out in np.array_split(indices, BLOCK_COUNT):
        training = np.ones(len(observed), dtype=bool)
        training[held_out] = False
        offset = float(np.mean(observed[training] - predicted[training]))
        estimates[held_out] = predicted[held_out] + offset
    return estimates


def _local_response_slope(co2: np.ndarray) -> np.ndarray:
    step = 0.25
    plus, _ = _model_co2_only_prediction(co2 + step)
    minus, _ = _model_co2_only_prediction(co2 - step)
    return (plus - minus) / (2.0 * step)


def _fit_excess_sigma(residual: np.ndarray, known_sigma: np.ndarray) -> float:
    def objective(candidate: float) -> float:
        variance = known_sigma**2 + candidate**2
        return float(0.5 * np.sum(np.log(variance) + residual**2 / variance))

    upper = max(100.0, 4.0 * float(np.sqrt(np.mean(residual**2))))
    result = minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    if not result.success:
        raise RuntimeError(f"predictive-error fit failed: {result.message}")
    return float(result.x)


def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, object]:
    rows = _load_rows(input_path)
    age = np.asarray([row["gas_age_ka_bp"] for row in rows], dtype=float)
    co2 = np.asarray([row["co2_ppm"] for row in rows], dtype=float)
    co2_sigma = np.asarray([row["co2_1sigma_ppm"] for row in rows], dtype=float)
    observed = np.asarray(
        [row["cap_delta17_modern_air_relative_ppm"] for row in rows], dtype=float
    )
    isotope_sigma = np.asarray(
        [row["cap_delta17_1sigma_ppm"] for row in rows], dtype=float
    )
    predicted, surface = _model_co2_only_prediction(co2)
    held_out_prediction = _blocked_predictions(observed, predicted)
    residual = observed - held_out_prediction
    slope = _local_response_slope(co2)
    propagated_co2_sigma = np.abs(slope) * co2_sigma

    # Zero values in the archive denote unavailable isotope uncertainty, not
    # exact measurements. They remain in predictive skill metrics but are
    # excluded from the variance decomposition.
    variance_rows = isotope_sigma > 0.0
    known_sigma = np.sqrt(
        isotope_sigma[variance_rows] ** 2
        + propagated_co2_sigma[variance_rows] ** 2
    )
    variance_residual = residual[variance_rows]
    excess_sigma = _fit_excess_sigma(variance_residual, known_sigma)
    total_sigma = np.sqrt(known_sigma**2 + excess_sigma**2)
    standardized = variance_residual / total_sigma

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "gas_age_ka_bp",
                "measured_co2_ppm",
                "observed_Delta17_ppm",
                "held_out_prediction_Delta17_ppm",
                "held_out_residual_Delta17_ppm",
                "reported_isotope_1sigma_ppm",
                "propagated_co2_1sigma_ppm",
                "used_for_variance_decomposition",
            ),
        )
        writer.writeheader()
        for values in zip(
            age,
            co2,
            observed,
            held_out_prediction,
            residual,
            isotope_sigma,
            propagated_co2_sigma,
            variance_rows,
            strict=True,
        ):
            writer.writerow(dict(zip(writer.fieldnames, values, strict=True)))

    report: dict[str, object] = {
        "audit": "Yang et al. (2022) domain-limited low-CO2 predictive error",
        "status": "candidate_domain_specific_predictive_scale",
        "probabilistic_scope": {
            "eligible": True,
            "promoted_to_public_default": False,
            "reason": (
                "The residuals are age-block held out, but the fitted excess scale "
                "also contains unmodeled GPP/pO2 variability and reference-coordinate "
                "effects; it is not pure structural model error."
            ),
        },
        "domain": {
            "age_ka_BP": [float(np.min(age)), float(np.max(age))],
            "pCO2_ppm": [float(np.min(co2)), float(np.max(co2))],
            "pO2_PAL": FIXED_PO2_PAL,
            "GPP_PgC_per_year": FIXED_GPP_PGC_PER_YEAR,
            "isotope_coordinate": "modern-air-relative Delta-17O, lambda=0.516",
        },
        "cross_validation": {
            "age_contiguous_blocks": BLOCK_COUNT,
            "all_rows": int(len(rows)),
            "variance_decomposition_rows": int(np.sum(variance_rows)),
            "rows_with_unavailable_isotope_sigma": int(np.sum(~variance_rows)),
            "held_out_residual_bias_ppm": float(np.mean(residual)),
            "held_out_residual_RMSE_ppm": float(np.sqrt(np.mean(residual**2))),
            "held_out_residual_MAE_ppm": float(np.mean(np.abs(residual))),
        },
        "variance_decomposition": {
            "method": "heteroscedastic Gaussian maximum likelihood",
            "known_variance_components": [
                "reported Delta-17O analytical one-sigma",
                "CO2 analytical one-sigma propagated through local model slope",
            ],
            "median_reported_isotope_sigma_ppm": float(
                np.median(isotope_sigma[variance_rows])
            ),
            "median_propagated_co2_sigma_ppm": float(
                np.median(propagated_co2_sigma[variance_rows])
            ),
            "candidate_excess_predictive_sigma_ppm": excess_sigma,
            "candidate_excess_predictive_sigma_permil": excess_sigma / 1000.0,
            "standardized_residual_RMSE": float(
                np.sqrt(np.mean(standardized**2))
            ),
            "coverage_within_one_total_sigma": float(
                np.mean(np.abs(standardized) <= 1.0)
            ),
            "coverage_within_two_total_sigma": float(
                np.mean(np.abs(standardized) <= 2.0)
            ),
        },
        "interpretation": {
            "permitted": (
                "Use as a sensitivity likelihood only for a declared late-Quaternary, "
                "1 PAL, fixed-GPP application with the Yang reference convention."
            ),
            "not_permitted": (
                "Do not extrapolate this scale to high pCO2, low pO2, low GPP, "
                "spherule conversion, or the full model domain."
            ),
            "central_model_changed": False,
        },
        "surface_data_id": surface.surface_data_id,
        "outputs": {
            "json": str(output.relative_to(ROOT)),
            "csv": str(csv_path.relative_to(ROOT)),
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
