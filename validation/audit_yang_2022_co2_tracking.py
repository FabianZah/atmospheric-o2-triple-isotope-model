"""Test whether the updated model tracks ice-core CO2-driven Delta-17O changes.

The CO2-only prediction fixes pO2 and GPP. An additive offset is allowed because
Yang et al. report modern-air-relative isotope values, while the tracking test
concerns temporal variation. No response amplitude is fitted to the isotope
data. Age-block cross-validation estimates the offset outside each held-out
block and compares the physical CO2 predictor with an intercept-only null.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from isotope_reference_frames import (  # noqa: E402
    MODEL_LAMBDA,
    PrimeIsotopeComposition,
    relative_cap_delta17_ppm,
)
from updated_output_surface import (  # noqa: E402
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)


INPUT_PATH = (
    ROOT
    / "outputs"
    / "yang_2022_co2_age_matched.csv"
)
OUTPUT_PATH = ROOT / "outputs" / "yang_2022_co2_tracking_audit.json"
YANG_RELATIVE_LAMBDA = 0.516
FIXED_PO2_PAL = 1.0
FIXED_GPP_PGC_PER_YEAR = 290.0
REFERENCE_CO2_PPM = 280.0
YANG_TB_SLOPE_PPM_PER_PPM = -0.55
BANERJEE_EMPIRICAL_SLOPE_PPM_PER_PPM = -1.0 / 1.6131
SMOOTHING_WINDOWS_KA = (0.0, 11.0, 21.0, 31.0)
BLOCK_COUNT = 8


def _composition(prediction) -> PrimeIsotopeComposition:
    return PrimeIsotopeComposition(
        delta18_prime_permil=float(prediction.central_delta18_prime_permil),
        cap_delta17_prime_permil=float(
            prediction.central_cap_delta17_prime_permil
        ),
        lambda_ref=MODEL_LAMBDA,
    )


def _load_rows(path: Path = INPUT_PATH) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    required = {
        "gas_age_ka_bp",
        "cap_delta17_modern_air_relative_ppm",
        "cap_delta17_1sigma_ppm",
        "co2_ppm",
        "co2_1sigma_ppm",
    }
    if not source or not required.issubset(source[0]):
        raise ValueError(f"Unexpected Yang/CO2 columns in {path}")
    rows = [{key: float(value) for key, value in row.items()} for row in source]
    return sorted(rows, key=lambda row: row["gas_age_ka_bp"])


def _model_co2_only_prediction(
    co2_ppm: np.ndarray,
) -> tuple[np.ndarray, object]:
    surface = load_updated_output_surface()
    reference = _composition(
        surface.evaluate(
            UpdatedOutputSurfaceInput(
                FIXED_PO2_PAL,
                REFERENCE_CO2_PPM,
                FIXED_GPP_PGC_PER_YEAR,
            )
        )
    )
    predicted = []
    for value in co2_ppm:
        sample = _composition(
            surface.evaluate(
                UpdatedOutputSurfaceInput(
                    FIXED_PO2_PAL,
                    float(value),
                    FIXED_GPP_PGC_PER_YEAR,
                )
            )
        )
        predicted.append(
            relative_cap_delta17_ppm(
                sample,
                reference,
                relative_lambda=YANG_RELATIVE_LAMBDA,
            )
        )
    return np.asarray(predicted, dtype=float), surface


def _moving_window_mean(
    age_ka: np.ndarray, values: np.ndarray, width_ka: float
) -> np.ndarray:
    if width_ka <= 0.0:
        return values.copy()
    half_width = width_ka / 2.0
    return np.asarray(
        [np.mean(values[np.abs(age_ka - age) <= half_width]) for age in age_ka],
        dtype=float,
    )


def _tracking_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    offset = float(np.mean(observed - predicted))
    aligned = predicted + offset
    residual = observed - aligned
    null_residual = observed - np.mean(observed)
    null_sse = float(np.sum(null_residual**2))
    slope, intercept = np.polyfit(predicted, observed, 1)
    return {
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
        "offset_ppm": offset,
        "fixed_amplitude_rmse_ppm": float(np.sqrt(np.mean(residual**2))),
        "intercept_only_rmse_ppm": float(np.sqrt(np.mean(null_residual**2))),
        "fixed_amplitude_variance_skill": float(
            1.0 - np.sum(residual**2) / null_sse
        ),
        "diagnostic_fitted_amplitude": float(slope),
        "diagnostic_fitted_intercept_ppm": float(intercept),
        "observed_standard_deviation_ppm": float(np.std(observed)),
        "predicted_standard_deviation_ppm": float(np.std(predicted)),
    }


def _blocked_cross_validation(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    block_count: int = BLOCK_COUNT,
) -> dict[str, float | int]:
    indices = np.arange(len(observed))
    model_estimate = np.empty_like(observed)
    null_estimate = np.empty_like(observed)
    for held_out in np.array_split(indices, block_count):
        training = np.ones(len(observed), dtype=bool)
        training[held_out] = False
        offset = float(np.mean(observed[training] - predicted[training]))
        model_estimate[held_out] = predicted[held_out] + offset
        null_estimate[held_out] = float(np.mean(observed[training]))
    model_sse = float(np.sum((observed - model_estimate) ** 2))
    null_sse = float(np.sum((observed - null_estimate) ** 2))
    return {
        "age_contiguous_blocks": block_count,
        "model_rmse_ppm": float(np.sqrt(model_sse / len(observed))),
        "intercept_only_rmse_ppm": float(np.sqrt(null_sse / len(observed))),
        "variance_skill": float(1.0 - model_sse / null_sse),
    }


def _blocked_co2_inversion(
    observed: np.ndarray,
    measured_co2: np.ndarray,
    predicted_at_measured_co2: np.ndarray,
    response_co2: np.ndarray,
    response_delta17: np.ndarray,
    *,
    block_count: int = BLOCK_COUNT,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Invert held-out isotope values with offsets learned outside each block."""

    indices = np.arange(len(observed))
    inferred = np.full(len(observed), np.nan, dtype=float)
    response_min = float(np.min(response_delta17))
    response_max = float(np.max(response_delta17))
    for held_out in np.array_split(indices, block_count):
        training = np.ones(len(observed), dtype=bool)
        training[held_out] = False
        offset = float(
            np.mean(observed[training] - predicted_at_measured_co2[training])
        )
        target = observed[held_out] - offset
        in_domain = (target >= response_min) & (target <= response_max)
        inferred[held_out[in_domain]] = np.interp(
            target[in_domain],
            response_delta17[::-1],
            response_co2[::-1],
        )

    solved = np.isfinite(inferred)
    residual = inferred[solved] - measured_co2[solved]
    null_estimate = np.empty(len(observed), dtype=float)
    for held_out in np.array_split(indices, block_count):
        training = np.ones(len(observed), dtype=bool)
        training[held_out] = False
        null_estimate[held_out] = float(np.mean(measured_co2[training]))
    null_residual = null_estimate[solved] - measured_co2[solved]
    return {
        "age_contiguous_blocks": block_count,
        "root_count": int(np.sum(solved)),
        "out_of_domain_count": int(np.sum(~solved)),
        "rmse_ppm": float(np.sqrt(np.mean(residual**2))),
        "mean_absolute_error_ppm": float(np.mean(np.abs(residual))),
        "bias_ppm": float(np.mean(residual)),
        "pearson_r": float(
            np.corrcoef(inferred[solved], measured_co2[solved])[0, 1]
        ),
        "intercept_only_rmse_ppm": float(
            np.sqrt(np.mean(null_residual**2))
        ),
    }, inferred


def _plot(
    age_ka: np.ndarray,
    co2_ppm: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    response_co2: np.ndarray,
    response_delta17: np.ndarray,
    inferred_co2_21ka: np.ndarray,
    measured_co2_21ka: np.ndarray,
    output: Path,
) -> None:
    observed_21 = _moving_window_mean(age_ka, observed, 21.0)
    predicted_21 = _moving_window_mean(age_ka, predicted, 21.0)
    aligned_21 = predicted_21 + np.mean(observed_21 - predicted_21)
    segment_starts = np.r_[0, np.flatnonzero(np.diff(age_ka) > 10.0) + 1]
    segment_ends = np.r_[segment_starts[1:], len(age_ka)]

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.2), constrained_layout=True)
    for segment_index, (start, end) in enumerate(
        zip(segment_starts, segment_ends, strict=True)
    ):
        axes[0].plot(
            age_ka[start:end],
            observed_21[start:end],
            color="#202020",
            linewidth=1.5,
            label="Observed" if segment_index == 0 else None,
        )
        axes[0].plot(
            age_ka[start:end],
            aligned_21[start:end],
            color="#007f78",
            linewidth=1.6,
            label="Updated model, CO$_2$ only" if segment_index == 0 else None,
        )
    axes[0].set(ylabel=r"Modern-air-relative $\Delta^{17}$O (ppm)")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.18)

    solved = np.isfinite(inferred_co2_21ka)
    points = axes[1].scatter(
        measured_co2_21ka[solved],
        inferred_co2_21ka[solved],
        c=age_ka[solved],
        cmap="viridis_r",
        s=22,
        linewidth=0.0,
    )
    limits = [
        min(
            float(np.min(measured_co2_21ka[solved])),
            float(np.min(inferred_co2_21ka[solved])),
        ),
        max(
            float(np.max(measured_co2_21ka[solved])),
            float(np.max(inferred_co2_21ka[solved])),
        ),
    ]
    axes[1].plot(limits, limits, color="0.25", linewidth=1.0)
    axes[1].set(
        xlabel="Measured CO$_2$, 21-kyr mean (ppm)",
        ylabel=r"Cross-validated CO$_2$ from $\Delta^{17}$O (ppm)",
    )
    axes[1].grid(alpha=0.18)
    colorbar = figure.colorbar(points, ax=axes[1], pad=0.02)
    colorbar.set_label("Gas age (ka BP)")

    axes[2].plot(
        response_co2,
        response_delta17,
        color="#007f78",
        linewidth=2.0,
        label="Updated model",
    )
    axes[2].plot(
        response_co2,
        YANG_TB_SLOPE_PPM_PER_PPM * (response_co2 - REFERENCE_CO2_PPM),
        color="#bf5b17",
        linestyle="--",
        linewidth=1.4,
        label="Yang et al. TB: -0.55 ppm/ppm",
    )
    axes[2].plot(
        response_co2,
        BANERJEE_EMPIRICAL_SLOPE_PPM_PER_PPM
        * (response_co2 - REFERENCE_CO2_PPM),
        color="#6a51a3",
        linestyle=":",
        linewidth=1.5,
        label="Banerjee empirical relation",
    )
    axes[2].set(
        xlabel="Atmospheric CO$_2$ (ppm)",
        ylabel=r"CO$_2$-only $\Delta^{17}$O change (ppm)",
    )
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.18)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, object]:
    rows = _load_rows(input_path)
    age = np.asarray([row["gas_age_ka_bp"] for row in rows], dtype=float)
    co2 = np.asarray([row["co2_ppm"] for row in rows], dtype=float)
    observed = np.asarray(
        [row["cap_delta17_modern_air_relative_ppm"] for row in rows], dtype=float
    )
    predicted, surface = _model_co2_only_prediction(co2)

    response_co2 = np.linspace(float(np.min(co2)), float(np.max(co2)), 501)
    response_delta17, _ = _model_co2_only_prediction(response_co2)
    response_slope, response_intercept = np.polyfit(
        response_co2, response_delta17, 1
    )
    linear_residual = response_delta17 - (
        response_slope * response_co2 + response_intercept
    )

    inversion_response_co2 = np.linspace(150.0, 600.0, 901)
    inversion_response_delta17, _ = _model_co2_only_prediction(
        inversion_response_co2
    )
    smoothing = {}
    inferred_by_window: dict[str, np.ndarray] = {}
    measured_co2_by_window: dict[str, np.ndarray] = {}
    for width in SMOOTHING_WINDOWS_KA:
        observed_smooth = _moving_window_mean(age, observed, width)
        predicted_smooth = _moving_window_mean(age, predicted, width)
        measured_co2_smooth = _moving_window_mean(age, co2, width)
        co2_inversion, inferred_co2 = _blocked_co2_inversion(
            observed_smooth,
            measured_co2_smooth,
            predicted_smooth,
            inversion_response_co2,
            inversion_response_delta17,
        )
        key = f"{width:g}_ka"
        inferred_by_window[key] = inferred_co2
        measured_co2_by_window[key] = measured_co2_smooth
        smoothing[key] = {
            **_tracking_metrics(observed_smooth, predicted_smooth),
            "blocked_cross_validation": _blocked_cross_validation(
                observed_smooth, predicted_smooth
            ),
            "cross_validated_CO2_inversion": co2_inversion,
        }

    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    aligned = predicted + np.mean(observed - predicted)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "gas_age_ka_bp",
                "measured_co2_ppm",
                "observed_Delta17_ppm",
                "model_CO2_only_Delta17_ppm",
                "aligned_model_CO2_only_Delta17_ppm",
                "CO2_only_residual_ppm",
            ),
        )
        writer.writeheader()
        for age_value, co2_value, obs, model, model_aligned in zip(
            age, co2, observed, predicted, aligned, strict=True
        ):
            writer.writerow(
                {
                    "gas_age_ka_bp": age_value,
                    "measured_co2_ppm": co2_value,
                    "observed_Delta17_ppm": obs,
                    "model_CO2_only_Delta17_ppm": model,
                    "aligned_model_CO2_only_Delta17_ppm": model_aligned,
                    "CO2_only_residual_ppm": obs - model_aligned,
                }
            )
    _plot(
        age,
        co2,
        observed,
        predicted,
        response_co2,
        response_delta17,
        inferred_by_window["21_ka"],
        measured_co2_by_window["21_ka"],
        figure_path,
    )

    raw_cv = smoothing["0_ka"]["blocked_cross_validation"]
    report = {
        "audit": "Yang et al. (2022) measured-CO2 tracking test",
        "status": "tracking_supported" if raw_cv["variance_skill"] > 0.0 else "tracking_not_supported",
        "role": (
            "External temporal tracking test. Measured CO2 is an independent "
            "predictor; GPP and pO2 are fixed. Only an additive reference offset "
            "is estimated, never the physical response amplitude."
        ),
        "data": {
            "rows": len(rows),
            "age_ka_bp": [float(np.min(age)), float(np.max(age))],
            "measured_CO2_ppm": [float(np.min(co2)), float(np.max(co2))],
            "isotope_coordinate": "modern-air-relative Delta-17O, lambda=0.516, ppm",
        },
        "fixed_model_conditions": {
            "pO2_PAL": FIXED_PO2_PAL,
            "GPP_PgC_per_year": FIXED_GPP_PGC_PER_YEAR,
            "reference_CO2_ppm": REFERENCE_CO2_PPM,
            "surface_data_id": surface.surface_data_id,
        },
        "co2_response": {
            "updated_model_slope_ppm_Delta17_per_ppm_CO2": float(response_slope),
            "Yang_2022_TB_slope_ppm_per_ppm": YANG_TB_SLOPE_PPM_PER_PPM,
            "relative_slope_difference": float(
                response_slope / YANG_TB_SLOPE_PPM_PER_PPM - 1.0
            ),
            "Banerjee_2026_empirical_slope_ppm_per_ppm": BANERJEE_EMPIRICAL_SLOPE_PPM_PER_PPM,
            "maximum_nonlinearity_about_linear_fit_ppm": float(
                np.max(np.abs(linear_residual))
            ),
        },
        "tracking_by_smoothing_window": smoothing,
        "interpretation": {
            "supported": (
                "Measured CO2 supplies out-of-block predictive information for "
                "Delta-17O with the model amplitude fixed."
            ),
            "not_identified": (
                "Residual isotope variation cannot be assigned uniquely to GPP; "
                "it also includes measurement noise and unmodeled process change."
            ),
            "inversion_scope": (
                "The raw individual-sample inversion is less accurate than an "
                "intercept-only CO2 comparator. The 21-kyr signal tracks measured "
                "CO2 substantially better, so support applies to coherent glacial-"
                "scale variation rather than precise single-sample CO2 estimates."
            ),
            "smoothing_caveat": (
                "Moving-window metrics reproduce Yang's descriptive comparison "
                "but overlapping windows are not statistically independent."
            ),
        },
        "outputs": {
            "json": str(output.relative_to(ROOT)),
            "csv": str(csv_path.relative_to(ROOT)),
            "figure": str(figure_path.relative_to(ROOT)),
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = run(args.input, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "co2_response": report["co2_response"],
                "tracking_by_smoothing_window": report[
                    "tracking_by_smoothing_window"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
