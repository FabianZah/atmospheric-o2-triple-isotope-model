"""Drive the updated global O2 reservoir with the continuous 800-kyr CO2 record.

This is a forward tracking test at fixed GPP and pO2. The fast molecular
response surface is evaluated at the continuously interpolated Bereiter CO2
forcing and the evolving global-O2 isotope state. No relaxation time or isotope
response amplitude is fitted. Only an additive modern-air reference offset is
estimated when comparing predictions with Yang et al. observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from global_o2_isotope_reservoir import (  # noqa: E402
    GlobalO2Reservoir,
    IsotopologueTendency,
    sum_tendencies,
)
from isotope_reference_frames import (  # noqa: E402
    MODEL_LAMBDA,
    PrimeIsotopeComposition,
    relative_cap_delta17_ppm,
)
from prepare_ice_core_holdout import load_bereiter_co2  # noqa: E402
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    build_updated_central_forcing,
    run_updated_central_state,
)
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL  # noqa: E402
from audit_yang_2022_co2_tracking import (  # noqa: E402
    BLOCK_COUNT,
    FIXED_GPP_PGC_PER_YEAR,
    FIXED_PO2_PAL,
    INPUT_PATH,
    REFERENCE_CO2_PPM,
    SMOOTHING_WINDOWS_KA,
    YANG_RELATIVE_LAMBDA,
    _blocked_cross_validation,
    _load_rows,
    _model_co2_only_prediction,
    _moving_window_mean,
    _tracking_metrics,
)


OUTPUT_PATH = ROOT / "outputs" / "yang_2022_transient_co2_tracking_audit.json"
INITIAL_CONDITION_GUARD_YEARS = 30_000.0
REGULAR_OUTPUT_STEP_YEARS = 250.0
MAXIMUM_SOLVER_STEP_YEARS = 250.0


def _co2_at_age(
    age_year_bp: float | np.ndarray,
    source_age_year_bp: np.ndarray,
    source_co2_ppm: np.ndarray,
) -> float | np.ndarray:
    return np.interp(age_year_bp, source_age_year_bp, source_co2_ppm)


def _composition_from_state(state: GlobalO2Reservoir) -> PrimeIsotopeComposition:
    return PrimeIsotopeComposition(
        delta18_prime_permil=state.delta18_prime_permil,
        cap_delta17_prime_permil=state.cap_delta17_prime_permil,
        lambda_ref=MODEL_LAMBDA,
    )


def _relative_prediction_ppm(
    states: tuple[GlobalO2Reservoir, ...],
    reference: PrimeIsotopeComposition,
) -> np.ndarray:
    return np.asarray(
        [
            relative_cap_delta17_ppm(
                _composition_from_state(state),
                reference,
                relative_lambda=YANG_RELATIVE_LAMBDA,
            )
            for state in states
        ],
        dtype=float,
    )


def _states_from_solution(
    solution, times: np.ndarray, scale: np.ndarray
) -> tuple[GlobalO2Reservoir, ...]:
    values = solution.sol(times)
    return tuple(
        GlobalO2Reservoir(
            *map(float, values[:, index] * scale),
            source=f"continuous CO2 trajectory at forward time {time:g} yr",
        )
        for index, time in enumerate(times)
    )


def _integrate_continuous_co2(
    source_age_year_bp: np.ndarray,
    source_co2_ppm: np.ndarray,
    output_times_year: np.ndarray,
):
    start_age = float(source_age_year_bp[-1])
    end_age = start_age - float(np.max(output_times_year))
    initial_co2 = float(
        _co2_at_age(start_age, source_age_year_bp, source_co2_ppm)
    )
    request = UpdatedForwardInput(
        FIXED_PO2_PAL,
        initial_co2,
        FIXED_GPP_PGC_PER_YEAR,
    )
    initial_state = run_updated_central_state(request)
    initial = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=FIXED_PO2_PAL * GLOBAL_MAJOR_O2_MOLES_1PAL,
        delta18_prime_permil=initial_state.delta18_prime_permil,
        cap_delta17_prime_permil=initial_state.cap_delta17_prime_permil,
        source=(
            f"steady updated-model initial condition at {start_age:g} yr BP "
            f"and {initial_co2:g} ppm CO2"
        ),
    )
    forcing = build_updated_central_forcing(request)
    scale = initial.isotopologue_moles

    def tendency(time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        age = start_age - time
        pco2 = float(_co2_at_age(age, source_age_year_bp, source_co2_ppm))
        inventory = scaled_inventory * scale
        reservoir = GlobalO2Reservoir(
            *map(float, inventory),
            source="continuous ice-core CO2 transient integration state",
        )
        native_photo = forcing.surface.evaluate_prime_tendency_at(
            np.asarray(
                [
                    reservoir.delta18_prime_permil,
                    reservoir.cap_delta17_prime_permil,
                ]
            ),
            po2_pal=FIXED_PO2_PAL,
            pco2_ppm=pco2,
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        photo = IsotopologueTendency(
            *map(float, forcing.forcing_scale * native_photo.values),
            source=(
                "updated molecular response under continuous Bereiter CO2 forcing"
            ),
        )
        biology = forcing.biological_budget(photo)
        total = sum_tendencies(
            (biology.tendency(reservoir), photo),
            source="fixed-GPP biology plus state-dependent molecular forcing",
        )
        return total.values / scale

    solution = solve_ivp(
        tendency,
        (0.0, float(np.max(output_times_year))),
        initial.isotopologue_moles / scale,
        method="DOP853",
        dense_output=True,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=MAXIMUM_SOLVER_STEP_YEARS,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"continuous CO2 trajectory failed: {solution.message}")
    return solution, scale, {
        "start_age_year_bp": start_age,
        "end_age_year_bp": end_age,
        "initial_CO2_ppm": initial_co2,
        "model_data_id": forcing.model_data_id,
        "transfer_convention": forcing.transfer_convention,
        "method": "scipy.solve_ivp DOP853",
        "relative_tolerance": 1.0e-9,
        "absolute_tolerance_scaled_inventory": 1.0e-11,
        "maximum_step_years": MAXIMUM_SOLVER_STEP_YEARS,
        "function_evaluations": int(solution.nfev),
        "success": bool(solution.success),
        "message": solution.message,
    }


def _best_response_lag(
    transient: np.ndarray,
    equilibrium: np.ndarray,
    *,
    time_step_years: float,
    guard_years: float = INITIAL_CONDITION_GUARD_YEARS,
    maximum_lag_years: float = 20_000.0,
) -> dict[str, float]:
    guard_steps = int(np.ceil(guard_years / time_step_years))
    maximum_steps = int(np.floor(maximum_lag_years / time_step_years))
    best_lag = 0
    best_correlation = -np.inf
    for lag_steps in range(maximum_steps + 1):
        start = guard_steps + lag_steps
        if start >= len(transient) - 3:
            break
        correlation = float(
            np.corrcoef(
                transient[start:],
                equilibrium[start - lag_steps : len(equilibrium) - lag_steps],
            )[0, 1]
        )
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag_steps
    return {
        "best_lag_years": float(best_lag * time_step_years),
        "pearson_r_at_best_lag": best_correlation,
        "zero_lag_pearson_r": float(
            np.corrcoef(transient[guard_steps:], equilibrium[guard_steps:])[0, 1]
        ),
    }


def _segmented_line(axis, age, values, *, color, label, linewidth) -> None:
    starts = np.r_[0, np.flatnonzero(np.diff(age) > 10.0) + 1]
    ends = np.r_[starts[1:], len(age)]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        axis.plot(
            age[start:end],
            values[start:end],
            color=color,
            linewidth=linewidth,
            label=label if index == 0 else None,
        )


def _plot(
    source_age_ka: np.ndarray,
    source_co2: np.ndarray,
    sample_age_ka: np.ndarray,
    sample_co2: np.ndarray,
    observed_21: np.ndarray,
    steady_21: np.ndarray,
    transient_21: np.ndarray,
    metrics: dict[str, object],
    output: Path,
) -> None:
    steady_aligned = steady_21 + np.mean(observed_21 - steady_21)
    transient_aligned = transient_21 + np.mean(observed_21 - transient_21)
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 10.3), constrained_layout=True)

    axes[0].plot(source_age_ka, source_co2, color="0.25", linewidth=1.0)
    axes[0].scatter(sample_age_ka, sample_co2, color="#007f78", s=13, zorder=2)
    axes[0].set(ylabel="Atmospheric CO$_2$ (ppm)")
    axes[0].grid(alpha=0.18)

    _segmented_line(
        axes[1], sample_age_ka, observed_21, color="#202020", label="Observed", linewidth=1.6
    )
    _segmented_line(
        axes[1], sample_age_ka, steady_aligned, color="#bf5b17", label="Instantaneous steady", linewidth=1.4
    )
    _segmented_line(
        axes[1], sample_age_ka, transient_aligned, color="#007f78", label="Continuous transient", linewidth=1.7
    )
    axes[1].set(ylabel=r"Modern-air-relative $\Delta^{17}$O (ppm)")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(alpha=0.18)

    windows = [f"{width:g}_ka" for width in SMOOTHING_WINDOWS_KA]
    positions = np.arange(len(windows), dtype=float)
    width = 0.34
    steady_skill = [
        metrics[key]["steady"]["blocked_cross_validation"]["variance_skill"]
        for key in windows
    ]
    transient_skill = [
        metrics[key]["transient"]["blocked_cross_validation"]["variance_skill"]
        for key in windows
    ]
    axes[2].bar(
        positions - width / 2,
        steady_skill,
        width,
        color="#bf5b17",
        label="Instantaneous steady",
    )
    axes[2].bar(
        positions + width / 2,
        transient_skill,
        width,
        color="#007f78",
        label="Continuous transient",
    )
    axes[2].axhline(0.0, color="0.25", linewidth=0.9)
    axes[2].set(
        xticks=positions,
        xticklabels=["Raw", "11 kyr", "21 kyr", "31 kyr"],
        ylabel="Age-block predictive skill",
        xlabel="Isotope smoothing window",
    )
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.18)
    for axis in axes[:2]:
        axis.set_xlim(float(np.max(sample_age_ka)), float(np.min(sample_age_ka)))
        axis.set_xlabel("Gas age (ka BP)")
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    rows = _load_rows(INPUT_PATH)
    sample_age_ka = np.asarray([row["gas_age_ka_bp"] for row in rows], dtype=float)
    sample_age_year = 1000.0 * sample_age_ka
    sample_co2 = np.asarray([row["co2_ppm"] for row in rows], dtype=float)
    observed = np.asarray(
        [row["cap_delta17_modern_air_relative_ppm"] for row in rows], dtype=float
    )

    co2_rows = load_bereiter_co2()
    source_age = np.asarray([row.age_year_bp for row in co2_rows], dtype=float)
    source_co2 = np.asarray([row.co2_ppm for row in co2_rows], dtype=float)
    start_age = float(source_age[-1])
    sample_forward_time = start_age - sample_age_year
    if np.any(sample_forward_time < 0.0):
        raise ValueError("Yang sample predates the continuous CO2 forcing")

    regular_time = np.arange(
        0.0,
        float(np.max(sample_forward_time)) + REGULAR_OUTPUT_STEP_YEARS,
        REGULAR_OUTPUT_STEP_YEARS,
    )
    solution, inventory_scale, solver = _integrate_continuous_co2(
        source_age, source_co2, regular_time
    )
    sample_states = _states_from_solution(
        solution, sample_forward_time, inventory_scale
    )
    regular_states = _states_from_solution(solution, regular_time, inventory_scale)

    reference_state = run_updated_central_state(
        UpdatedForwardInput(
            FIXED_PO2_PAL,
            REFERENCE_CO2_PPM,
            FIXED_GPP_PGC_PER_YEAR,
        )
    )
    reference = PrimeIsotopeComposition(
        reference_state.delta18_prime_permil,
        reference_state.cap_delta17_prime_permil,
        MODEL_LAMBDA,
    )
    transient_prediction = _relative_prediction_ppm(sample_states, reference)
    regular_transient = _relative_prediction_ppm(regular_states, reference)
    steady_prediction, _surface = _model_co2_only_prediction(sample_co2)
    regular_age = start_age - regular_time
    regular_co2 = np.asarray(
        _co2_at_age(regular_age, source_age, source_co2), dtype=float
    )
    regular_steady, _surface = _model_co2_only_prediction(regular_co2)
    lag = _best_response_lag(
        regular_transient,
        regular_steady,
        time_step_years=REGULAR_OUTPUT_STEP_YEARS,
    )

    metrics: dict[str, object] = {}
    for width in SMOOTHING_WINDOWS_KA:
        key = f"{width:g}_ka"
        observed_smooth = _moving_window_mean(sample_age_ka, observed, width)
        steady_smooth = _moving_window_mean(sample_age_ka, steady_prediction, width)
        transient_smooth = _moving_window_mean(
            sample_age_ka, transient_prediction, width
        )
        metrics[key] = {
            "steady": {
                **_tracking_metrics(observed_smooth, steady_smooth),
                "blocked_cross_validation": _blocked_cross_validation(
                    observed_smooth, steady_smooth, block_count=BLOCK_COUNT
                ),
            },
            "transient": {
                **_tracking_metrics(observed_smooth, transient_smooth),
                "blocked_cross_validation": _blocked_cross_validation(
                    observed_smooth, transient_smooth, block_count=BLOCK_COUNT
                ),
            },
        }

    mature = sample_forward_time >= INITIAL_CONDITION_GUARD_YEARS
    mature_metrics = {
        "row_count": int(np.sum(mature)),
        "steady": _tracking_metrics(observed[mature], steady_prediction[mature]),
        "transient": _tracking_metrics(
            observed[mature], transient_prediction[mature]
        ),
    }

    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    steady_aligned = steady_prediction + np.mean(observed - steady_prediction)
    transient_aligned = transient_prediction + np.mean(
        observed - transient_prediction
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "gas_age_ka_bp",
                "measured_co2_ppm",
                "observed_Delta17_ppm",
                "steady_CO2_only_Delta17_ppm",
                "transient_CO2_only_Delta17_ppm",
                "aligned_steady_Delta17_ppm",
                "aligned_transient_Delta17_ppm",
                "years_since_initial_condition",
                "initial_condition_guard_passed",
            ),
        )
        writer.writeheader()
        for values in zip(
            sample_age_ka,
            sample_co2,
            observed,
            steady_prediction,
            transient_prediction,
            steady_aligned,
            transient_aligned,
            sample_forward_time,
            mature,
            strict=True,
        ):
            writer.writerow(dict(zip(writer.fieldnames, values, strict=True)))

    observed_21 = _moving_window_mean(sample_age_ka, observed, 21.0)
    steady_21 = _moving_window_mean(sample_age_ka, steady_prediction, 21.0)
    transient_21 = _moving_window_mean(
        sample_age_ka, transient_prediction, 21.0
    )
    _plot(
        source_age / 1000.0,
        source_co2,
        sample_age_ka,
        sample_co2,
        observed_21,
        steady_21,
        transient_21,
        metrics,
        figure_path,
    )

    report = {
        "audit": "Yang et al. continuous-CO2 transient tracking test",
        "status": "complete",
        "role": (
            "Forward validation at fixed GPP and pO2. The continuous measured "
            "CO2 forcing is independent; no isotope amplitude or response time "
            "is fitted."
        ),
        "fixed_model_conditions": {
            "pO2_PAL": FIXED_PO2_PAL,
            "GPP_PgC_per_year": FIXED_GPP_PGC_PER_YEAR,
            "initial_condition": "steady state at oldest Bereiter CO2 value",
            "initial_condition_guard_years": INITIAL_CONDITION_GUARD_YEARS,
        },
        "forcing": {
            "source": "Bereiter et al. (2015) corrected Antarctic composite",
            "source_rows": len(co2_rows),
            "age_year_bp": [float(source_age[0]), float(source_age[-1])],
            "CO2_ppm": [float(np.min(source_co2)), float(np.max(source_co2))],
        },
        "solver": solver,
        "emergent_response_lag": lag,
        "tracking_by_smoothing_window": metrics,
        "initial_condition_guarded_raw_metrics": mature_metrics,
        "interpretation": {
            "comparison": (
                "Transient and instantaneous predictions use identical physical "
                "forcing and differ only by slow global-O2 memory."
            ),
            "scope": (
                "The test isolates CO2 tracking. Residuals are not uniquely GPP "
                "because measurement noise and other process changes remain."
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
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = run(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "solver": report["solver"],
                "emergent_response_lag": report["emergent_response_lag"],
                "tracking_by_smoothing_window": report[
                    "tracking_by_smoothing_window"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
