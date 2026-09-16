"""Test the updated model against the Termination V CO2-Delta17 decoupling.

Brandon et al. (2020) identify 430--415 ka as a sustained interval in which
CO2 and atmospheric-O2 Delta17 no longer follow the relation seen during the
four younger terminations. This audit fixes that interval before evaluating
the model. A continuous measured CO2 trajectory is first propagated at fixed
GPP. A second run applies one constant GPP multiplier only inside the published
interval. Its amplitude is inferred from the isotope residual; neither event
timing nor the atmospheric response time is fitted.
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

from biological_o2_ensemble import (  # noqa: E402
    fixed_po2_partitioned_biological_budget,
)
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
from prepare_ice_core_holdout import (  # noqa: E402
    interpolate_co2,
    load_bereiter_co2,
)
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    build_updated_central_forcing,
    run_updated_central_state,
)
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL  # noqa: E402


INPUT_PATH = (
    ROOT / "model_data" / "literature" / "brandon_2020_termination_v_delta17.csv"
)
OUTPUT_PATH = ROOT / "outputs" / "brandon_2020_termination_v_audit.json"
FIXED_PO2_PAL = 1.0
FIXED_GPP_PGC_PER_YEAR = 290.0
REFERENCE_CO2_PPM = 280.0
RELATIVE_LAMBDA = 0.516
EVENT_OLDER_BOUND_KA = 430.0
EVENT_YOUNGER_BOUND_KA = 415.0
PRE_EVENT_CONTROL_YOUNGER_BOUND_KA = EVENT_OLDER_BOUND_KA
INTEGRATION_START_KA = 475.0
ANALYTICAL_PRECISION_PPM = 6.0
LINEAR_RESPONSE_TEST_FACTOR = 1.2
MAXIMUM_SOLVER_STEP_YEARS = 200.0


def _load_observations(path: Path = INPUT_PATH) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    required = {
        "depth_ice_m",
        "gas_age_ka_bp",
        "cap_delta17_modern_air_relative_ppm",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Unexpected Brandon columns in {path}")
    rows.sort(key=lambda row: row["gas_age_ka_bp"])
    if len(rows) != 50:
        raise ValueError(f"Expected 50 Termination V rows, found {len(rows)}")
    return rows


def _composition(state: GlobalO2Reservoir) -> PrimeIsotopeComposition:
    return PrimeIsotopeComposition(
        delta18_prime_permil=state.delta18_prime_permil,
        cap_delta17_prime_permil=state.cap_delta17_prime_permil,
        lambda_ref=MODEL_LAMBDA,
    )


def _reference_composition() -> PrimeIsotopeComposition:
    state = run_updated_central_state(
        UpdatedForwardInput(FIXED_PO2_PAL, REFERENCE_CO2_PPM, FIXED_GPP_PGC_PER_YEAR)
    )
    return PrimeIsotopeComposition(
        state.delta18_prime_permil,
        state.cap_delta17_prime_permil,
        MODEL_LAMBDA,
    )


def _integrate_trajectory(
    source_age_year_bp: np.ndarray,
    source_co2_ppm: np.ndarray,
    sample_age_year_bp: np.ndarray,
    *,
    event_gpp_factor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not np.isfinite(event_gpp_factor) or event_gpp_factor <= 0.0:
        raise ValueError("event GPP factor must be finite and positive")
    start_age_year = INTEGRATION_START_KA * 1000.0
    if start_age_year > float(source_age_year_bp[-1]):
        raise ValueError("CO2 forcing does not reach the requested initial age")
    initial_co2 = float(np.interp(start_age_year, source_age_year_bp, source_co2_ppm))
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
        source="steady updated-model state at 475 ka",
    )
    forcing = build_updated_central_forcing(request)
    scale = initial.isotopologue_moles
    maximum_time = start_age_year - float(np.min(sample_age_year_bp))

    def tendency(time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        age_year = start_age_year - time
        age_ka = age_year / 1000.0
        pco2 = float(np.interp(age_year, source_age_year_bp, source_co2_ppm))
        in_event = EVENT_YOUNGER_BOUND_KA <= age_ka <= EVENT_OLDER_BOUND_KA
        gpp = FIXED_GPP_PGC_PER_YEAR * (event_gpp_factor if in_event else 1.0)
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * scale),
            source="Termination V continuous-forcing integration state",
        )
        native_photo = forcing.surface.evaluate_prime_tendency_at(
            np.asarray(
                [reservoir.delta18_prime_permil, reservoir.cap_delta17_prime_permil]
            ),
            po2_pal=FIXED_PO2_PAL,
            pco2_ppm=pco2,
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        photo = IsotopologueTendency(
            *map(float, forcing.forcing_scale * native_photo.values),
            source="updated molecular response under Termination V CO2 forcing",
        )
        biology = fixed_po2_partitioned_biological_budget(
            forcing.biological_member,
            po2_pal=FIXED_PO2_PAL,
            gpp_pgC_per_year=gpp,
            photochemical=photo,
        )
        total = sum_tendencies(
            (biology.tendency(reservoir), photo),
            source="time-dependent GPP biology plus state-dependent molecular forcing",
        )
        return total.values / scale

    solution = solve_ivp(
        tendency,
        (0.0, maximum_time),
        initial.isotopologue_moles / scale,
        method="DOP853",
        dense_output=True,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=MAXIMUM_SOLVER_STEP_YEARS,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"Termination V integration failed: {solution.message}")
    sample_time = start_age_year - sample_age_year_bp
    values = solution.sol(sample_time)
    states = tuple(
        GlobalO2Reservoir(
            *map(float, values[:, index] * scale),
            source=f"Termination V state at {age / 1000.0:g} ka BP",
        )
        for index, age in enumerate(sample_age_year_bp)
    )
    reference = _reference_composition()
    prediction = np.asarray(
        [
            relative_cap_delta17_ppm(
                _composition(state), reference, relative_lambda=RELATIVE_LAMBDA
            )
            for state in states
        ],
        dtype=float,
    )
    return prediction, {
        "method": "scipy.solve_ivp DOP853",
        "relative_tolerance": 1.0e-9,
        "absolute_tolerance_scaled_inventory": 1.0e-11,
        "maximum_step_years": MAXIMUM_SOLVER_STEP_YEARS,
        "function_evaluations": int(solution.nfev),
        "success": bool(solution.success),
    }


def _metrics(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    residual = observed[mask] - predicted[mask]
    return {
        "rows": int(np.sum(mask)),
        "mean_residual_ppm": float(np.mean(residual)),
        "rmse_ppm": float(np.sqrt(np.mean(residual**2))),
        "mean_absolute_residual_ppm": float(np.mean(np.abs(residual))),
        "residual_standard_deviation_ppm": float(np.std(residual, ddof=1)),
    }


def _infer_event_gpp_factor(
    observed: np.ndarray,
    fixed_prediction: np.ndarray,
    response_per_factor: np.ndarray,
    event_mask: np.ndarray,
) -> float:
    response = response_per_factor[event_mask]
    residual = observed[event_mask] - fixed_prediction[event_mask]
    denominator = float(np.dot(response, response))
    if denominator <= 0.0:
        raise ValueError("event GPP response must be non-zero")
    return 1.0 + float(np.dot(response, residual) / denominator)


def _factor_uncertainty(
    response_per_factor: np.ndarray,
    event_residual: np.ndarray,
    pre_event_residual: np.ndarray,
    *,
    analytical_sigma_ppm: float,
) -> dict[str, float | list[float]]:
    """Propagate event noise and the independently trained reference offset."""

    response = np.asarray(response_per_factor, dtype=float)
    denominator = float(np.dot(response, response))
    offset_coefficient = float(np.sum(response) / denominator)
    analytical_variance = analytical_sigma_ppm**2 / denominator + (
        offset_coefficient**2 * analytical_sigma_ppm**2 / len(pre_event_residual)
    )
    empirical_variance = float(np.var(event_residual, ddof=1)) / denominator + (
        offset_coefficient**2
        * float(np.var(pre_event_residual, ddof=1))
        / len(pre_event_residual)
    )
    return {
        "analytical_only_standard_error_factor": float(np.sqrt(analytical_variance)),
        "empirical_residual_standard_error_factor": float(np.sqrt(empirical_variance)),
    }


def _plot(
    age_ka: np.ndarray,
    co2_ppm: np.ndarray,
    observed: np.ndarray,
    fixed: np.ndarray,
    pulse: np.ndarray,
    inferred_factor: float,
    output: Path,
) -> None:
    event = (age_ka >= EVENT_YOUNGER_BOUND_KA) & (age_ka <= EVENT_OLDER_BOUND_KA)
    figure, axes = plt.subplots(4, 1, figsize=(9.0, 11.0), sharex=True, constrained_layout=True)
    axes[0].plot(age_ka, co2_ppm, color="#2f4b7c", marker="o", markersize=3)
    axes[0].set_ylabel("Atmospheric CO$_2$ (ppm)")

    axes[1].scatter(age_ka, observed, color="0.15", s=22, label="Brandon et al. (2020)", zorder=3)
    axes[1].plot(age_ka, fixed, color="#bf5b17", linewidth=1.8, label="Fixed GPP")
    axes[1].plot(age_ka, pulse, color="#007f78", linewidth=2.0, label="Inferred GPP pulse")
    axes[1].set_ylabel(r"Modern-air-relative $\Delta^{17}$O (ppm)")
    axes[1].legend(frameon=False, ncol=3)

    axes[2].axhspan(-ANALYTICAL_PRECISION_PPM, ANALYTICAL_PRECISION_PPM, color="0.88")
    axes[2].axhline(0.0, color="0.25", linewidth=0.8)
    axes[2].plot(age_ka, observed - fixed, color="#bf5b17", marker="o", markersize=3, label="Fixed GPP")
    axes[2].plot(age_ka, observed - pulse, color="#007f78", marker="o", markersize=3, label="GPP pulse")
    axes[2].set_ylabel("Observation - model (ppm)")
    axes[2].legend(frameon=False, ncol=2)

    gpp_percent = np.full(age_ka.shape, 100.0)
    gpp_percent[event] = 100.0 * inferred_factor
    axes[3].step(age_ka, gpp_percent, where="mid", color="#007f78", linewidth=2.0)
    axes[3].axhspan(110.0, 130.0, color="#007f78", alpha=0.15, label="Brandon reported range")
    axes[3].set(ylabel="GPP (% pre-industrial)", xlabel="Gas age (ka BP)")
    axes[3].legend(frameon=False)

    for axis in axes:
        axis.axvspan(EVENT_YOUNGER_BOUND_KA, EVENT_OLDER_BOUND_KA, color="#f0c36e", alpha=0.14)
        axis.grid(alpha=0.18)
        axis.set_xlim(float(np.max(age_ka)), float(np.min(age_ka)))
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    rows = _load_observations()
    age_ka = np.asarray([row["gas_age_ka_bp"] for row in rows], dtype=float)
    age_year = 1000.0 * age_ka
    observed = np.asarray(
        [row["cap_delta17_modern_air_relative_ppm"] for row in rows], dtype=float
    )
    co2_rows = load_bereiter_co2()
    co2 = np.asarray([interpolate_co2(age, co2_rows)["co2_ppm"] for age in age_year])
    source_age = np.asarray([row.age_year_bp for row in co2_rows], dtype=float)
    source_co2 = np.asarray([row.co2_ppm for row in co2_rows], dtype=float)

    fixed_raw, fixed_solver = _integrate_trajectory(
        source_age, source_co2, age_year, event_gpp_factor=1.0
    )
    response_raw, response_solver = _integrate_trajectory(
        source_age,
        source_co2,
        age_year,
        event_gpp_factor=LINEAR_RESPONSE_TEST_FACTOR,
    )

    event = (age_ka >= EVENT_YOUNGER_BOUND_KA) & (age_ka <= EVENT_OLDER_BOUND_KA)
    pre_event = age_ka > PRE_EVENT_CONTROL_YOUNGER_BOUND_KA
    post_event = age_ka < EVENT_YOUNGER_BOUND_KA
    offset = float(np.mean(observed[pre_event] - fixed_raw[pre_event]))
    fixed = fixed_raw + offset
    response_per_factor = (response_raw - fixed_raw) / (LINEAR_RESPONSE_TEST_FACTOR - 1.0)
    inferred_factor = _infer_event_gpp_factor(
        observed, fixed, response_per_factor, event
    )
    factor_delta = inferred_factor - 1.0
    if not 0.5 <= inferred_factor <= 1.5:
        raise RuntimeError(
            f"Linearized inferred GPP factor {inferred_factor:.4g} is outside the declared audit domain"
        )
    pulse_raw, pulse_solver = _integrate_trajectory(
        source_age, source_co2, age_year, event_gpp_factor=inferred_factor
    )
    pulse = pulse_raw + offset
    linearized_pulse = fixed + factor_delta * response_per_factor
    linearization_error = pulse - linearized_pulse
    factor_errors = _factor_uncertainty(
        response_per_factor[event],
        observed[event] - pulse[event],
        observed[pre_event] - fixed[pre_event],
        analytical_sigma_ppm=ANALYTICAL_PRECISION_PPM,
    )
    analytical_se = float(factor_errors["analytical_only_standard_error_factor"])
    empirical_se = float(factor_errors["empirical_residual_standard_error_factor"])
    reference_window_sensitivity = {}
    for younger_bound in (430.0, 433.0, 435.0):
        reference_mask = age_ka > younger_bound
        reference_offset = float(
            np.mean(observed[reference_mask] - fixed_raw[reference_mask])
        )
        reference_fixed = fixed_raw + reference_offset
        reference_window_sensitivity[f"older_than_{younger_bound:g}_ka"] = {
            "rows": int(np.sum(reference_mask)),
            "offset_ppm": reference_offset,
            "linearized_inferred_GPP_percent": 100.0
            * _infer_event_gpp_factor(
                observed, reference_fixed, response_per_factor, event
            ),
        }

    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "gas_age_ka_bp",
                "measured_co2_ppm",
                "observed_Delta17_ppm",
                "fixed_GPP_Delta17_ppm",
                "pulse_GPP_Delta17_ppm",
                "fixed_residual_ppm",
                "pulse_residual_ppm",
                "published_event_window",
            )
        )
        writer.writerows(
            zip(
                age_ka,
                co2,
                observed,
                fixed,
                pulse,
                observed - fixed,
                observed - pulse,
                event,
                strict=True,
            )
        )

    _plot(age_ka, co2, observed, fixed, pulse, inferred_factor, figure_path)
    fixed_event = _metrics(observed, fixed, event)
    pulse_event = _metrics(observed, pulse, event)
    report = {
        "audit": "Brandon et al. (2020) Termination V CO2-Delta17 decoupling",
        "status": "complete",
        "observations": {
            "paper_doi": "10.1038/s41467-020-15739-2",
            "dataset_doi": "10.1594/PANGAEA.914283",
            "rows": len(rows),
            "coordinate": "modern-air-relative logarithmic Delta-17O, lambda=0.516",
            "reported_pooled_analytical_precision_ppm": ANALYTICAL_PRECISION_PPM,
        },
        "predeclared_design": {
            "published_decoupling_window_ka_bp": [EVENT_YOUNGER_BOUND_KA, EVENT_OLDER_BOUND_KA],
            "offset_training_window_ka_bp": [
                f">{PRE_EVENT_CONTROL_YOUNGER_BOUND_KA:g}",
                float(np.max(age_ka)),
            ],
            "fixed_pO2_PAL": FIXED_PO2_PAL,
            "baseline_GPP_PgC_per_year": FIXED_GPP_PGC_PER_YEAR,
            "fitted_quantities": ["one event-window GPP multiplier"],
            "not_fitted": ["event timing", "response time", "CO2 forcing", "isotope response amplitude"],
        },
        "co2_forcing": {
            "source": "Bereiter et al. (2015) corrected Antarctic composite",
            "dataset_doi": "10.25921/n8y4-bp27",
            "sample_range_ppm": [float(np.min(co2)), float(np.max(co2))],
        },
        "model_runs": {
            "fixed_GPP": fixed_solver,
            "linear_response_factor": LINEAR_RESPONSE_TEST_FACTOR,
            "linear_response": response_solver,
            "inferred_pulse": pulse_solver,
            "pre_event_reference_offset_ppm": offset,
            "maximum_linearization_error_ppm": float(np.max(np.abs(linearization_error))),
            "reference_window_sensitivity": reference_window_sensitivity,
        },
        "fixed_GPP_test": {
            "pre_event_control": _metrics(observed, fixed, pre_event),
            "published_event_window": fixed_event,
            "post_event": _metrics(observed, fixed, post_event),
        },
        "GPP_pulse_test": {
            "inferred_factor_of_preindustrial": inferred_factor,
            "inferred_percent_of_preindustrial": 100.0 * inferred_factor,
            "analytical_only_95_percent_interval_percent": [
                100.0 * (inferred_factor - 1.96 * analytical_se),
                100.0 * (inferred_factor + 1.96 * analytical_se),
            ],
            "empirical_residual_95_percent_interval_percent": [
                100.0 * (inferred_factor - 1.96 * empirical_se),
                100.0 * (inferred_factor + 1.96 * empirical_se),
            ],
            "Brandon_reported_range_percent": [110.0, 130.0],
            "Brandon_reported_average_percent": 117.0,
            "within_reported_range": bool(1.10 <= inferred_factor <= 1.30),
            "published_event_window": pulse_event,
            "event_SSE_reduction_fraction": float(
                1.0 - pulse_event["rmse_ppm"] ** 2 / fixed_event["rmse_ppm"] ** 2
            ),
            "post_event": _metrics(observed, pulse, post_event),
        },
        "interpretation": {
            "falsification_logic": (
                "A coherent event residual in the fixed-GPP run is required before GPP is varied. "
                "Agreement of the one-parameter inferred amplitude with Brandon's independently "
                "reported 10-30% range is an external concordance check, not a fit target."
            ),
            "limits": (
                "The 6 ppm value is analytical precision before corrections, not total proxy-model "
                "uncertainty. The inferred pulse is an interval-average diagnostic and does not "
                "uniquely exclude ozone or biosphere-isotope changes."
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
                "fixed_GPP_test": report["fixed_GPP_test"],
                "GPP_pulse_test": report["GPP_pulse_test"],
                "model_runs": report["model_runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
