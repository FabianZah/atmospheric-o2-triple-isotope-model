"""Check a compact sulfate transfer approximation against published examples.

Run with ``python validation/audit_sulfate_transfer.py``. This source-arithmetic
audit requires neither the original PDFs nor a photochemical model run. Every
calculation assumes preservation of the supplied primary sulfate signal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from math import exp, expm1, isfinite, log, log1p
from pathlib import Path
import sys
from time import perf_counter

ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
sys.path.insert(0, str(ROOT / "code"))

from isotopes import R17_VSMOW, R18_VSMOW, cap_delta17  # noqa: E402
from sulfate_to_air import (  # noqa: E402
    AnomalyCoordinate, ConditionalSulfateTransfer, SulfateProcessAssumptions,
    exact_air_from_sulfate,
)


SOURCES = {
    "bao_2008": {
        "citation": "Bao, Lyons and Zhou (2008), Nature 453, 504-506",
        "doi": "10.1038/nature06959",
        "locator": "Main p. 505; SI sections 2-5, pp. 7-11; Table S2, pp. 12-13",
        "coordinate": "logarithmic Delta-prime-17O, reference slope 0.52",
        "bias_definition": "reported partial-yield anomaly minus bulk sulfate anomaly",
        "reported_bias_permil": -0.05,
        "bias_scope": "Authors' provisional correction for their partial-yield analyses only",
        "fraction_scope": "Provisional marine-sulfate signal fraction; Fig. 3 uses 5%, 10%, 20%",
        "air_coordinate_caveat": "The study discusses a photochemical anomaly component; these historical outputs are not automatically full OXYTIB air Delta-prime-17O_0.528",
    },
    "cao_bao_2013": {
        "citation": "Cao and Bao (2013), PNAS 110, 14546-14550",
        "doi": "10.1073/pnas.1302972110",
        "locator": "Main p. 14548, Svalbard example; p. 14549, isotope definition",
        "coordinate": "classical linear Delta-17O = delta17O - 0.52 delta18O",
        "sulfate_input_scope": "Inferred oxidative-weathering endmember, approximately -4.2 per mil, from the Delta-17O versus delta34S relationship; not a raw bulk measurement",
        "endmember_primary_reference": "Bao, Fairchild, Wynn and Spotl (2009), Science 323, 119-122, cited as reference 6",
        "fraction_scope": "Assumed 10-25% atmospheric oxygen in the inferred weathering endmember",
        "harmonization_limit": "The scalar atmospheric interval supplies no paired air delta18O for conversion to logarithmic 0.528 notation",
    },
    "cao_bao_2021": {
        "citation": "Cao and Bao (2021), Small Triple Oxygen Isotope Variations in Sulfate: Mechanisms and Applications",
        "locator": "Printed p. 466, Eq. 6 and following paragraph",
        "equation": "1000 ln(alpha18) = 2.68e6 / T_K**2 - 7.45",
        "temperature_domain_c": [0.0, 150.0],
        "theta": 0.524,
        "theta_status": "Assumed by the authors; not an experimentally calibrated sulfate-water exponent",
    },
    "waldeck_2022": {
        "citation": "Waldeck et al. (2022), EPSL 578, 117320",
        "doi": "10.1016/j.epsl.2021.117320",
        "locator": "p. 5, Fig. 3 and text; p. 6, inferred resetting fractions",
        "coordinate": "logarithmic Delta-prime-17O, reference slope 0.5305",
        "rounded_equilibrium_delta18_permil": 23.0,
        "rounded_equilibrium_anomaly_permil": -0.148,
        "temperature_c": 25.0,
        "scope": "Water-equilibrated sulfate endmember; not a universal primary-weathering background",
        "resetting_examples": {"Sorbas": 0.20, "Caltanissetta": 0.30},
    },
    "waldeck_2025": {
        "citation": "Waldeck et al. (2025), Nature Communications 16, 2087",
        "doi": "10.1038/s41467-025-57282-y",
        "locator": "p. 3, Fig. 2 caption and discussion; p. 4, isotope nomenclature",
        "coordinate": "logarithmic Delta-prime-17O, reference slope 0.5305",
        "scope": "Modern air is used for the illustrative mixing vector; more negative ancient air is explicitly considered. Atmospheric inheritance can change with weathering regime.",
    },
}


def _finite(*values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError("all isotope values and fractions must be finite")


def first_order_air_anomaly(
    sulfate_permil: float,
    fraction: float,
    *,
    nonair_endmember_permil: float,
    reported_bias_permil: float,
) -> float:
    """Invert S_reported - bias = f A + (1-f) B in one supplied coordinate.

    This is an explicitly approximate transfer, not isotope-atom mass balance.
    ``B`` describes the non-air contribution after its formation effects. A
    separate air-path fractionation effect would need independent treatment.
    """
    _finite(sulfate_permil, fraction, nonair_endmember_permil, reported_bias_permil)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("atmospheric fraction must be greater than zero and at most one")
    corrected = sulfate_permil - reported_bias_permil
    return (corrected - (1.0 - fraction) * nonair_endmember_permil) / fraction


def air_interval(
    sulfate_bounds: tuple[float, float],
    fraction_bounds: tuple[float, float],
    *,
    nonair_bounds: tuple[float, float],
    reported_bias_permil: float,
) -> tuple[float, float]:
    """Return endpoint extrema over independent bounded inputs, not a posterior."""
    for bounds in (sulfate_bounds, fraction_bounds, nonair_bounds):
        _finite(*bounds)
        if bounds[0] > bounds[1]:
            raise ValueError("bounds must be ordered from low to high")
    results = [
        first_order_air_anomaly(sulfate, fraction, nonair_endmember_permil=background,
                               reported_bias_permil=reported_bias_permil)
        for sulfate, fraction, background in product(sulfate_bounds, fraction_bounds, nonair_bounds)
    ]
    return min(results), max(results)


def convert_anomaly(
    anomaly_permil: float,
    delta18_permil: float,
    *,
    source_slope: float,
    source_form: str,
    target_slope: float = 0.528,
) -> float:
    """Convert notation on the same isotope standard using paired delta18O."""
    _finite(anomaly_permil, delta18_permil, source_slope, target_slope)
    if delta18_permil <= -1000.0:
        raise ValueError("delta18O must exceed -1000 per mil")
    if source_form == "logarithmic":
        return anomaly_permil + (source_slope - target_slope) * 1000.0 * log1p(delta18_permil / 1000.0)
    if source_form == "linear":
        delta17 = anomaly_permil + source_slope * delta18_permil
        if delta17 <= -1000.0:
            raise ValueError("derived delta17O must exceed -1000 per mil")
        return cap_delta17(delta17, delta18_permil, slope=target_slope)
    raise ValueError("source_form must be logarithmic or linear")


def equilibrium_sulfate(temperature_c: float, *, slope: float) -> tuple[float, float]:
    """Cao/Bao Eq. 6, adopted theta=0.524, and VSMOW-like water for the audit."""
    _finite(temperature_c, slope)
    if not 0.0 <= temperature_c <= 150.0:
        raise ValueError("temperature exceeds the cited 0-150 C equation domain")
    alpha_prime18 = 2.68e6 / (temperature_c + 273.15) ** 2 - 7.45
    return 1000.0 * expm1(alpha_prime18 / 1000.0), (0.524 - slope) * alpha_prime18


def _synthetic_atom_mixture(air_anomaly: float, fraction: float, *, air_delta18: float,
                            background_delta18: float, background_anomaly: float) -> tuple[float, float]:
    """Independent exact-atom reference, with explicitly zero pathway fractionation.

    This is a mathematical test case, not a primary-sulfate formation model.
    f counts total oxygen atoms, unlike a 16O-weighted ratio mixing fraction.
    """
    def atoms(anomaly, delta18):
        r18 = R18_VSMOW * (1.0 + delta18 / 1000.0)
        r17 = R17_VSMOW * exp(anomaly / 1000.0) * (1.0 + delta18 / 1000.0)**0.528
        total = 1.0 + r17 + r18
        return (1.0 / total, r17 / total, r18 / total)
    air, background = atoms(air_anomaly, air_delta18), atoms(background_anomaly, background_delta18)
    mixed = [fraction * a + (1.0-fraction) * b for a, b in zip(air, background)]
    delta18 = 1000.0 * (mixed[2] / mixed[0] / R18_VSMOW - 1.0)
    prime17 = 1000.0 * log(mixed[1] / mixed[0] / R17_VSMOW)
    return delta18, prime17 - 0.528 * 1000.0 * log1p(delta18 / 1000.0)


def implementation_checks() -> dict:
    native = AnomalyCoordinate("logarithmic", 0.52)
    transfer = ConditionalSulfateTransfer(
        coordinate=native, background_permil=0.0, air_path_shift_permil=0.0,
        reported_bias_permil=0.0, air_scope="full_air",
        assumption_note="Synthetic coordinate test: preserved signal and explicitly zero B/T in log-0.52.",
    )
    air, fraction, air_d18, sulfate_d18 = -10.0, 0.1, 23.9, 15.0
    sulfate = native.to_oxytib(transfer.sulfate_native(native.from_oxytib(air, air_d18), fraction), sulfate_d18)
    recovered = transfer.infer_air(sulfate_cap_delta17_permil=sulfate, sulfate_delta18_permil=sulfate_d18,
                                  fraction=fraction, air_delta18_permil=air_d18)
    rows = []
    process = SulfateProcessAssumptions(
        background_cap_delta17_permil=0.0, alpha18_air_to_sulfate=1.0, theta_air_to_sulfate=0.528,
        assumption_note="Synthetic atom-conservation check; alpha18=1 means no fractionation and makes theta immaterial.",
    )
    for background_d18, air, fraction in product((0.0, 10.0, 20.0), (-0.432, -2.0, -10.0, -20.0, -40.0), (0.1, 0.25)):
        delta18, anomaly = _synthetic_atom_mixture(air, fraction, air_delta18=23.9,
                                                 background_delta18=background_d18, background_anomaly=0.0)
        naive_air = anomaly / fraction
        exact = exact_air_from_sulfate(sulfate_cap_delta17_permil=anomaly, sulfate_delta18_permil=delta18,
                                      fraction=fraction, air_delta18_permil=23.9, process=process)
        rows.append({"true_air_0528_permil": air, "fraction": fraction,
                     "air_delta18_permil": 23.9, "background_delta18_permil": background_d18,
                     "background_0528_permil": 0.0, "exact_sulfate_delta18_permil": delta18,
                     "exact_sulfate_0528_permil": anomaly, "linear_anomaly_air_permil": naive_air,
                     "linear_anomaly_air_error_permil": naive_air-air,
                     "exact_recovered_air_permil": exact["air"]["cap_delta17_permil"],
                     "exact_air_error_permil": exact["air"]["cap_delta17_permil"]-air,
                     "exact_implied_background_delta18_permil": exact["implied_nonair_oxygen"]["delta18_permil"]})
    errors = {str(a): max(abs(r["linear_anomaly_air_error_permil"]) for r in rows
                         if r["true_air_0528_permil"] == a) for a in (-0.432, -2.0, -10.0, -20.0, -40.0)}
    return {
        "coordinate_roundtrip": {"true_air_0528_permil": -10.0, "fraction": 0.1,
                                 "recovered_air_0528_permil": recovered["air_cap_delta17_permil"],
                                 "naive_converted_sulfate_divided_by_fraction_permil": sulfate/0.1,
                                 "scope": "Synthetic full-air coordinate test, not a new calibration of Bao 2008"},
        "finite_anomaly_approximation": {
            "scope": "Synthetic exact oxygen-atom mixtures, preserved and no fractionation on either path; tests the approximation only",
            "inputs_status": "Chosen stress-test scenarios, not measured sulfate background values or an uncertainty distribution",
            "rows": rows, "maximum_absolute_air_error_permil_by_true_air": errors,
            "exact_inverse_maximum_absolute_air_error_permil": max(abs(r["exact_air_error_permil"]) for r in rows),
            "decision": "Log-anomaly dilution must not be treated as exact even if reference conventions are correct. These deviations are neither an OXYTIB model error nor a universal uncertainty correction.",
        },
        "implementation_scope": "Exact conditional forward/inverse isotope-atom transfer, convergence-checked uncertainty integration and local atmospheric inference through console or browser",
        "likelihood": "Exact forward sulfate likelihood integrates incorporation, background and correlated analytical errors; the first-order expression remains an independent comparison",
    }


def build_report() -> dict:
    rows = []
    for label, sulfate, expected in (
        ("Early Cambrian", -0.29, -2.4),
        ("Marinoan barite", -0.70, -6.5),
    ):
        raw_air = first_order_air_anomaly(sulfate, 0.10, nonair_endmember_permil=0.0, reported_bias_permil=0.0)
        air = first_order_air_anomaly(sulfate, 0.10, nonair_endmember_permil=0.0, reported_bias_permil=-0.05)
        rows.append({
            "case": label, "source": "bao_2008", "coordinate": SOURCES["bao_2008"]["coordinate"],
            "reported_sulfate_permil": sulfate, "bulk_corrected_sulfate_permil": sulfate + 0.05,
            "fraction": 0.10, "nonair_endmember_permil": 0.0,
            "air_without_analytical_correction_permil": raw_air,
            "calculated_air_permil": air, "published_air_permil": expected,
            "residual_permil": air - expected,
            "fraction_scenarios": {str(f): first_order_air_anomaly(sulfate, f, nonair_endmember_permil=0.0,
                                       reported_bias_permil=-0.05) for f in (0.05, 0.10, 0.20)},
        })
    svalbard = air_interval((-4.2, -4.2), (0.10, 0.25), nonair_bounds=(0.0, 0.0), reported_bias_permil=0.0)

    # These are table lookups/interpolation in Bao's own photochemical output.
    # They audit the historical calculation and do not evaluate OXYTIB pCO2.
    table_pco2 = {"Early Cambrian": 4263.0,
                  "Marinoan barite": 12057.0 + (6.50 - 6.46) / (7.00 - 6.46) * (13146.0 - 12057.0)}
    d18, eq_native = equilibrium_sulfate(25.0, slope=0.5305)
    _, eq_528 = equilibrium_sulfate(25.0, slope=0.528)
    wrong_air = first_order_air_anomaly(eq_528, 0.10, nonair_endmember_permil=0.0, reported_bias_permil=0.0)
    background_sensitivity = [
        {"fraction": f, "air_shift_for_plus_0_05_background_permil": -(1.0-f) / f * 0.05,
         "air_shift_for_plus_0_05_bulk_correction_permil": 0.05 / f}
        for f in (0.05, 0.10, 0.20, 0.25)
    ]
    return {
        "audit": "Compact sulfate transfer: historical arithmetic and applicability",
        "sources": SOURCES,
        "implementation_checks": implementation_checks(),
        "formula": "A = (S_reported - bias - (1-f)*B)/f; shared isotope coordinate; primary preservation assumed",
        "scope": "Reproduces published conversion arithmetic; does not independently validate preservation or the atmospheric model",
        "published_bao_2008_examples": rows,
        "bao_2008_co2_from_published_table_only": {
            "method": "Exact Table S2 entry at -2.40; linear interpolation between -6.46 and -7.00 for -6.50",
            "calculated_pco2_ppm": table_pco2,
            "main_text_approximate_pco2_ppm": {"Early Cambrian": 4200.0, "Marinoan barite": 12000.0},
        },
        "cao_bao_2013_svalbard": {
            "sulfate_endmember_permil": -4.2, "fraction_bounds": [0.10, 0.25],
            "coordinate": SOURCES["cao_bao_2013"]["coordinate"],
            "calculated_air_bounds_permil": svalbard, "published_rounded_air_bounds_permil": [-42.0, -17.0],
            "maximum_rounding_difference_permil": max(abs(a-b) for a,b in zip(svalbard, (-42.0, -17.0))),
        },
        "water_equilibrium_counterexample": {
            "source": "cao_bao_2021", "comparison_source": "waldeck_2022",
            "temperature_c": 25.0, "theta_assumed": 0.524,
            "water_delta18_permil": 0.0, "water_anomaly_permil": 0.0,
            "calculated_sulfate_delta18_permil": d18,
            "calculated_sulfate_anomaly_05305_permil": eq_native,
            "rounded_waldeck_sulfate_anomaly_05305_permil": -0.148,
            "calculated_sulfate_anomaly_0528_permil": eq_528,
            "apparent_air_if_misclassified_as_primary_10pct_permil": wrong_air,
            "interpretation": "An endmember reset to water equilibrium has no retained constraint on air; its negative anomaly cannot be inverted using an assumed primary atmospheric fraction",
        },
        "notation_check": {
            "bao_2008_observed_strong_anomaly_delta18_range_permil": [14.0, 19.0],
            "log052_to_log0528_anomaly_shifts_permil": [
                convert_anomaly(0.0, d, source_slope=0.52, source_form="logarithmic") for d in (14.0, 19.0)],
            "scope": "Illustrates the reference-slope effect using the printed sample range; does not assign delta18O to either reconstruction endmember",
        },
        "background_sensitivity": background_sensitivity,
        "sensitivity_scope": "0.05 per mil is an illustrative perturbation, not a universal background uncertainty or prior",
        "decision": {
            "compact_transfer": "Supported as a conditional first-order approximation for suitable primary sulfate or an independently inferred weathering endmember",
            "universal_incorporation_fraction": None,
            "universal_background_range": None,
            "unknown_preservation_distribution": None,
            "next_input_contract": ["measured calibrated sulfate isotope composition and analytical uncertainty",
                                    "explicit atmospheric-incorporation constraint with archive-specific justification",
                                    "explicit background treatment in the same reference coordinate",
                                    "primary-preservation assumption or independently reconstructed primary endmember"],
            "before_modern_coordinate_inference": "Check the air-path fractionation/background treatment and paired delta18O; historical output anomalies cannot simply be renamed Delta-prime-17O_0.528",
        },
    }


def inference_checks(grid_sizes=(81, 161, 321, 641, 1281, 2561)) -> dict:
    """Synthetic end-to-end grid convergence, separate from source arithmetic."""
    import numpy as np

    from isotopes import conventional_delta_from_prime
    from public_model_service import forward
    from sulfate_to_air import IncorporationConstraint, exact_sulfate_from_air
    from sulfate_uncertainty import BackgroundConstraint, SulfateLikelihoodInput
    from updated_molecular_forward_model import UpdatedForwardInput
    from updated_output_surface_joint_posterior import UpdatedJointPosteriorInput, joint_updated_posterior

    truth = UpdatedForwardInput(p_o2_pal=0.5, p_co2_ppm=10000.0, gpp_pgC_per_year=72.5)
    atmosphere = forward(truth)["result"]
    air17 = atmosphere["central_cap_delta17_prime_permil"]
    air18 = conventional_delta_from_prime(atmosphere["central_delta18_prime_permil"])
    assumptions = "Synthetic primary-sulfate recovery test; specified B=-0.02, f=0.2 and no air-path fractionation."
    sulfate17 = exact_sulfate_from_air(
        air_cap_delta17_permil=air17, air_delta18_permil=air18, sulfate_delta18_permil=15.0,
        fraction=0.2, process=SulfateProcessAssumptions(-0.02, 1.0, 0.528, assumptions))
    observation = SulfateLikelihoodInput(
        measured_cap_delta17_permil=sulfate17, measured_delta18_permil=15.0,
        cap_delta17_sigma_permil=0.03, delta18_sigma_permil=0.5, isotope_error_correlation=0.0,
        incorporation=IncorporationConstraint("fixed", center=0.2),
        background=BackgroundConstraint("fixed", center=-0.02),
        alpha18_air_to_sulfate=1.0, theta_air_to_sulfate=0.528, assumption_note=assumptions)
    uncertain_fraction = replace(observation, incorporation=IncorporationConstraint("range", lower=0.15, upper=0.25))
    scenarios = {
        "analytical_only": observation,
        "uncertain_incorporation": uncertain_fraction,
        "uncertain_incorporation_and_background": replace(
            uncertain_fraction, background=BackgroundConstraint("normal", center=-0.02, sigma=0.03)),
    }
    rows = []
    for name, obs in scenarios.items():
        estimates = []
        for count in grid_sizes:
            started = perf_counter()
            result = joint_updated_posterior(UpdatedJointPosteriorInput(
                sulfate=obs, free_coordinates=("pCO2",), p_o2_pal=truth.p_o2_pal,
                gpp_pgC_per_year=truth.gpp_pgC_per_year, pco2_bounds_ppm=(3000.0, 60000.0),
                pco2_prior="uniform", pco2_grid_size=count))
            estimates.append({
                "grid_size": count, "seconds": perf_counter()-started,
                "median_pco2_ppm": result.posterior_median["pCO2"],
                "credible_interval_95_ppm": result.equal_tailed_credible_intervals["pCO2"],
                "likelihood_diagnostics": result.sulfate_likelihood_diagnostics,
            })
            if len(estimates) >= 3:
                last, previous = estimates[-1], estimates[-2]
                interval = last["credible_interval_95_ppm"]
                change = max(abs(last["median_pco2_ppm"]-previous["median_pco2_ppm"]),
                             float(np.max(np.abs(np.asarray(interval)-previous["credible_interval_95_ppm"]))))
                if change/(interval[1]-interval[0]) < 0.01:
                    break
        last, previous = estimates[-1], estimates[-2]
        interval = last["credible_interval_95_ppm"]
        change = max(abs(last["median_pco2_ppm"]-previous["median_pco2_ppm"]),
                     float(np.max(np.abs(np.asarray(interval)-previous["credible_interval_95_ppm"]))))
        relative_change = change/(interval[1]-interval[0])
        rows.append({"case": name, "estimates": estimates,
                     "last_refinement_maximum_quantile_change_ppm": change,
                     "last_refinement_change_as_fraction_of_interval_width": relative_change,
                     "true_pco2_in_95_interval": interval[0] < truth.p_co2_ppm < interval[1],
                     "grid_check_passes": relative_change < 0.01})
    return {
        "scope": "Synthetic software and numerical integration validation; not observational sulfate validation",
        "truth": asdict(truth), "generated_sulfate_observation": asdict(observation),
        "air_cap_delta17_permil": air17, "air_delta18_permil": air18,
        "coordinate_prior": "uniform pCO2 per ppm on 3000-60000 ppm; GPP and pO2 fixed",
        "convergence_criterion": "last quantile change below 1% of the final 95% interval width",
        "cases": rows,
        "all_checks_pass": all(row["grid_check_passes"] and row["true_pco2_in_95_interval"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sulfate_transfer_audit.json")
    parser.add_argument("--inference", action="store_true", help="Include synthetic atmospheric-inference convergence checks.")
    args = parser.parse_args()
    report = build_report()
    if args.inference:
        report["inference_checks"] = inference_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in report["published_bao_2008_examples"]:
        print(f"{row['case']}: {row['calculated_air_permil']:.3f} per mil; published {row['published_air_permil']:.3f}")
    print("Svalbard air bounds:", report["cao_bao_2013_svalbard"]["calculated_air_bounds_permil"])
    print("Water-equilibrium sulfate, 0.528:", report["water_equilibrium_counterexample"]["calculated_sulfate_anomaly_0528_permil"])
    print(f"Wrote {args.output}")
    if args.inference and not report["inference_checks"]["all_checks_pass"]:
        raise RuntimeError("synthetic sulfate inference has not passed its declared checks")


if __name__ == "__main__":
    main()
