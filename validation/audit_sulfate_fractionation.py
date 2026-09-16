"""Literature-pathway sensitivity audit; never imported by the public solver.

Run: python validation/audit_sulfate_fractionation.py
The scenarios are synthetic conditional comparisons, not geological calibration
or a probability distribution for process uncertainty.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from math import exp, expm1, log, log1p
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
sys.path.insert(0, str(ROOT / "code"))
from isotopes import R17_VSMOW, R18_VSMOW  # noqa: E402
from sulfate_to_air import (  # noqa: E402
    SulfateProcessAssumptions, exact_air_from_sulfate, exact_sulfate_from_air,
    exact_sulfate_grid,
)
from updated_output_surface import (  # noqa: E402
    DEFAULT_OUTPUT_SURFACE_PATH, UpdatedOutputSurfaceInput, load_updated_output_surface,
)

SLOPE = 0.528
PAPER_SLOPE = 0.5305
MODERN_GPP = 290.0
PATHWAYS = ("surface", "dissolved_o2", "ferric", "equilibrium")
LABELS = {"surface": "Surface oxidation (Eq. 15)",
          "dissolved_o2": "Oxidation by dissolved O2 (Eq. 16)",
          "ferric": "Oxidation by Fe3+ (Eq. 17)",
          "equilibrium": "Sulfate-water equilibrium (Eq. 13)"}


@dataclass(frozen=True)
class LiteratureParameters:
    """Central Table 1 coefficients, not independent probability distributions."""
    sulfite_water_eq: float = 1.0129
    thiosulfate_kinetic: float = 0.9916
    sulfite_o2_kinetic: float = 0.9916
    sulfite_fe_kinetic: float = 0.9984
    water_kinetic: float = 0.9976
    o2_kinetic: float = 0.9850
    theta_kinetic: float = 0.511
    theta_equilibrium: float = 0.524


def ratios(delta18: float, anomaly: float, slope: float = SLOPE) -> np.ndarray:
    if not np.isfinite([delta18, anomaly, slope]).all() or delta18 <= -1000:
        raise ValueError("invalid isotope composition")
    prime18 = log1p(delta18 / 1000.)
    return np.array([R17_VSMOW * exp(anomaly / 1000. + slope * prime18),
                     R18_VSMOW * exp(prime18)])


def isotope_pair(r: np.ndarray, slope: float = SLOPE) -> dict:
    prime = 1000. * np.log(r / np.array([R17_VSMOW, R18_VSMOW]))
    return {"delta18_permil": float(1000. * np.expm1(prime[1] / 1000.)),
            "cap_delta17_permil": float(prime[0] - slope * prime[1])}


def fractionation(alpha18: float, theta: float) -> np.ndarray:
    if alpha18 <= 0 or not np.isfinite([alpha18, theta]).all():
        raise ValueError("invalid fractionation")
    return np.array([alpha18**theta, alpha18])


def mix(parts: list[tuple[float, np.ndarray]], *, exact_atoms: bool) -> np.ndarray:
    weights = np.array([weight for weight, _ in parts])
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-14, rtol=0):
        raise ValueError("oxygen fractions must sum to one")
    if not exact_atoms:
        return sum(weight * r for weight, r in parts)
    abundances = [np.r_[1., r] / (1. + r.sum()) for _, r in parts]
    total = sum(weight * a for (weight, _), a in zip(parts, abundances))
    return total[1:] / total[0]


def water_ratios(delta18: float) -> np.ndarray:
    # Cao/Bao Eq. 18: delta'17 = .528 delta'18 + .000033 in dimensionless units.
    return ratios(delta18, 0.033)


def formation(pathway: str, water: np.ndarray, dissolved_o2: np.ndarray,
              parameters: LiteratureParameters = LiteratureParameters(), *, exact_atoms=True) -> dict:
    """Reproduce Eqs. 13, 15-17, or retain their components in exact atom balance.

    Printed equations linearly combine isotope ratios. Exact mode instead
    combines isotope abundances with the stated all-oxygen atom fractions.
    """
    p = parameters
    eq = fractionation(p.sulfite_water_eq, p.theta_equilibrium)
    kinetics = lambda alpha: fractionation(alpha, p.theta_kinetic)
    if pathway == "dissolved_o2":
        nonair = water * eq * kinetics(p.sulfite_o2_kinetic)
        transferred = dissolved_o2 * kinetics(p.o2_kinetic)
        parts = [(0.75, nonair), (0.25, transferred)]
        f = 0.25
    elif pathway in ("surface", "ferric"):
        alpha = p.thiosulfate_kinetic if pathway == "surface" else p.sulfite_fe_kinetic
        parts = [(0.75, water * eq * kinetics(alpha)), (0.25, water * kinetics(p.water_kinetic))]
        nonair = mix(parts, exact_atoms=exact_atoms)
        f = 0.
    elif pathway == "equilibrium":
        # Table 1's 1.023 is rounded; use its specified source, Eq. 6 at 25 C.
        alpha = exp((2.68e6 / 298.15**2 - 7.45) / 1000.)
        nonair = water * fractionation(alpha, p.theta_equilibrium)
        parts, f = [(1., nonair)], 0.
    else:
        raise ValueError("unknown sulfate pathway")
    return {"ratios": mix(parts, exact_atoms=exact_atoms), "nonair_ratios": nonair, "fraction": f}


def dissolution_factor(mode: str) -> np.ndarray:
    if mode == "identity":
        return np.ones(2)
    if mode == "paper_pair_ratio":
        # A declared sensitivity bridge, not a temperature/salinity-dependent law.
        return ratios(24.2, -0.554, PAPER_SLOPE) / ratios(23.88, -0.553, PAPER_SLOPE)
    raise ValueError("unknown gas-to-water assumption")


def compact_process(result: dict, gas_factor: np.ndarray, p: LiteratureParameters) -> SulfateProcessAssumptions:
    total = gas_factor * fractionation(p.o2_kinetic, p.theta_kinetic)
    if total[1] == 1. and total[0] != 1.:
        raise ValueError("alpha18=1 with alpha17!=1 cannot be represented by a finite theta")
    theta = log(total[0]) / log(total[1]) if total[1] != 1. else SLOPE
    return SulfateProcessAssumptions(
        background_cap_delta17_permil=isotope_pair(result["nonair_ratios"])["cap_delta17_permil"],
        alpha18_air_to_sulfate=float(total[1]), theta_air_to_sulfate=theta,
        assumption_note="Synthetic Cao/Bao Eq. 16 pathway; fixed water and preserved primary sulfate; specified gas-to-water treatment.",
    )


@lru_cache(maxsize=20000)
def atmospheric_pair(po2: float, pco2: float, gpp: float) -> dict:
    output = load_updated_output_surface().evaluate(UpdatedOutputSurfaceInput(po2, pco2, gpp))
    return {"delta18_permil": 1000. * expm1(output.central_delta18_prime_permil / 1000.),
            "cap_delta17_permil": output.central_cap_delta17_prime_permil}


@lru_cache(maxsize=32)
def atmospheric_curve(po2: float, gpp: float, nodes: int) -> tuple:
    surface = load_updated_output_surface()
    lower, upper = surface.domain["pco2_ppm"]
    axis = np.unique(np.r_[np.geomspace(lower, upper, nodes), surface.pco2_nodes])
    d17 = surface.evaluate_central_cap_delta17_grid(p_o2_pal=po2, p_co2_ppm=axis, gpp_pgC_per_year=gpp)
    d18 = 1000. * np.expm1(surface.evaluate_central_delta18_prime_grid(
        p_o2_pal=po2, p_co2_ppm=axis, gpp_pgC_per_year=gpp) / 1000.)
    return axis, d17, d18


def solve_pco2(sulfate: dict, process: SulfateProcessAssumptions, *, po2: float, gpp: float,
               nodes: int = 513) -> dict:
    axis, air17, air18 = atmospheric_curve(po2, gpp, nodes)
    values, physical = exact_sulfate_grid(
        air_cap_delta17_permil=air17, air_delta18_permil=air18,
        sulfate_delta18_permil=sulfate["delta18_permil"], fraction=0.25,
        background_cap_delta17_permil=process.background_cap_delta17_permil,
        alpha18_air_to_sulfate=process.alpha18_air_to_sulfate,
        theta_air_to_sulfate=process.theta_air_to_sulfate)
    residual = values - sulfate["cap_delta17_permil"]

    def objective(co2):
        air = atmospheric_pair(po2, float(co2), gpp)
        return exact_sulfate_from_air(
            air_cap_delta17_permil=air["cap_delta17_permil"], air_delta18_permil=air["delta18_permil"],
            sulfate_delta18_permil=sulfate["delta18_permil"], fraction=0.25,
            process=process) - sulfate["cap_delta17_permil"]

    roots = [float(x) for x, r, ok in zip(axis, residual, physical) if ok and abs(r) < 1e-10]
    for i in range(len(axis) - 1):
        # Node roots are already retained; roundoff around them can flip signs
        # between the independent scalar and batched isotope solvers.
        if (physical[i] and physical[i+1] and min(abs(residual[i]), abs(residual[i+1])) > 1e-10
                and residual[i] * residual[i+1] < 0):
            root = brentq(objective, axis[i], axis[i+1], xtol=1e-7, rtol=1e-12)
            if not any(abs(root - old) < 1e-4 for old in roots):
                roots.append(float(root))
    roots.sort()
    return {"status": "one_root" if len(roots) == 1 else "multiple_roots" if roots else "no_root_in_domain",
            "roots_ppm": roots, "domain_ppm": [float(axis[0]), float(axis[-1])],
            "maximum_sulfate_residual_permil": max((abs(objective(x)) for x in roots), default=None)}


def comparison(air: dict, water18: float, mode="identity", parameters=LiteratureParameters()) -> dict:
    gas = dissolution_factor(mode)
    result = formation("dissolved_o2", water_ratios(water18),
                       ratios(air["delta18_permil"], air["cap_delta17_permil"]) * gas, parameters)
    sulfate = isotope_pair(result["ratios"])
    correct = compact_process(result, gas, parameters)
    variants = {"literature": correct,
                "no_air_fractionation": replace(correct, alpha18_air_to_sulfate=1., theta_air_to_sulfate=SLOPE),
                "zero_nonair_and_no_fractionation": replace(correct, alpha18_air_to_sulfate=1.,
                    theta_air_to_sulfate=SLOPE, background_cap_delta17_permil=0.)}
    answers = {}
    for name, process in variants.items():
        recovered = exact_air_from_sulfate(sulfate_cap_delta17_permil=sulfate["cap_delta17_permil"],
            sulfate_delta18_permil=sulfate["delta18_permil"], fraction=0.25,
            air_delta18_permil=air["delta18_permil"], process=process)
        forward = exact_sulfate_from_air(air_cap_delta17_permil=air["cap_delta17_permil"],
            air_delta18_permil=air["delta18_permil"], sulfate_delta18_permil=sulfate["delta18_permil"],
            fraction=0.25, process=process)
        answers[name] = {"air_anomaly_permil": recovered["air"]["cap_delta17_permil"],
                         "air_bias_permil": recovered["air"]["cap_delta17_permil"] - air["cap_delta17_permil"],
                         "sulfate_residual_permil": forward - sulfate["cap_delta17_permil"],
                         "implied_nonair_delta18_permil": recovered["implied_nonair_oxygen"]["delta18_permil"],
                         "atom_closure_residual": recovered["maximum_atom_closure_residual"],
                         "process": asdict(process)}
    return {"air": air, "water_delta18_permil": water18, "water_anomaly_permil": .033,
            "gas_to_water": mode, "sulfate": sulfate, "fraction": .25,
            "nonair": isotope_pair(result["nonair_ratios"]), "variants": answers}


# Wei et al. (2026), Table 1, p. 4. Preserve printed compositions and printed
# differences separately: subtraction of the rounded columns can differ by .001.
# T, pH, d18, SD18, D17_5305, SD17, diff18, SDdiff18, diff17, SDdiff17, N.
WEI_2026_TABLE1 = (
    (12, 4.60, 7.116, .173, -.111, .009, 16.126, .173, -.159, .009, 2),
    (25, 4.60, 6.771, .038, -.106, .003, 15.781, .040, -.153, .003, 2),
    (40, 4.60, 5.234, .322, -.095, .005, 14.244, .323, -.143, .005, 2),
    (55, 4.60, 4.471, .057, -.092, .017, 13.481, .059, -.139, .017, 2),
    (12, 6.28, 6.290, .256, -.115, .001, 15.300, .256, -.162, .002, 2),
    (25, 6.28, 5.868, .170, -.101, .011, 14.878, .171, -.148, .011, 2),
    (40, 6.28, 3.858, .154, -.100, .006, 12.868, .155, -.148, .006, 2),
    (55, 6.28, 3.196, .036, -.080, .003, 12.206, .038, -.128, .003, 2),
    (12, 7.30, 3.705, .185, -.107, .005, 12.715, .185, -.154, .005, 2),
    (25, 7.30, 3.186, .107, -.104, .002, 12.196, .108, -.152, .003, 2),
    (40, 7.30, 1.399, .031, -.091, .025, 10.409, .034, -.138, .025, 2),
    (55, 7.30, .159, .585, -.066, .004, 9.169, .586, -.114, .004, 2),
    (12, 8.89, 1.804, .144, -.102, .003, 10.814, .144, -.150, .003, 2),
    (25, 8.89, 1.525, .176, -.094, .003, 10.535, .177, -.141, .003, 2),
    (40, 8.89, -.379, .157, -.082, .002, 8.631, .158, -.130, .003, 2),
    (55, 8.89, -.658, .320, -.076, .021, 8.353, .321, -.123, .021, 2),
)
WEI_2026_COEFFICIENTS = {
    "bisulfite": {"a18": 7060., "b18": -8.80, "a17": 3690., "b17": -4.664,
                  "a18_sd": 1060., "b18_sd": 3.49, "a17_sd": 570., "b17_sd": 1.86,
                  "theta": .5202, "theta_sd": .0003, "b17_rounding_halfwidth": .0005},
    "sulfite": {"a18": 6590., "b18": -12.56, "a17": 3420., "b17": -6.56,
                "a18_sd": 1320., "b18_sd": 4.34, "a17_sd": 700., "b17_sd": 2.30,
                "theta": .5155, "theta_sd": .0008, "b17_rounding_halfwidth": .005},
}


def log_transfer(log18_permil: float, log17_permil: float) -> dict:
    """Effective two-ratio transfer, including net alpha18=1 with alpha17!=1."""
    if not np.isfinite([log18_permil, log17_permil]).all():
        raise ValueError("isotope log shifts must be finite")
    return {"log18_permil": log18_permil, "log17_permil": log17_permil,
            "alpha18": exp(log18_permil/1000.), "alpha17": exp(log17_permil/1000.),
            "theta": log17_permil/log18_permil if log18_permil != 0. else None,
            "anomaly_shift_528_permil": log17_permil-SLOPE*log18_permil,
            "anomaly_shift_5305_permil": log17_permil-PAPER_SLOPE*log18_permil}


def peng_2026_apparent_transfer(log18_permil: float) -> dict:
    """Reconstruct Fig. 7's straight-line construction, not a reversibility rate law.

    The return-process slope is anchored at the forward kinetic endpoint.
    It must not instead be drawn through unmodified atmospheric O2.
    """
    if not np.isfinite(log18_permil) or not -21.6 <= log18_permil <= 23.3:
        raise ValueError("outside the illustrated Peng Figure 7 interval")
    log17 = .5145*(-21.6) + .5134*(log18_permil+21.6)
    return log_transfer(log18_permil, log17)


def wei_2026_transfer(species: str, temperature_c: float, representation: str) -> dict:
    if species not in WEI_2026_COEFFICIENTS:
        raise ValueError("unknown Wei sulfite species")
    if not np.isfinite(temperature_c) or not 12. <= temperature_c <= 55.:
        raise ValueError("outside Wei's measured 12-55 C range")
    p = WEI_2026_COEFFICIENTS[species]
    kelvin = temperature_c + 273.15
    log18 = p["a18"]/kelvin + p["b18"]
    if representation == "paired_printed_equations":
        log17 = p["a17"]/kelvin + p["b17"]
    elif representation == "reported_mean_theta":
        log17 = p["theta"]*log18
    else:
        raise ValueError("unknown Wei coefficient representation")
    return {"species": species, "temperature_c": temperature_c,
            "representation": representation, **log_transfer(log18, log17)}


def source_2026_report() -> dict:
    water18, water17 = -9.010, .048
    observations = []
    for t, ph, d18, sd18, d17, sd17, diff18, sdiff18, diff17, sdiff17, n in WEI_2026_TABLE1:
        log18 = 1000.*(log1p(d18/1000.)-log1p(water18/1000.))
        observations.append({"temperature_c": t, "pH": ph, "delta18_permil": d18,
            "delta18_sd": sd18, "anomaly_5305_permil": d17, "anomaly_sd": sd17,
            "printed_delta18_difference_permil": diff18, "printed_delta18_difference_sd": sdiff18,
            "printed_anomaly_difference_permil": diff17, "printed_anomaly_difference_sd": sdiff17, "n": n,
            "log18_fractionation_permil": log18,
            "anomaly_528_permil": d17+(PAPER_SLOPE-SLOPE)*1000.*log1p(d18/1000.),
            "anomaly_difference_528_from_compositions_permil": d17-water17+(PAPER_SLOPE-SLOPE)*log18,
            "printed_minus_subtracted_anomaly_difference_permil": diff17-(d17-water17)})
    wei_cases = [wei_2026_transfer(s, float(t), rep) for s in WEI_2026_COEFFICIENTS
                 for t in np.linspace(12, 55, 87) for rep in ("paired_printed_equations", "reported_mean_theta")]
    wei_25 = []
    for s, p in WEI_2026_COEFFICIENTS.items():
        paired = wei_2026_transfer(s, 25., "paired_printed_equations")
        theta = wei_2026_transfer(s, 25., "reported_mean_theta")
        gap = paired["anomaly_shift_528_permil"]-theta["anomaly_shift_528_permil"]
        # This last-digit bound is not a statistical uncertainty. Unknown full
        # precision and coefficient covariance preclude inferring one from it.
        rounding_bound = 5./298.15+p["b17_rounding_halfwidth"]+p["theta"]*(5./298.15+.005)
        wei_25.append({"species": s, "paired_equations": paired, "reported_theta": theta,
                       "anomaly_representation_difference_permil": gap,
                       "independent_last_digit_rounding_bound_permil": rounding_bound,
                       "rounding_could_cover_25C_gap": abs(gap) <= rounding_bound})
    peng_paths = [dict(name=name, **peng_2026_apparent_transfer(l18)) for name, l18 in (
        ("irreversible_forward", -21.6), ("Fig7_pyrite_example", -9.8),
        ("net_zero_delta18", 0.), ("Fig7_apparent_inverse_lower", 7.1), ("Fig7_apparent_inverse_upper", 23.3))]
    peng_paths.append(dict(name="reported_SO5_O2_equilibrium", **log_transfer(-3.5, .5199*(-3.5))))
    return {
        "scope": "Source-term comparisons only; no combined sulfate preset, atmospheric change or process probability",
        "peng": {"doi": "10.1016/j.gca.2025.11.036", "source_pdf_sha256": "82e3f111bf5fd8255b96d570be4e171add93f9c332e7ccfa5b2e1ecd0d9a0c6c",
                 "locations": "pp. 65-69, 71-73; Eqs. 1-13; Figs. 2, 6, 7; Table 1",
                 "temperature_c": 25., "paper_slope": PAPER_SLOPE,
                 "statistics": "Quoted errors are 2SE across configurations, not 1SD process priors",
                 "forward": {"log18_permil": -21.6, "log18_2se": .6, "theta": .5145, "theta_2se": .0009, "n": 4},
                 "equilibrium": {"log18_permil": -3.5, "log18_2se": .8, "theta": .5199, "theta_2se": .0021, "n": 3},
                 "backward": {"log18_permil": -18.1, "log18_2se": 1., "theta": .5134, "theta_2se": .0012},
                 "transfers": peng_paths,
                 "figure7_construction": [peng_2026_apparent_transfer(float(x)) for x in np.linspace(-21.6, 23.3, 201)],
                 "supplement": {"archive_sha256": "d8273731a52fa65f9cf2fb3481dc721cc0710c74377863faf50009f08aae1127",
                      "inspection": "Local inspection of 11 Gaussian logs and Table S1; not rerun by this audit",
                      "table_s1_sha256": "83ae269a66f57ae496579294738edb9a8791b55c28c7de9169349d230ce3120f",
                      "contents": "Optimized geometries and frequency output, plus geometry summary; no environmental reversibility law",
                      "ts_mean_OO_bond_angstrom": 1.203595, "ts_mean_SO_bond_angstrom": 2.43292,
                      "quality_note": "SO5-B equilibrium-conformer log contains one -15.2735 cm^-1 mode; treatment needs clarification for a full partition-function reproduction"}},
        "wei": {"doi": "10.1016/j.epsl.2026.119862", "source_pdf_sha256": "fb460ba380e4eac746a0fb3997d50abcbc3041568d4ee0d212306642191f17f4",
                "locations": "pp. 3-5, 7-9; Table 1, Eqs. 1-4, Fig. 3, Sections 5.1-5.2",
                "temperature_range_c": [12., 55.], "experimental_pH_range": [4.60, 8.89],
                "statistics": "1SD as reported; no joint regression-coefficient covariance supplied",
                "water": {"delta18_permil": water18, "delta18_sd": .013,
                          "anomaly_5305_permil": water17, "anomaly_sd": .001, "n": 5},
                "coefficients": WEI_2026_COEFFICIENTS, "observations": observations,
                "temperature_comparisons": wei_cases, "comparisons_25C": wei_25},
        "unresolved": [
            "Net kinetic effects on sulfite-derived oxygen and their species dependence are not jointly calibrated here",
            "Peng gives a reversibility interpretation but no quantitative mapping from atmospheric pO2 to apparent transfer",
            "Unknown resetting and geological pathway proportions remain external constraints",
            "Wei Section 5.1 applies -0.1 per mil analytical corrections to selected previous datasets; no blanket correction is applied here",
            "Effective alpha18=1 with nonunit alpha17 needs a two-ratio or two-log-shift descriptor, not one alpha/theta pair",
        ],
    }


def build_report() -> dict:
    started = perf_counter()
    paper_dissolved = ratios(24.2, -.554, PAPER_SLOPE)
    figure5, ratio_errors = [], []
    for pathway in PATHWAYS:
        for water18 in np.linspace(-20., 0., 81):
            printed = formation(pathway, water_ratios(water18), paper_dissolved, exact_atoms=False)
            exact = formation(pathway, water_ratios(water18), paper_dissolved, exact_atoms=True)
            literal_pair = isotope_pair(printed["ratios"], PAPER_SLOPE)
            exact_pair = isotope_pair(exact["ratios"], PAPER_SLOPE)
            figure5.append({"pathway": pathway, "water_delta18_permil": float(water18), **literal_pair})
            ratio_errors.append(abs(exact_pair["cap_delta17_permil"] - literal_pair["cap_delta17_permil"]))
    air_cases = [comparison({"delta18_permil": 23.9, "cap_delta17_permil": float(a)}, water)
                 for water in (-20., -10., 0.) for a in np.unique(np.r_[-.432, -10., np.linspace(-.1, -20., 101)])]
    linked = []
    for po2, percent in ((1., 100.), (1., 25.), (.5, 100.), (.5, 25.)):
        for co2 in (294., 1000., 10000., 30000., 60000.):
            for water18 in (-20., 0.):
                for mode in ("identity", "paper_pair_ratio"):
                    row = comparison(atmospheric_pair(po2, co2, MODERN_GPP * percent/100.), water18, mode)
                    row.update(po2_pal=po2, gpp_percent=percent, generating_pco2_ppm=co2)
                    for variant in row["variants"].values():
                        variant["pco2_inversion"] = solve_pco2(row["sulfate"], SulfateProcessAssumptions(**variant["process"]),
                                                               po2=po2, gpp=MODERN_GPP * percent/100.)
                    linked.append(row)
    sensitivities = []
    # One-factor scenarios only: the fitted coefficients lack a joint covariance.
    for name, p in [("central", LiteratureParameters()),
                    ("O2 KIE - reported spread", replace(LiteratureParameters(), o2_kinetic=.984)),
                    ("O2 KIE + reported spread", replace(LiteratureParameters(), o2_kinetic=.986)),
                    ("kinetic theta lower reduced-mass bound", replace(LiteratureParameters(), theta_kinetic=.5052)),
                    ("kinetic theta upper reduced-mass bound", replace(LiteratureParameters(), theta_kinetic=.5168))]:
        row = comparison({"delta18_permil": 23.9, "cap_delta17_permil": -.432}, 0., parameters=p)
        sensitivities.append({"scenario": name, "parameters": asdict(p), "result": row})
    surface = load_updated_output_surface()
    # Compare a few paired isotope states to the uncached central solver too.
    from updated_molecular_forward_model import UpdatedForwardInput, run_updated_central_state
    live_checks = []
    for po2, co2, percent in ((1., 294., 100.), (1., 30000., 100.), (.5, 1000., 25.), (.5, 30000., 25.)):
        gpp = MODERN_GPP * percent / 100.
        live = run_updated_central_state(UpdatedForwardInput(po2, co2, gpp))
        air = {"delta18_permil": 1000. * expm1(live.delta18_prime_permil / 1000.),
               "cap_delta17_permil": live.cap_delta17_prime_permil}
        cached = atmospheric_pair(po2, co2, gpp)
        row = comparison(air, 0.)
        live_checks.append({"po2_pal": po2, "pco2_ppm": co2, "gpp_percent": percent,
                            "converged": live.numerically_converged,
                            "fixed_point_residual_permil": live.maximum_fixed_point_residual_permil,
                            "surface_minus_live_air_permil": {key: cached[key]-air[key] for key in air},
                            "no_fractionation_air_bias_permil": row["variants"]["no_air_fractionation"]["air_bias_permil"]})
    roundtrip = max(abs(r["variants"]["literature"]["air_bias_permil"]) for r in air_cases + linked)
    roots = [r["variants"]["literature"]["pco2_inversion"]["roots_ppm"] for r in linked]
    if not all(len(values) == 1 for values in roots):
        raise RuntimeError("matching synthetic sulfate transfer failed to recover a unique generating state")
    co2_error = max(abs(values[0]-r["generating_pco2_ppm"]) for values, r in zip(roots, linked))
    if roundtrip > 1e-8 or co2_error > 1e-3 or not all(r["converged"] for r in live_checks):
        raise RuntimeError("sulfate audit numerical closure failed")
    summary = {"matched_transfer_max_air_error_permil": roundtrip,
               "matched_transfer_max_pco2_error_ppm": co2_error,
               "matched_transfer_cases": len(linked),
               "no_root_counts_by_variant": {name: sum(not r["variants"][name]["pco2_inversion"]["roots_ppm"] for r in linked)
                   for name in ("literature", "no_air_fractionation", "zero_nonair_and_no_fractionation")}}
    return {
        "scope": "Synthetic formation and inversion sensitivity; no empirical process-error distribution, default change or atmospheric recalibration",
        "source": {"citation": "Cao and Bao (2021), Reviews in Mineralogy and Geochemistry 86, 463-488",
                   "doi": "10.2138/rmg.2021.86.14", "equations": [6, 13, 15, 16, 17, 18],
                   "locators": "pp. 466, 469, 471-473, 476-478; Table 1; Figure 5",
                   "paper_slope": PAPER_SLOPE, "paper_dissolved_delta18_permil": 24.2,
                   "paper_dissolved_anomaly_permil": -.554, "parameters": asdict(LiteratureParameters()),
                   "parameter_status": "Conditional Monte Carlo estimates for selected laboratory conditions; kinetic/equilibrium theta assumed, not universal measured exponents"},
        "conventions": {"comparison_slope": SLOPE, "isotope_standard": "VSMOW", "R17": R17_VSMOW, "R18": R18_VSMOW,
                        "modern_gpp_pgC_per_year": MODERN_GPP,
                        "formation": "Printed ratio sum for Figure 5; separately declared exact isotope-atom mixtures for inverse experiments",
                        "gas_to_water_identity": "Controlled assumption to isolate formation effects; not a claim that gas dissolution is fractionation-free",
                        "gas_to_water_paper_pair_ratio": "Constant ratio from cited atmospheric 23.88/-0.553 and dissolved 24.2/-0.554 pairs in log-0.5305; diagnostic bridge, not a temperature/salinity law",
                        "inversion": "Central roots at fixed GPP and pO2, using both modelled air isotope coordinates; no roots outside accepted domain are invented",
                        "water": "Eq. 18 meteoric-water line, -20 to 0 per mil delta18; no resetting, pathway Eq.16 fixes f=0.25",
                        "nonair": "The full pathway sets its post-formation anomaly; inverse atom balance derives its delta18 from measured sulfate and candidate air. Generating water delta18 is not an extra inverse constraint."},
        "provenance": {"surface_data_id": surface.surface_data_id, "upstream_model_data_id": surface.upstream_model_data_id,
                       "surface_sha256": hashlib.sha256(DEFAULT_OUTPUT_SURFACE_PATH.read_bytes()).hexdigest(),
                       "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "figure5": figure5, "max_printed_vs_exact_atom_anomaly_difference_permil": max(ratio_errors),
        "air_cases": air_cases, "atmospheric_cases": linked, "one_factor_sensitivities": sensitivities,
        "live_central_checks": live_checks, "summary": summary, "source_2026": source_2026_report(),
        "elapsed_seconds": perf_counter() - started,
    }


def plot_report(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 9.), constrained_layout=True)
    colors = {"surface": "#965590", "dissolved_o2": "#187e80", "ferric": "#4263a1", "equilibrium": "#a87525"}
    for pathway in PATHWAYS:
        rows = [r for r in report["figure5"] if r["pathway"] == pathway]
        axs[0, 0].plot([r["delta18_permil"] for r in rows], [r["cap_delta17_permil"] for r in rows],
                       label=LABELS[pathway], color=colors[pathway])
    water18 = np.linspace(-20, 0, 81)
    axs[0, 0].plot(water18, .033 + (SLOPE-PAPER_SLOPE)*1000*np.log1p(water18/1000), ":", color="#666666", label="Water (Eq. 18)")
    axs[0, 0].set(title="A  Published formation equations", xlabel=r"Sulfate $\delta^{18}$O (per mil, VSMOW)",
                  ylabel=r"$\Delta'^{17}$O$_{0.5305}$ (per mil)")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    variants = {"no_air_fractionation": ("No air fractionation; same non-air anomaly", "#2673ac"),
                "zero_nonair_and_no_fractionation": ("No air fractionation; non-air anomaly = 0", "#bc702b")}
    for key, (label, color) in variants.items():
        for water, style in ((0., "-"), (-20., "--")):
            rows = sorted([r for r in report["air_cases"] if r["water_delta18_permil"] == water], key=lambda r: r["air"]["cap_delta17_permil"])
            axs[0, 1].plot([r["air"]["cap_delta17_permil"] for r in rows], [r["variants"][key]["air_bias_permil"] for r in rows],
                           style, color=color, label=label if water == 0 else None)
    axs[0, 1].axhline(0., color="#444444", linewidth=1)
    axs[0, 1].set(title="B  Air reconstruction bias: Eq. 16 sulfate", xlabel=r"Generating air $\Delta'^{17}$O$_{0.528}$ (per mil)",
                  ylabel="Reconstructed minus generating air (per mil)")
    axs[0, 1].legend(fontsize=8, loc="best")
    axs[0, 1].text(.04, .08, r"Water $\delta^{18}$O: solid 0; dashed -20 per mil", transform=axs[0, 1].transAxes, fontsize=8)
    for ax, percent, letter in ((axs[1, 0], 100., "C"), (axs[1, 1], 25., "D")):
        rows = [r for r in report["atmospheric_cases"] if r["po2_pal"] == 1. and r["gpp_percent"] == percent
                and r["water_delta18_permil"] == 0. and r["gas_to_water"] == "identity"]
        co2 = [r["generating_pco2_ppm"] for r in rows]
        matched = [r["variants"]["literature"]["pco2_inversion"]["roots_ppm"][0] for r in rows]
        ax.plot(co2, matched, color="#333333", label="Matched transfer (round trip)")
        for key, (label, color) in variants.items():
            values = [r["variants"][key]["pco2_inversion"]["roots_ppm"] for r in rows]
            ax.plot(co2, [v[0] if len(v) == 1 else np.nan for v in values], "o--", color=color, label=label, markersize=4)
        ax.set(xscale="log", yscale="log", title=rf"{letter}  pCO$_2$ reconstruction: {percent:g}% GPP, 1 PAL O$_2$",
               xlabel=r"Generating pCO$_2$ (ppm)", ylabel=r"Reconstructed pCO$_2$ (ppm)", ylim=(50, 80000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.legend(fontsize=8, loc="upper left")
    for ax in axs.flat:
        ax.grid(alpha=.18)
    fig.suptitle("Sulfate formation sensitivity: synthetic cases, not an uncertainty envelope\n"
                 "B-D: 25% air-derived oxygen; gas-to-water fractionation held at zero. C-D: water delta18O = 0 per mil.\n"
                 "Missing reconstruction points have no root in the accepted 50-60,000 ppm domain.", fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_2026_sources(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    data = report["source_2026"]
    fig, axs = plt.subplots(1, 2, figsize=(12., 4.9), constrained_layout=True)
    curve = data["peng"]["figure7_construction"]
    axs[0].plot([r["log18_permil"] for r in curve], [r["anomaly_shift_528_permil"] for r in curve],
                color="#187e80", label="Peng Fig. 7 construction (25 C)")
    for name, label, color in (("irreversible_forward", "Forward kinetic limit", "#2673ac"),
                              ("reported_SO5_O2_equilibrium", "Reported equilibrium limit", "#bc702b")):
        p = next(r for r in data["peng"]["transfers"] if r["name"] == name)
        axs[0].scatter(p["log18_permil"], p["anomaly_shift_528_permil"], s=35, c=color, label=label, zorder=3)
    axs[0].axhline(0, color="#666666", lw=.7)
    axs[0].axvline(0, color="#666666", lw=.7)
    axs[0].set(title=r"A  O$_2$-derived contribution", xlabel=r"$1000\ln\alpha^{18}$ (per mil)",
               ylabel=r"Transferred minus air $\Delta'^{17}$O$_{0.528}$ (per mil)")
    axs[0].legend(fontsize=8)
    for species, color in (("bisulfite", "#2673ac"), ("sulfite", "#bc702b")):
        for rep, style, suffix in (("reported_mean_theta", "-", "reported mean theta"),
                                    ("paired_printed_equations", "--", "paired printed equations")):
            rows = [r for r in data["wei"]["temperature_comparisons"] if r["species"] == species and r["representation"] == rep]
            axs[1].plot([r["temperature_c"] for r in rows], [r["anomaly_shift_528_permil"] for r in rows],
                        style, color=color, label=f"{species.capitalize()}: {suffix}")
    axs[1].set(title="B  Water-equilibrated precursor", xlabel="Temperature (C)",
               ylabel=r"Precursor minus water $\Delta'^{17}$O$_{0.528}$ (per mil)")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.18)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle("Sulfate source terms from Peng et al. (2026) and Wei et al. (2026)\n"
                 "Individual transfer effects, before final sulfate mixing; not a combined archive correction", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sulfate_fractionation_audit.json")
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plot_report(report, args.output.with_suffix(".png"))
    plot_2026_sources(report, args.output.with_name(args.output.stem+"_2026.png"))
    print(json.dumps({"output": str(args.output), "plot": str(args.output.with_suffix('.png')),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "max_ratio_atom_difference_permil": report["max_printed_vs_exact_atom_anomaly_difference_permil"]}, indent=2))


if __name__ == "__main__":
    main()
