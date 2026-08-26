"""Configurable workbench for the Young et al. model reconstruction.

This script evaluates the reconstructed reaction network at a scaled Table 3
state. It is not a steady-state solver yet. Its purpose is to make parameter
experiments reproducible while unresolved model choices remain explicit.
"""

from __future__ import annotations

# --- path bootstrap (direct execution) ---
import sys as _sys
from pathlib import Path as _Path
_root = next((p for p in _Path(__file__).resolve().parents if (p / ".project-root").exists()), None)
if _root is not None:
    for _sub in ("code", "validation"):
        _p = str(_root / _sub)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
# --- end path bootstrap ---
import argparse
from dataclasses import dataclass
from math import exp, log

import numpy as np

from isotopes import R17_VSMOW, R18_VSMOW, cap_delta17_from_primes, delta_from_ratio
from reactions import derivative
from table3_state import SPECIES_ORDER, table3_state
from young_model_inventory import PARAMETERS, TABLE3_TARGETS
from young_reactions import executable_reactions


TABLE3_XO2 = 0.212
TABLE3_XCO2 = 2.944e-4
TABLE3_O2_TROP = 3.80e19
TABLE3_CO2_TROP = 5.29e16
TABLE3_CO2_STRAT = 5.29e15


@dataclass(frozen=True)
class IsotopeSummary:
    label: str
    delta18_prime: float
    delta17_prime: float
    cap_delta17: float


def prime_delta_from_ratio(ratio: float, reference_ratio: float) -> float:
    return 1000.0 * log(delta_from_ratio(ratio, reference_ratio) / 1000.0 + 1.0)


def singly_substituted_summary(label: str, n16: float, n17: float, n18: float, sites: float) -> IsotopeSummary:
    r17 = n17 / (sites * n16)
    r18 = n18 / (sites * n16)
    d17p = prime_delta_from_ratio(r17, R17_VSMOW)
    d18p = prime_delta_from_ratio(r18, R18_VSMOW)
    return IsotopeSummary(label, d18p, d17p, cap_delta17_from_primes(d17p, d18p))


def set_singly_substituted_from_primes(
    y: np.ndarray,
    idx: dict[str, int],
    n16_key: str,
    n17_key: str,
    n18_key: str,
    delta17_prime: float,
    delta18_prime: float,
    sites: float,
) -> None:
    n16 = y[idx[n16_key]]
    y[idx[n17_key]] = sites * n16 * R17_VSMOW * exp(delta17_prime / 1000.0)
    y[idx[n18_key]] = sites * n16 * R18_VSMOW * exp(delta18_prime / 1000.0)


def scaled_table3_state(p_o2_pal: float, p_co2_ppm: float, isotope_mode: str = "printed") -> np.ndarray:
    """Return Table 3 state scaled by pO2 PAL and pCO2 ppm.

    Scaling keeps isotope ratios fixed. This is a diagnostic state generator,
    not an atmospheric photochemistry solution. O2 isotopologues in both
    stratosphere and troposphere scale with pO2. CO2 isotopologues in both
    reservoirs scale with pCO2.
    """
    y = table3_state()
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    o2_scale = p_o2_pal
    co2_scale = (p_co2_ppm * 1.0e-6) / TABLE3_XCO2

    for name in ("O2_trop", "O18O_trop", "O17O_trop", "O2_strat", "O18O_strat", "O17O_strat"):
        y[idx[name]] *= o2_scale
    for name in ("CO2_trop", "CO18O_trop", "CO17O_trop", "CO2_strat", "CO18O_strat", "CO17O_strat"):
        y[idx[name]] *= co2_scale
    if isotope_mode == "printed":
        set_singly_substituted_from_primes(
            y,
            idx,
            "O2_trop",
            "O17O_trop",
            "O18O_trop",
            TABLE3_TARGETS["d17_O2_trop_permil"],
            TABLE3_TARGETS["d18_O2_trop_permil"],
            2.0,
        )
        set_singly_substituted_from_primes(
            y,
            idx,
            "O2_strat",
            "O17O_strat",
            "O18O_strat",
            11.886,
            23.212,
            2.0,
        )
        set_singly_substituted_from_primes(
            y,
            idx,
            "CO2_trop",
            "CO17O_trop",
            "CO18O_trop",
            TABLE3_TARGETS["d17_CO2_trop_permil"],
            TABLE3_TARGETS["d18_CO2_trop_permil"],
            1.0,
        )
        set_singly_substituted_from_primes(
            y,
            idx,
            "CO2_strat",
            "CO17O_strat",
            "CO18O_strat",
            TABLE3_TARGETS["d17_CO2_strat_permil"],
            TABLE3_TARGETS["d18_CO2_strat_permil"],
            1.0,
        )
    elif isotope_mode != "rounded":
        raise ValueError(f"unknown isotope mode: {isotope_mode}")
    return y


def initialize_finite_geosphere_burial_state(y: np.ndarray) -> np.ndarray:
    """Initialize candidate finite geosphere reservoirs for burial diagnostics.

    The initialization balances the explicit O2-weathering fluxes at modern
    pO2 for the PAL-normalized organic-burial interpretation. It deliberately
    does not include CO2-weathering oxygen, because that would turn carbonate/
    silicate CO2 drawdown into an O2-producing redox source in the lumped
    geosphere reservoir.
    """
    initialized = y.copy()
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    k_weather = PARAMETERS["k_O2_weathering_per_year"]
    k_org = PARAMETERS["k_organic_burial_per_year"]
    f_o2 = k_weather * initialized[idx["O2_trop"]]
    f_o18o = k_weather * initialized[idx["O18O_trop"]]
    f_o17o = k_weather * initialized[idx["O17O_trop"]]

    initialized[idx["O_geo"]] = (2.0 * f_o2 + f_o18o + f_o17o) / k_org
    initialized[idx["O18_geo"]] = f_o18o / k_org
    initialized[idx["O17_geo"]] = f_o17o / k_org

    # The water/biosphere reservoir is effectively infinite in Young. Give the
    # formal species nonzero values so 27-species log solvers can include them
    # without changing any rate law.
    initialized[idx["O_bio"]] = 1.0e24
    initialized[idx["O18_bio"]] = initialized[idx["O_bio"]] * R18_VSMOW
    initialized[idx["O17_bio"]] = initialized[idx["O_bio"]] * R17_VSMOW
    return initialized


def isotope_summaries(y: np.ndarray) -> list[IsotopeSummary]:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    return [
        singly_substituted_summary("O2_trop", y[idx["O2_trop"]], y[idx["O17O_trop"]], y[idx["O18O_trop"]], 2.0),
        # Young Table 3 CO18O/CO17O mole entries appear to represent one
        # singly substituted CO2 isotopomer, unlike the O2 equations where
        # singly substituted molecular O2 scales as 2R.
        singly_substituted_summary("CO2_trop", y[idx["CO2_trop"]], y[idx["CO17O_trop"]], y[idx["CO18O_trop"]], 1.0),
        singly_substituted_summary("CO2_strat", y[idx["CO2_strat"]], y[idx["CO17O_strat"]], y[idx["CO18O_strat"]], 1.0),
        singly_substituted_summary("O3_strat", y[idx["O3_strat"]], y[idx["OO17O_strat"]], y[idx["OO18O_strat"]], 3.0),
    ]


def largest_residuals(y: np.ndarray, dydt: np.ndarray, n: int = 12) -> list[tuple[str, float, float, float]]:
    scale = np.maximum(np.abs(y), 1.0)
    rel = np.abs(dydt) / scale
    order = np.argsort(rel)[::-1]
    return [(SPECIES_ORDER[i], y[i], dydt[i], rel[i]) for i in order[:n]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pO2-pal", type=float, default=1.0, help="Atmospheric O2 relative to present atmospheric level.")
    parser.add_argument("--pCO2-ppm", type=float, default=TABLE3_XCO2 * 1.0e6, help="Atmospheric CO2 in ppm.")
    parser.add_argument("--gpp-scale", type=float, default=1.0, help="GPP relative to nominal Young/Table 3 value.")
    parser.add_argument(
        "--isotope-mode",
        choices=("printed", "rounded"),
        default="printed",
        help="Use Young's printed isotope deltas or rounded Table 3 mole counts for isotopologues.",
    )
    parser.add_argument(
        "--r5-mode",
        choices=("paper", "table3-balanced"),
        default="paper",
        help="Use printed R5 M or diagnostic M that balances Eq. 26 at Table 3.",
    )
    parser.add_argument(
        "--closure-mode",
        choices=("none", "modern"),
        default="none",
        help="Enable explicitly labelled diagnostic closures for missing slow terms.",
    )
    parser.add_argument(
        "--co2-photo-sink-factor",
        type=float,
        default=1.0,
        help="Multiplier for CO18O/CO17O photosynthetic sinks. Use 2 as a diagnostic two-oxygen CO2 uptake convention.",
    )
    parser.add_argument(
        "--co2-photo-sink-mode",
        choices=("smow", "o2_source_water"),
        default="smow",
        help="Convention for CO18O/CO17O photosynthetic sink isotope composition.",
    )
    parser.add_argument(
        "--co2-closure-isotope-mode",
        choices=("rounded_table3", "printed_table3", "smow"),
        default="printed_table3",
        help="Isotope ratios used by the diagnostic modern CO2 closure.",
    )
    args = parser.parse_args()

    y = scaled_table3_state(args.pO2_pal, args.pCO2_ppm, isotope_mode=args.isotope_mode)
    r5_m = 5.589807e18 if args.r5_mode == "table3-balanced" else None
    reactions = executable_reactions(
        r5_collision_partner_moles=r5_m,
        gpp_scale=args.gpp_scale,
        closure_mode=args.closure_mode,
        co2_photo_sink_factor=args.co2_photo_sink_factor,
        co2_photo_sink_mode=args.co2_photo_sink_mode,
        co2_closure_isotope_mode=args.co2_closure_isotope_mode,
    )
    dydt = derivative(y, reactions, SPECIES_ORDER)
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}

    print("Young model reconstruction workbench")
    print("------------------------------------")
    print(f"pO2:       {args.pO2_pal:.6g} PAL")
    print(f"pCO2:      {args.pCO2_ppm:.6g} ppm")
    print(f"GPP scale: {args.gpp_scale:.6g}")
    print(f"R5 mode:   {args.r5_mode}")
    print(f"Isotopes:  {args.isotope_mode}")
    print(f"Closure:   {args.closure_mode}")
    print(f"CO2 closure isotope mode: {args.co2_closure_isotope_mode}")
    print(f"CO2 photo sink factor: {args.co2_photo_sink_factor:.6g}")
    print(f"CO2 photo sink mode:   {args.co2_photo_sink_mode}")
    print()
    print("Major reservoirs:")
    print(f"O2_trop:   {y[idx['O2_trop']]:.6e} mol")
    print(f"CO2_trop:  {y[idx['CO2_trop']]:.6e} mol")
    print(f"CO2_strat: {y[idx['CO2_strat']]:.6e} mol")
    print()
    if args.isotope_mode == "printed":
        print("Isotope summaries, initialized from Young's printed Table 3 deltas:")
    else:
        print("Isotope summaries, computed from rounded Table 3 mole counts:")
    for summary in isotope_summaries(y):
        print(
            f"{summary.label:10s} d18'={summary.delta18_prime:9.3f} per mil "
            f"d17'={summary.delta17_prime:9.3f} per mil "
            f"D17'={summary.cap_delta17:8.3f} per mil"
        )
    print()
    print("Selected residuals:")
    for name in ("O2_trop", "CO2_trop", "O2_strat", "CO2_strat", "O1D_strat"):
        i = idx[name]
        print(f"{name:10s} dydt={dydt[i]: .6e} mol/yr rel={abs(dydt[i]) / max(abs(y[i]), 1.0):.6e} yr^-1")
    print()
    print("Largest relative residuals:")
    for name, value, resid, rel in largest_residuals(y, dydt):
        print(f"{name:14s} y={value:.4e} dydt={resid:.4e} rel/yr={rel:.4e}")


if __name__ == "__main__":
    main()
