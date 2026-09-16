"""Oxygen isotope utilities for the Young et al. (2014) model rebuild.

The functions here implement notation and mechanical calculations that are
explicit in the paper. They do not encode any model-specific assumptions.
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
from dataclasses import dataclass
from math import expm1, log


# VSMOW isotope ratios used by Young-style oxygen isotope calculations.
# These are standard accepted values; if the original Fortran used different
# rounded constants, Table 3 validation will reveal the mismatch.
R17_VSMOW = 0.0003799
R18_VSMOW = 0.0020052


@dataclass(frozen=True)
class IsotopeMasses:
    O16: float = 15.99491461957
    O17: float = 16.99913175650
    O18: float = 17.99915961286
    C12: float = 12.0


MASSES = IsotopeMasses()


def delta_from_ratio(ratio: float, reference_ratio: float) -> float:
    """Return conventional delta in per mil."""
    return 1000.0 * (ratio / reference_ratio - 1.0)


def ratio_from_delta(delta_permil: float, reference_ratio: float) -> float:
    """Return isotope ratio from conventional delta in per mil."""
    return reference_ratio * (delta_permil / 1000.0 + 1.0)


def log_delta(delta_permil: float) -> float:
    """Return delta-prime: 1000 * ln(delta/1000 + 1)."""
    return 1000.0 * log(delta_permil / 1000.0 + 1.0)


def conventional_delta_from_prime(delta_prime_permil: float) -> float:
    """Return conventional delta from logarithmic delta-prime, in per mil."""
    return 1000.0 * expm1(delta_prime_permil / 1000.0)


def cap_delta17(
    delta17_permil: float,
    delta18_permil: float,
    slope: float = 0.528,
    gamma: float = 0.0,
    *,
    intercept: float | None = None,
) -> float:
    """Return Δ′17O in per mil."""
    if intercept is not None:
        if gamma != 0.0:
            raise ValueError("provide gamma or the legacy intercept alias, not both")
        gamma = -intercept
    return log_delta(delta17_permil) - slope * log_delta(delta18_permil) - gamma


def cap_delta17_from_primes(
    delta17_prime: float,
    delta18_prime: float,
    slope: float = 0.528,
    gamma: float = 0.0,
    *,
    intercept: float | None = None,
) -> float:
    """Return Δ′17O from already logarithmic delta-prime values."""
    if intercept is not None:
        if gamma != 0.0:
            raise ValueError("provide gamma or the legacy intercept alias, not both")
        gamma = -intercept
    return delta17_prime - slope * delta18_prime - gamma


def reduced_mass(m1: float, m2: float) -> float:
    """Reduced mass for a two-body reactant pair."""
    return (m1 * m2) / (m1 + m2)


def collision_frequency_alpha(mu_unsubstituted: float, mu_substituted: float) -> float:
    """Young Eq. 25: alpha_MD = sqrt(mu / mu')."""
    return (mu_unsubstituted / mu_substituted) ** 0.5


def atoms_mass(formula: dict[str, int]) -> float:
    """Return molecular mass from an isotope-specific atom count."""
    return sum(getattr(MASSES, iso) * count for iso, count in formula.items())


def molecular_masses() -> dict[str, float]:
    """Masses for species that occur in the Young reaction table."""
    return {
        "O": MASSES.O16,
        "17O": MASSES.O17,
        "18O": MASSES.O18,
        "O(1D)": MASSES.O16,
        "17O(1D)": MASSES.O17,
        "18O(1D)": MASSES.O18,
        "O2": atoms_mass({"O16": 2}),
        "O17O": atoms_mass({"O16": 1, "O17": 1}),
        "O18O": atoms_mass({"O16": 1, "O18": 1}),
        "O3": atoms_mass({"O16": 3}),
        "OO17O": atoms_mass({"O16": 2, "O17": 1}),
        "OO18O": atoms_mass({"O16": 2, "O18": 1}),
        "CO2": atoms_mass({"C12": 1, "O16": 2}),
        "CO17O": atoms_mass({"C12": 1, "O16": 1, "O17": 1}),
        "CO18O": atoms_mass({"C12": 1, "O16": 1, "O18": 1}),
    }


if __name__ == "__main__":
    # Basic checks from Young notation: modern model O2 troposphere gives
    # Δ′17O = about -0.410 per mil from Table 3 values.
    print(cap_delta17_from_primes(11.887, 23.212))
