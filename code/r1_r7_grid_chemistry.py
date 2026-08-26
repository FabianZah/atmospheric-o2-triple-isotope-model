"""Source-traceable R1-R7 rate fields for the gridded isotope model.

Young et al. (2014) used constants representative of 25 km and 220 K. This
module restores explicit temperature and collider dependence from Sander et
al. (2006). Young's R6 exchange coefficient is retained as a named compatibility
convention because the alternative Liang et al. (2006) expression has not yet
been reconciled with Young's reaction definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridded_oxygen_chemistry import ElementaryGridReaction
from isotopes import collision_frequency_alpha, molecular_masses, reduced_mass


@dataclass(frozen=True)
class RateField:
    key: str
    coefficient: np.ndarray
    units: str
    source: str
    status: str


MOLECULAR_MASSES = molecular_masses()


def _md_alpha(unsub_a: str, unsub_b: str, sub_a: str, sub_b: str) -> float:
    """Young Eq. 25 collision-frequency fractionation for a reactant pair."""
    mu_unsub = reduced_mass(MOLECULAR_MASSES[unsub_a], MOLECULAR_MASSES[unsub_b])
    mu_sub = reduced_mass(MOLECULAR_MASSES[sub_a], MOLECULAR_MASSES[sub_b])
    return collision_frequency_alpha(mu_unsub, mu_sub)


def _temperature(value: np.ndarray | float) -> np.ndarray:
    temperature = np.asarray(value, dtype=float)
    if np.any(temperature <= 0.0) or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must be finite and positive")
    return temperature


def r2_o_o2_m_cm6_per_molecule2_s(temperature_k: np.ndarray | float) -> np.ndarray:
    """O + O2 + M -> O3 + M; Sander et al. (2006), Table 2-1."""
    temperature = _temperature(temperature_k)
    return 6.0e-34 * (temperature / 300.0) ** -2.4


def r4_o_o3_cm3_per_molecule_s(temperature_k: np.ndarray | float) -> np.ndarray:
    """O + O3 -> 2 O2; Sander et al. (2006), Table 1-1."""
    temperature = _temperature(temperature_k)
    return 8.0e-12 * np.exp(-2060.0 / temperature)


def r4_o1d_o3_channel_cm3_per_molecule_s(temperature_k: np.ndarray | float) -> np.ndarray:
    """Each O(1D)+O3 product channel; Sander et al. (2006), Table 1-1."""
    return np.full_like(_temperature(temperature_k), 1.2e-10, dtype=float)


def r5_o1d_o2_cm3_per_molecule_s(temperature_k: np.ndarray | float) -> np.ndarray:
    temperature = _temperature(temperature_k)
    return 3.3e-11 * np.exp(55.0 / temperature)


def r5_o1d_n2_cm3_per_molecule_s(temperature_k: np.ndarray | float) -> np.ndarray:
    temperature = _temperature(temperature_k)
    return 2.15e-11 * np.exp(110.0 / temperature)


def r5_o1d_co2_cm3_per_molecule_s(temperature_k: np.ndarray | float) -> np.ndarray:
    """O(1D)+CO2 total physical quenching rate, also used for R7."""
    temperature = _temperature(temperature_k)
    return 7.5e-11 * np.exp(115.0 / temperature)


def modern_r1_r7_rate_fields(
    temperature_k: np.ndarray,
    number_density_air_cm3: np.ndarray,
    *,
    o2_mixing_ratio: np.ndarray | float,
    n2_mixing_ratio: np.ndarray | float = 0.7808,
    co2_mixing_ratio: np.ndarray | float = 400.0e-6,
) -> dict[str, RateField]:
    """Return cellwise kinetic fields without fitted or box-volume factors."""
    temperature = _temperature(temperature_k)
    air = np.asarray(number_density_air_cm3, dtype=float)
    if air.shape != temperature.shape or np.any(air <= 0.0) or not np.all(np.isfinite(air)):
        raise ValueError("air number density must match temperature and be positive")
    o2 = np.broadcast_to(np.asarray(o2_mixing_ratio, dtype=float), temperature.shape)
    n2 = np.broadcast_to(np.asarray(n2_mixing_ratio, dtype=float), temperature.shape)
    co2 = np.broadcast_to(np.asarray(co2_mixing_ratio, dtype=float), temperature.shape)
    if any(np.any(field < 0.0) for field in (o2, n2, co2)):
        raise ValueError("collider mixing ratios cannot be negative")

    r2 = r2_o_o2_m_cm6_per_molecule2_s(temperature) * air
    r5 = air * (
        r5_o1d_o2_cm3_per_molecule_s(temperature) * o2
        + r5_o1d_n2_cm3_per_molecule_s(temperature) * n2
        + r5_o1d_co2_cm3_per_molecule_s(temperature) * co2
    )
    return {
        "R2": RateField("R2", r2, "cm3 molecule-1 s-1", "Sander et al. (2006) k0(T) times local [M]", "source-derived"),
        "R4a": RateField("R4a", r4_o_o3_cm3_per_molecule_s(temperature), "cm3 molecule-1 s-1", "Sander et al. (2006), Table 1-1", "source-derived; differs from Young Table 2 at 220 K"),
        "R4f": RateField("R4f", r4_o1d_o3_channel_cm3_per_molecule_s(temperature), "cm3 molecule-1 s-1 per channel", "Sander et al. (2006), Table 1-1", "same value as Young R4f and R4k"),
        "R5": RateField("R5", r5, "s-1", "Sander et al. (2006) O2, N2, and CO2 quenching sum", "source-derived pseudo-first-order field"),
        "R6": RateField("R6", np.full_like(temperature, 2.0e-16), "cm3 molecule-1 s-1", "Young et al. (2014), Table 2", "compatibility convention; source reconciliation pending"),
        "R7": RateField("R7", r5_o1d_co2_cm3_per_molecule_s(temperature), "cm3 molecule-1 s-1", "Sander et al. (2006), O(1D)+CO2", "source-derived; Young-compatible isotope branching pending"),
    }


def basic_r1_r7_grid_reactions(
    rate_fields: dict[str, RateField],
    photolysis_fields: dict[str, np.ndarray],
) -> tuple[ElementaryGridReaction, ...]:
    """Build atom-balanced parent channels plus Young-compatible R6 exchange."""
    keys = (
        "j_r1_o2_to_o_o_per_s",
        "j_o2_to_o_o1d_per_s",
        "j_r3a_o3_to_o2_o_per_s",
        "j_r3f_o3_to_o2_o1d_per_s",
    )
    missing = [key for key in keys if key not in photolysis_fields]
    if missing:
        raise ValueError(f"missing photolysis fields: {missing}")
    return (
        ElementaryGridReaction("R1a", {"O2": 1}, {"O": 2}, photolysis_fields[keys[0]], "pinned vertical photolysis"),
        ElementaryGridReaction("R1d", {"O2": 1}, {"O": 1, "O1D": 1}, photolysis_fields[keys[1]], "pinned upper-atmosphere photolysis"),
        ElementaryGridReaction("R2a", {"O": 1, "O2": 1}, {"O3": 1}, rate_fields["R2"].coefficient, rate_fields["R2"].source),
        ElementaryGridReaction("R3a", {"O3": 1}, {"O2": 1, "O": 1}, photolysis_fields[keys[2]], "pinned vertical photolysis"),
        ElementaryGridReaction("R3f", {"O3": 1}, {"O2": 1, "O1D": 1}, photolysis_fields[keys[3]], "pinned vertical photolysis"),
        ElementaryGridReaction("R4a", {"O3": 1, "O": 1}, {"O2": 2}, rate_fields["R4a"].coefficient, rate_fields["R4a"].source),
        ElementaryGridReaction("R4f", {"O3": 1, "O1D": 1}, {"O2": 2}, rate_fields["R4f"].coefficient, rate_fields["R4f"].source),
        ElementaryGridReaction("R4k", {"O3": 1, "O1D": 1}, {"O2": 1, "O": 2}, rate_fields["R4f"].coefficient, rate_fields["R4f"].source),
        ElementaryGridReaction("R5a", {"O1D": 1}, {"O": 1}, rate_fields["R5"].coefficient, rate_fields["R5"].source),
        ElementaryGridReaction("R6f18", {"O2": 1, "O18": 1}, {"O18O": 1, "O": 1}, rate_fields["R6"].coefficient, rate_fields["R6"].source),
        ElementaryGridReaction("R6r18", {"O18O": 1, "O": 1}, {"O2": 1, "O18": 1}, 0.5 * rate_fields["R6"].coefficient, rate_fields["R6"].source),
        ElementaryGridReaction("R6f17", {"O2": 1, "O17": 1}, {"O17O": 1, "O": 1}, rate_fields["R6"].coefficient, rate_fields["R6"].source),
        ElementaryGridReaction("R6r17", {"O17O": 1, "O": 1}, {"O2": 1, "O17": 1}, 0.5 * rate_fields["R6"].coefficient, rate_fields["R6"].source),
        ElementaryGridReaction("R7a", {"CO2": 1, "O1D": 1}, {"CO2": 1, "O": 1}, rate_fields["R7"].coefficient, rate_fields["R7"].source),
    )



def full_young_r1_r7_grid_reactions(
    rate_fields: dict[str, RateField],
    photolysis_fields: dict[str, np.ndarray],
    *,
    a_mif: float = 1.065,
) -> tuple[ElementaryGridReaction, ...]:
    """Build all 53 isotope-resolved R1-R7 reactions in Young Table 2.

    Coefficients remain cellwise. Reaction topology, Eq. 25 reduced-mass
    factors, ozone MIF, and statistical branching follow Young et al. (2014).
    The additional upper-atmosphere O2 -> O + O(1D) channel in the basic
    builder is excluded because it is not part of Young's 53 reactions.
    """
    required = (
        "j_r1_o2_to_o_o_per_s",
        "j_r3a_o3_to_o2_o_per_s",
        "j_r3f_o3_to_o2_o1d_per_s",
    )
    missing = [key for key in required if key not in photolysis_fields]
    if missing:
        raise ValueError(f"missing photolysis fields: {missing}")
    if not np.isfinite(a_mif) or a_mif <= 0.0:
        raise ValueError("a_mif must be finite and positive")

    j1 = np.asarray(photolysis_fields[required[0]], dtype=float)
    j3a = np.asarray(photolysis_fields[required[1]], dtype=float)
    j3f = np.asarray(photolysis_fields[required[2]], dtype=float)
    k2 = rate_fields["R2"].coefficient
    k4a = rate_fields["R4a"].coefficient
    k4f = rate_fields["R4f"].coefficient
    k5 = rate_fields["R5"].coefficient
    k6 = rate_fields["R6"].coefficient
    k7 = rate_fields["R7"].coefficient
    photolysis_source = (
        "Young et al. (2014), Table 2 branching; cellwise photolysis field"
    )
    reactions: list[ElementaryGridReaction] = []

    def add(key, reactants, products, coefficient, source):
        reactions.append(
            ElementaryGridReaction(key, reactants, products, coefficient, source)
        )

    # R1: O2 photolysis.
    add("R1a", {"O2": 1}, {"O": 2}, j1, photolysis_source)
    add("R1b", {"O17O": 1}, {"O": 1, "O17": 1}, j1, photolysis_source)
    add("R1c", {"O18O": 1}, {"O": 1, "O18": 1}, j1, photolysis_source)

    # R2: isotope-resolved ozone formation, including Young's a_MIF.
    r2_source = rate_fields["R2"].source
    add("R2a", {"O": 1, "O2": 1}, {"O3": 1}, k2, r2_source)
    add("R2b", {"O18": 1, "O2": 1}, {"OO18O": 1},
        k2 * _md_alpha("O", "O2", "18O", "O2") * a_mif, r2_source)
    add("R2c", {"O17": 1, "O2": 1}, {"OO17O": 1},
        k2 * _md_alpha("O", "O2", "17O", "O2") * a_mif, r2_source)
    add("R2d", {"O": 1, "O18O": 1}, {"OO18O": 1},
        k2 * _md_alpha("O", "O2", "O", "O18O") * a_mif, r2_source)
    add("R2e", {"O": 1, "O17O": 1}, {"OO17O": 1},
        k2 * _md_alpha("O", "O2", "O", "O17O") * a_mif, r2_source)

    # R3: O3 photolysis; 1/3 and 2/3 are Young Table 2 footnote c.
    add("R3a", {"O3": 1}, {"O2": 1, "O": 1}, j3a, photolysis_source)
    add("R3b", {"OO18O": 1}, {"O2": 1, "O18": 1}, j3a / 3.0, photolysis_source)
    add("R3c", {"OO18O": 1}, {"O18O": 1, "O": 1}, 2.0 * j3a / 3.0, photolysis_source)
    add("R3d", {"OO17O": 1}, {"O2": 1, "O17": 1}, j3a / 3.0, photolysis_source)
    add("R3e", {"OO17O": 1}, {"O17O": 1, "O": 1}, 2.0 * j3a / 3.0, photolysis_source)
    add("R3f", {"O3": 1}, {"O2": 1, "O1D": 1}, j3f, photolysis_source)
    add("R3g", {"OO18O": 1}, {"O2": 1, "O18_1D": 1}, j3f / 3.0, photolysis_source)
    add("R3h", {"OO18O": 1}, {"O18O": 1, "O1D": 1}, 2.0 * j3f / 3.0, photolysis_source)
    add("R3i", {"OO17O": 1}, {"O2": 1, "O17_1D": 1}, j3f / 3.0, photolysis_source)
    add("R3j", {"OO17O": 1}, {"O17O": 1, "O1D": 1}, 2.0 * j3f / 3.0, photolysis_source)

    # R4a-e: O(3P) ozone destruction.
    r4a_source = rate_fields["R4a"].source
    add("R4a", {"O3": 1, "O": 1}, {"O2": 2}, k4a, r4a_source)
    add("R4b", {"OO18O": 1, "O": 1}, {"O2": 1, "O18O": 1},
        k4a * _md_alpha("O3", "O", "OO18O", "O"), r4a_source)
    add("R4c", {"OO17O": 1, "O": 1}, {"O2": 1, "O17O": 1},
        k4a * _md_alpha("O3", "O", "OO17O", "O"), r4a_source)
    add("R4d", {"O3": 1, "O18": 1}, {"O2": 1, "O18O": 1},
        k4a * _md_alpha("O3", "O", "O3", "18O"), r4a_source)
    add("R4e", {"O3": 1, "O17": 1}, {"O2": 1, "O17O": 1},
        k4a * _md_alpha("O3", "O", "O3", "17O"), r4a_source)

    # R4f-s: the two O(1D) + O3 product families.
    r4f_source = rate_fields["R4f"].source
    alpha_oo18 = _md_alpha("O3", "O(1D)", "OO18O", "O(1D)")
    alpha_oo17 = _md_alpha("O3", "O(1D)", "OO17O", "O(1D)")
    alpha_o18d = _md_alpha("O3", "O(1D)", "O3", "18O(1D)")
    alpha_o17d = _md_alpha("O3", "O(1D)", "O3", "17O(1D)")
    add("R4f", {"O3": 1, "O1D": 1}, {"O2": 2}, k4f, r4f_source)
    add("R4g", {"OO18O": 1, "O1D": 1}, {"O2": 1, "O18O": 1}, k4f * alpha_oo18, r4f_source)
    add("R4h", {"OO17O": 1, "O1D": 1}, {"O2": 1, "O17O": 1}, k4f * alpha_oo17, r4f_source)
    add("R4i", {"O3": 1, "O18_1D": 1}, {"O2": 1, "O18O": 1}, k4f * alpha_o18d, r4f_source)
    add("R4j", {"O3": 1, "O17_1D": 1}, {"O2": 1, "O17O": 1}, k4f * alpha_o17d, r4f_source)
    add("R4k", {"O3": 1, "O1D": 1}, {"O2": 1, "O": 2}, k4f, r4f_source)
    add("R4l", {"OO18O": 1, "O1D": 1}, {"O2": 1, "O": 1, "O18": 1}, 0.5 * k4f * alpha_oo18, r4f_source)
    add("R4m", {"OO18O": 1, "O1D": 1}, {"O18O": 1, "O": 2}, 0.5 * k4f * alpha_oo18, r4f_source)
    add("R4n", {"OO17O": 1, "O1D": 1}, {"O2": 1, "O": 1, "O17": 1}, 0.5 * k4f * alpha_oo17, r4f_source)
    add("R4o", {"OO17O": 1, "O1D": 1}, {"O17O": 1, "O": 2}, 0.5 * k4f * alpha_oo17, r4f_source)
    add("R4p", {"O3": 1, "O18_1D": 1}, {"O2": 1, "O": 1, "O18": 1}, 0.5 * k4f * alpha_o18d, r4f_source)
    add("R4q", {"O3": 1, "O18_1D": 1}, {"O18O": 1, "O": 2}, 0.5 * k4f * alpha_o18d, r4f_source)
    add("R4r", {"O3": 1, "O17_1D": 1}, {"O2": 1, "O": 1, "O17": 1}, 0.5 * k4f * alpha_o17d, r4f_source)
    add("R4s", {"O3": 1, "O17_1D": 1}, {"O17O": 1, "O": 2}, 0.5 * k4f * alpha_o17d, r4f_source)

    # R5: isotope-preserving O(1D) quenching.
    r5_source = rate_fields["R5"].source
    add("R5a", {"O1D": 1}, {"O": 1}, k5, r5_source)
    add("R5b", {"O18_1D": 1}, {"O18": 1}, k5, r5_source)
    add("R5c", {"O17_1D": 1}, {"O17": 1}, k5, r5_source)

    # R6: isotope exchange between atomic and molecular oxygen.
    r6_source = rate_fields["R6"].source
    add("R6a", {"O2": 1, "O18": 1}, {"O18O": 1, "O": 1},
        k6 * _md_alpha("O2", "O", "O2", "18O"), r6_source)
    add("R6b", {"O18O": 1, "O": 1}, {"O2": 1, "O18": 1},
        0.5 * k6 * _md_alpha("O2", "O", "O18O", "O"), r6_source)
    add("R6c", {"O2": 1, "O17": 1}, {"O17O": 1, "O": 1},
        k6 * _md_alpha("O2", "O", "O2", "17O"), r6_source)
    add("R6d", {"O17O": 1, "O": 1}, {"O2": 1, "O17": 1},
        0.5 * k6 * _md_alpha("O2", "O", "O17O", "O"), r6_source)

    # R7: O(1D)-CO2 isotope exchange and quenching branches.
    r7_source = rate_fields["R7"].source
    alpha_co2_18d = _md_alpha("CO2", "O(1D)", "CO2", "18O(1D)")
    alpha_co2_17d = _md_alpha("CO2", "O(1D)", "CO2", "17O(1D)")
    alpha_co18 = _md_alpha("CO2", "O(1D)", "CO18O", "O(1D)")
    alpha_co17 = _md_alpha("CO2", "O(1D)", "CO17O", "O(1D)")
    add("R7a", {"CO2": 1, "O1D": 1}, {"CO2": 1, "O": 1}, k7, r7_source)
    add("R7b", {"CO2": 1, "O18_1D": 1}, {"CO2": 1, "O18": 1}, 0.5 * k7 * alpha_co2_18d, r7_source)
    add("R7c", {"CO2": 1, "O18_1D": 1}, {"CO18O": 1, "O": 1}, 0.5 * k7 * alpha_co2_18d, r7_source)
    add("R7d", {"CO18O": 1, "O1D": 1}, {"CO2": 1, "O18": 1}, 0.5 * k7 * alpha_co18, r7_source)
    add("R7e", {"CO18O": 1, "O1D": 1}, {"CO18O": 1, "O": 1}, 0.5 * k7 * alpha_co18, r7_source)
    add("R7f", {"CO2": 1, "O17_1D": 1}, {"CO2": 1, "O17": 1}, 0.5 * k7 * alpha_co2_17d, r7_source)
    add("R7g", {"CO2": 1, "O17_1D": 1}, {"CO17O": 1, "O": 1}, 0.5 * k7 * alpha_co2_17d, r7_source)
    add("R7h", {"CO17O": 1, "O1D": 1}, {"CO2": 1, "O17": 1}, 0.5 * k7 * alpha_co17, r7_source)
    add("R7i", {"CO17O": 1, "O1D": 1}, {"CO17O": 1, "O": 1}, 0.5 * k7 * alpha_co17, r7_source)
    return tuple(reactions)
