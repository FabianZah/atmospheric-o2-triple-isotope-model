"""Reaction inventory for the Young et al. (2014) rebuild.

Only unambiguous reactions are executable here initially. Reactions that require
careful isotope-specific reduced-mass factors are listed as TODO records until
their multipliers are implemented and checked.
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
from dataclasses import dataclass, field
from typing import Mapping

from isotopes import R17_VSMOW, R18_VSMOW, collision_frequency_alpha, molecular_masses, reduced_mass
from reactions import Reaction
from young_model_inventory import PARAMETERS


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
CM3_RATE_TO_MOL_PER_YEAR = SECONDS_PER_YEAR * PARAMETERS["avogadro_over_strat_volume_cm_minus3"]
TABLE3_O2_TROP = 3.80e19
TABLE3_O18O_TROP = 1.59e17
TABLE3_O17O_TROP = 2.85e16
TABLE3_CO2_TROP = 5.29e16
TABLE3_CO18O_TROP = 1.13e14
TABLE3_CO17O_TROP = 2.00e13
TABLE3_O2_STRAT = 3.80e18
TABLE3_O18O_STRAT = 1.59e16
TABLE3_O17O_STRAT = 2.85e15
TABLE3_TOTAL_O2_STRAT = TABLE3_O2_STRAT + TABLE3_O18O_STRAT + TABLE3_O17O_STRAT
TABLE3_N2_STRAT = PARAMETERS["moles_stratosphere"] - TABLE3_TOTAL_O2_STRAT
R5_BASE_O2_QUENCH_CM3_S = 2.93e-11
R5_TABLE3_BALANCED_EFFECTIVE_MOLES = 5.589807e18


@dataclass(frozen=True)
class TodoReaction:
    key: str
    reason: str
    source: str


@dataclass(frozen=True)
class ReactionRecord:
    key: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    rate_rule: str
    source: str
    executable: bool = False
    note: str = ""


@dataclass(frozen=True)
class InverseSourceReaction:
    """A source with rate numerator / denominator_species.

    This is used only for the explicitly labelled diagnostic organic burial
    closure. It is not a general Young Table 2 mass-action reaction.
    """

    key: str
    denominator_species: str
    products: Mapping[str, float]
    numerator: float
    units: str
    note: str = ""
    reactants: Mapping[str, float] = field(default_factory=dict)

    def rate(self, y, index: Mapping[str, int]) -> float:
        return float(self.numerator / y[index[self.denominator_species]])

    def apply(self, dydt, rate: float, index: Mapping[str, int]) -> None:
        for species, coeff in self.products.items():
            dydt[index[species]] += coeff * rate


def lasaga_ohmoto_burial_relative(p_o2_pal: float, k_o2_uM: float = 20.0, dmin_uM: float = 15.0) -> float:
    """Organic burial efficiency relative to modern from Lasaga & Ohmoto (2002).

    This implements their printed Section 4 feedback in normalized form:
    deep-ocean [O2] from Eq. 64, Michaelis-Menten oxidation from Eq. 59, and
    the low-O2 sulfate-reduction cap from Eq. 67. It is an optional
    literature-improved closure, not the default Young shorthand.
    """
    deep_o2_uM = max(340.0 * p_o2_pal - 172.0, 0.0)
    if deep_o2_uM <= dmin_uM + 1.0e-9:
        return 0.021 / 0.003
    deep_today = 340.0 - 172.0
    f_today = deep_today / (k_o2_uM + deep_today)
    f = deep_o2_uM / (k_o2_uM + deep_o2_uM)
    return (0.003 ** (f / f_today)) / 0.003


@dataclass(frozen=True)
class LasagaOhmotoBurialReaction:
    """Modern-normalized organic burial source with Lasaga-Ohmoto pO2 feedback."""

    key: str
    products: Mapping[str, float]
    modern_rate: float
    units: str
    note: str = ""
    reactants: Mapping[str, float] = field(default_factory=dict)

    def rate(self, y, index: Mapping[str, int]) -> float:
        p_o2_pal = y[index["O2_trop"]] / TABLE3_O2_TROP
        return float(self.modern_rate * lasaga_ohmoto_burial_relative(p_o2_pal))

    def apply(self, dydt, rate: float, index: Mapping[str, int]) -> None:
        for species, coeff in self.products.items():
            dydt[index[species]] += coeff * rate


@dataclass(frozen=True)
class PALInverseGeosphereBurialReaction:
    """Finite-reservoir burial rate law using geosphere atoms as substrate."""

    key: str
    substrate_species: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_constant: float
    units: str
    note: str = ""

    def rate(self, y, index: Mapping[str, int]) -> float:
        p_o2_pal = y[index["O2_trop"]] / TABLE3_O2_TROP
        return float(self.rate_constant * y[index[self.substrate_species]] / p_o2_pal)

    def apply(self, dydt, rate: float, index: Mapping[str, int]) -> None:
        for species, coeff in self.reactants.items():
            dydt[index[species]] -= coeff * rate
        for species, coeff in self.products.items():
            dydt[index[species]] += coeff * rate


@dataclass(frozen=True)
class PseudoFirstOrderReservoirReaction:
    """Pseudo-first-order reaction with explicit bookkeeping stoichiometry.

    Young folds effectively infinite water abundances into R8 rate constants.
    This diagnostic lets the same pseudo-first-order rates move atoms into and
    out of the formal biosphere/hydrosphere species without making the rate law
    depend on the huge water reservoir.
    """

    key: str
    driver_species: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_constant: float
    units: str
    note: str = ""

    def rate(self, y, index: Mapping[str, int]) -> float:
        return float(self.rate_constant * y[index[self.driver_species]])

    def apply(self, dydt, rate: float, index: Mapping[str, int]) -> None:
        for species, coeff in self.reactants.items():
            dydt[index[species]] -= coeff * rate
        for species, coeff in self.products.items():
            dydt[index[species]] += coeff * rate


@dataclass(frozen=True)
class R5QuenchReaction:
    """R5 O(1D) quenching with an explicit collision-partner convention.

    Young Table 2 labels R5 as k_R5a[M], but the footnote says the value was
    adjusted from an O2 quenching rate so that k[O2] = k5a[M]. Keeping this as a
    separate rate law lets us test the possible [M] conventions directly.
    """

    key: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_constant: float
    units: str
    note: str
    partner_mode: str
    fixed_partner_moles: float | None = None
    co2_partner_factor: float = 0.0

    def partner_moles(self, y, index: Mapping[str, int]) -> float:
        if self.partner_mode == "fixed":
            if self.fixed_partner_moles is None:
                partner = float(PARAMETERS["moles_stratosphere"])
            else:
                partner = float(self.fixed_partner_moles)
            if self.co2_partner_factor == 0.0:
                return partner
            total_co2 = y[index["CO2_strat"]] + y[index["CO18O_strat"]] + y[index["CO17O_strat"]]
            return float(partner + self.co2_partner_factor * total_co2)
        total_o2 = (
            y[index["O2_strat"]]
            + y[index["O18O_strat"]]
            + y[index["O17O_strat"]]
        )
        if self.partner_mode in ("dynamic_air", "dynamic_air_calibrated"):
            partner = float(TABLE3_N2_STRAT + total_o2)
            if self.co2_partner_factor == 0.0:
                return partner
            total_co2 = y[index["CO2_strat"]] + y[index["CO18O_strat"]] + y[index["CO17O_strat"]]
            return float(partner + self.co2_partner_factor * total_co2)
        if self.partner_mode == "dynamic_o2":
            return float(total_o2)
        raise ValueError(f"unknown R5 partner mode: {self.partner_mode}")

    def rate(self, y, index: Mapping[str, int]) -> float:
        reactant, power = next(iter(self.reactants.items()))
        return float(self.rate_constant * self.partner_moles(y, index) * y[index[reactant]] ** power)

    def apply(self, dydt, rate: float, index: Mapping[str, int]) -> None:
        for species, coeff in self.reactants.items():
            dydt[index[species]] -= coeff * rate
        for species, coeff in self.products.items():
            dydt[index[species]] += coeff * rate


def per_second_to_per_year(k: float) -> float:
    return k * SECONDS_PER_YEAR


def cm3_s_to_mol_year(k: float) -> float:
    """Convert Young's second-order stratospheric rates to mol^-1 yr^-1."""
    return k * CM3_RATE_TO_MOL_PER_YEAR


MOLECULAR_MASSES = molecular_masses()


def md_alpha(unsub_a: str, unsub_b: str, sub_a: str, sub_b: str) -> float:
    """Mass-dependent alpha from Young Eq. 25 for a reactant pair."""
    mu_unsub = reduced_mass(MOLECULAR_MASSES[unsub_a], MOLECULAR_MASSES[unsub_b])
    mu_sub = reduced_mass(MOLECULAR_MASSES[sub_a], MOLECULAR_MASSES[sub_b])
    return collision_frequency_alpha(mu_unsub, mu_sub)


def second_order_rate(k_cm3_s: float) -> float:
    return cm3_s_to_mol_year(k_cm3_s)


def r5_quench_reactions(
    r5_collision_partner_moles: float | None = None,
    r5_collision_partner_mode: str = "fixed",
    r5_co2_partner_factor: float = 0.0,
) -> list[R5QuenchReaction]:
    if r5_collision_partner_mode == "dynamic_o2":
        k_r5a = second_order_rate(R5_BASE_O2_QUENCH_CM3_S)
        note = "dynamic O2-only quench using Young footnote base k=2.93e-11"
    elif r5_collision_partner_mode == "dynamic_air_calibrated":
        modern_air_moles = TABLE3_N2_STRAT + TABLE3_TOTAL_O2_STRAT
        scale = R5_TABLE3_BALANCED_EFFECTIVE_MOLES / modern_air_moles
        k_r5a = second_order_rate(PARAMETERS["k_R5a_cm3_s"] * scale)
        note = (
            "dynamic [M]=fixed N2 + solved stratospheric O2; "
            f"k scaled by {scale:.6g} to preserve modern effective Young R5"
        )
    else:
        k_r5a = second_order_rate(PARAMETERS["k_R5a_cm3_s"])
        if r5_collision_partner_mode == "dynamic_air":
            note = "dynamic [M]=fixed N2 + solved stratospheric O2 isotopologues"
        elif r5_collision_partner_moles is None:
            note = "fixed whole stratosphere [M] from printed Table 3 total"
        else:
            note = f"fixed diagnostic [M]={r5_collision_partner_moles:.6g} mol"
    return [
        R5QuenchReaction("R5a", {"O1D_strat": 1}, {"O_strat": 1}, k_r5a, "mol^-1 yr^-1", note, r5_collision_partner_mode, r5_collision_partner_moles, r5_co2_partner_factor),
        R5QuenchReaction("R5b", {"O18_1D_strat": 1}, {"O18_strat": 1}, k_r5a, "mol^-1 yr^-1", note, r5_collision_partner_mode, r5_collision_partner_moles, r5_co2_partner_factor),
        R5QuenchReaction("R5c", {"O17_1D_strat": 1}, {"O17_strat": 1}, k_r5a, "mol^-1 yr^-1", note, r5_collision_partner_mode, r5_collision_partner_moles, r5_co2_partner_factor),
    ]


def reduced_mass_stratosphere_reactions(
    r5_collision_partner_moles: float | None = None,
    r5_collision_partner_mode: str = "fixed",
    r5_co2_partner_factor: float = 0.0,
    a_mif: float | None = None,
) -> list[Reaction]:
    """Executable R2/R4/R6/R7 reactions whose rate rules use Young Eq. 25."""
    if a_mif is None:
        a_mif = PARAMETERS["a_MIF"]
    k_r2a = second_order_rate(PARAMETERS["k_R2a_base_cm3_s"])
    k_r4a = second_order_rate(PARAMETERS["k_R4a_cm3_s"])
    k_r4f = second_order_rate(PARAMETERS["k_R4f_cm3_s"])
    k_r6 = second_order_rate(PARAMETERS["k_R6_cm3_s"])
    k_r7a = second_order_rate(PARAMETERS["k_R7a_cm3_s"])

    reactions = [
        Reaction("R2a", {"O_strat": 1, "O2_strat": 1}, {"O3_strat": 1}, k_r2a, "mol^-1 yr^-1"),
        Reaction("R2b", {"O18_strat": 1, "O2_strat": 1}, {"OO18O_strat": 1}, k_r2a * md_alpha("O", "O2", "18O", "O2") * a_mif, "mol^-1 yr^-1"),
        Reaction("R2c", {"O17_strat": 1, "O2_strat": 1}, {"OO17O_strat": 1}, k_r2a * md_alpha("O", "O2", "17O", "O2") * a_mif, "mol^-1 yr^-1"),
        Reaction("R2d", {"O_strat": 1, "O18O_strat": 1}, {"OO18O_strat": 1}, k_r2a * md_alpha("O", "O2", "O", "O18O") * a_mif, "mol^-1 yr^-1"),
        Reaction("R2e", {"O_strat": 1, "O17O_strat": 1}, {"OO17O_strat": 1}, k_r2a * md_alpha("O", "O2", "O", "O17O") * a_mif, "mol^-1 yr^-1"),
        Reaction("R4a", {"O3_strat": 1, "O_strat": 1}, {"O2_strat": 2}, k_r4a, "mol^-1 yr^-1"),
        Reaction("R4b", {"OO18O_strat": 1, "O_strat": 1}, {"O2_strat": 1, "O18O_strat": 1}, k_r4a * md_alpha("O3", "O", "OO18O", "O"), "mol^-1 yr^-1"),
        Reaction("R4c", {"OO17O_strat": 1, "O_strat": 1}, {"O2_strat": 1, "O17O_strat": 1}, k_r4a * md_alpha("O3", "O", "OO17O", "O"), "mol^-1 yr^-1"),
        Reaction("R4d", {"O3_strat": 1, "O18_strat": 1}, {"O2_strat": 1, "O18O_strat": 1}, k_r4a * md_alpha("O3", "O", "O3", "18O"), "mol^-1 yr^-1"),
        Reaction("R4e", {"O3_strat": 1, "O17_strat": 1}, {"O2_strat": 1, "O17O_strat": 1}, k_r4a * md_alpha("O3", "O", "O3", "17O"), "mol^-1 yr^-1"),
        Reaction("R4f", {"O3_strat": 1, "O1D_strat": 1}, {"O2_strat": 2}, k_r4f, "mol^-1 yr^-1"),
        Reaction("R4g", {"OO18O_strat": 1, "O1D_strat": 1}, {"O2_strat": 1, "O18O_strat": 1}, k_r4f * md_alpha("O3", "O(1D)", "OO18O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4h", {"OO17O_strat": 1, "O1D_strat": 1}, {"O2_strat": 1, "O17O_strat": 1}, k_r4f * md_alpha("O3", "O(1D)", "OO17O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4i", {"O3_strat": 1, "O18_1D_strat": 1}, {"O2_strat": 1, "O18O_strat": 1}, k_r4f * md_alpha("O3", "O(1D)", "O3", "18O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4j", {"O3_strat": 1, "O17_1D_strat": 1}, {"O2_strat": 1, "O17O_strat": 1}, k_r4f * md_alpha("O3", "O(1D)", "O3", "17O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4k", {"O3_strat": 1, "O1D_strat": 1}, {"O2_strat": 1, "O_strat": 2}, k_r4f, "mol^-1 yr^-1"),
        Reaction("R4l", {"OO18O_strat": 1, "O1D_strat": 1}, {"O2_strat": 1, "O_strat": 1, "O18_strat": 1}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "OO18O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4m", {"OO18O_strat": 1, "O1D_strat": 1}, {"O18O_strat": 1, "O_strat": 2}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "OO18O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4n", {"OO17O_strat": 1, "O1D_strat": 1}, {"O2_strat": 1, "O_strat": 1, "O17_strat": 1}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "OO17O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4o", {"OO17O_strat": 1, "O1D_strat": 1}, {"O17O_strat": 1, "O_strat": 2}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "OO17O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4p", {"O3_strat": 1, "O18_1D_strat": 1}, {"O2_strat": 1, "O_strat": 1, "O18_strat": 1}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "O3", "18O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4q", {"O3_strat": 1, "O18_1D_strat": 1}, {"O18O_strat": 1, "O_strat": 2}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "O3", "18O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4r", {"O3_strat": 1, "O17_1D_strat": 1}, {"O2_strat": 1, "O_strat": 1, "O17_strat": 1}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "O3", "17O(1D)"), "mol^-1 yr^-1"),
        Reaction("R4s", {"O3_strat": 1, "O17_1D_strat": 1}, {"O17O_strat": 1, "O_strat": 2}, 0.5 * k_r4f * md_alpha("O3", "O(1D)", "O3", "17O(1D)"), "mol^-1 yr^-1"),
        *r5_quench_reactions(
            r5_collision_partner_moles=r5_collision_partner_moles,
            r5_collision_partner_mode=r5_collision_partner_mode,
            r5_co2_partner_factor=r5_co2_partner_factor,
        ),
        Reaction("R6a", {"O2_strat": 1, "O18_strat": 1}, {"O18O_strat": 1, "O_strat": 1}, k_r6 * md_alpha("O2", "O", "O2", "18O"), "mol^-1 yr^-1"),
        Reaction("R6b", {"O18O_strat": 1, "O_strat": 1}, {"O2_strat": 1, "O18_strat": 1}, 0.5 * k_r6 * md_alpha("O2", "O", "O18O", "O"), "mol^-1 yr^-1"),
        Reaction("R6c", {"O2_strat": 1, "O17_strat": 1}, {"O17O_strat": 1, "O_strat": 1}, k_r6 * md_alpha("O2", "O", "O2", "17O"), "mol^-1 yr^-1"),
        Reaction("R6d", {"O17O_strat": 1, "O_strat": 1}, {"O2_strat": 1, "O17_strat": 1}, 0.5 * k_r6 * md_alpha("O2", "O", "O17O", "O"), "mol^-1 yr^-1"),
        Reaction("R7a", {"CO2_strat": 1, "O1D_strat": 1}, {"CO2_strat": 1, "O_strat": 1}, k_r7a, "mol^-1 yr^-1"),
        Reaction("R7b", {"CO2_strat": 1, "O18_1D_strat": 1}, {"CO2_strat": 1, "O18_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO2", "18O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7c", {"CO2_strat": 1, "O18_1D_strat": 1}, {"CO18O_strat": 1, "O_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO2", "18O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7d", {"CO18O_strat": 1, "O1D_strat": 1}, {"CO2_strat": 1, "O18_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO18O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7e", {"CO18O_strat": 1, "O1D_strat": 1}, {"CO18O_strat": 1, "O_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO18O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7f", {"CO2_strat": 1, "O17_1D_strat": 1}, {"CO2_strat": 1, "O17_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO2", "17O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7g", {"CO2_strat": 1, "O17_1D_strat": 1}, {"CO17O_strat": 1, "O_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO2", "17O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7h", {"CO17O_strat": 1, "O1D_strat": 1}, {"CO2_strat": 1, "O17_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO17O", "O(1D)"), "mol^-1 yr^-1"),
        Reaction("R7i", {"CO17O_strat": 1, "O1D_strat": 1}, {"CO17O_strat": 1, "O_strat": 1}, 0.5 * k_r7a * md_alpha("CO2", "O(1D)", "CO17O", "O(1D)"), "mol^-1 yr^-1"),
    ]
    return reactions


def transport_reactions() -> list[Reaction]:
    """Stratosphere-troposphere first-order exchange for O2 and CO2 species."""
    k_st = PARAMETERS["k_ST_per_year"]
    k_ts = PARAMETERS["k_TS_per_year"]
    pairs = [
        ("O2_strat", "O2_trop"),
        ("O18O_strat", "O18O_trop"),
        ("O17O_strat", "O17O_trop"),
        ("CO2_strat", "CO2_trop"),
        ("CO18O_strat", "CO18O_trop"),
        ("CO17O_strat", "CO17O_trop"),
    ]
    reactions: list[Reaction] = []
    for strat, trop in pairs:
        reactions.append(Reaction(f"k_ST_{strat}", {strat: 1}, {trop: 1}, k_st, "yr^-1", "strat -> trop transport"))
        reactions.append(Reaction(f"k_TS_{trop}", {trop: 1}, {strat: 1}, k_ts, "yr^-1", "trop -> strat transport"))
    return reactions


def r8_co2_h2o_exchange_reactions(
    bookkeep_biosphere: bool = False,
    r8_rate_factor: float = 1.0,
) -> list[Reaction]:
    """Tropospheric CO2-H2O isotope exchange R8.

    Water reservoir terms are folded into pseudo-first-order rate constants as
    described by Young Table 2. The 17O term uses the expression printed in
    Table 2 with the 0.528 exponent. Young also discusses bwater=0.520 for
    photosynthetic source water; keep this distinction visible in validation.
    """
    k = PARAMETERS["k_R8b_per_year"] * r8_rate_factor
    alpha18 = PARAMETERS["alpha_CO2_H2O_18"]
    beta_eq = PARAMETERS["beta_CO2_H2O_17"]
    water_alpha18 = PARAMETERS["evapotranspiration_alpha_18"]
    k_r8a = k * alpha18 * (water_alpha18 * R18_VSMOW)
    k_r8c = k * (alpha18**beta_eq) * (water_alpha18**beta_eq * R17_VSMOW)
    if not bookkeep_biosphere:
        return [
            Reaction("R8a", {"CO2_trop": 1}, {"CO18O_trop": 1}, k_r8a, "yr^-1", "CO2 + H2-18O -> CO18O + H2O"),
            Reaction("R8b", {"CO18O_trop": 1}, {"CO2_trop": 1}, k, "yr^-1", "CO18O + H2O -> CO2 + H2-18O"),
            Reaction("R8c", {"CO2_trop": 1}, {"CO17O_trop": 1}, k_r8c, "yr^-1", "CO2 + H2-17O -> CO17O + H2O"),
            Reaction("R8d", {"CO17O_trop": 1}, {"CO2_trop": 1}, k, "yr^-1", "CO17O + H2O -> CO2 + H2-17O"),
        ]
    return [
        PseudoFirstOrderReservoirReaction(
            "R8a",
            "CO2_trop",
            {"CO2_trop": 1, "O18_bio": 1},
            {"CO18O_trop": 1, "O_bio": 1},
            k_r8a,
            "yr^-1",
            "CO2 + H2-18O -> CO18O + H2O; water in rate constant, biosphere bookkeeping explicit",
        ),
        PseudoFirstOrderReservoirReaction(
            "R8b",
            "CO18O_trop",
            {"CO18O_trop": 1, "O_bio": 1},
            {"CO2_trop": 1, "O18_bio": 1},
            k,
            "yr^-1",
            "CO18O + H2O -> CO2 + H2-18O; water in rate constant, biosphere bookkeeping explicit",
        ),
        PseudoFirstOrderReservoirReaction(
            "R8c",
            "CO2_trop",
            {"CO2_trop": 1, "O17_bio": 1},
            {"CO17O_trop": 1, "O_bio": 1},
            k_r8c,
            "yr^-1",
            "CO2 + H2-17O -> CO17O + H2O; water in rate constant, biosphere bookkeeping explicit",
        ),
        PseudoFirstOrderReservoirReaction(
            "R8d",
            "CO17O_trop",
            {"CO17O_trop": 1, "O_bio": 1},
            {"CO2_trop": 1, "O17_bio": 1},
            k,
            "yr^-1",
            "CO17O + H2O -> CO2 + H2-17O; water in rate constant, biosphere bookkeeping explicit",
        ),
    ]


def co2_source_sink_reactions(
    co2_source_override_mol_per_year: float | None = None,
    co2_source_isotope_mode: str = "smow",
    co2_sink_factor: float = 1.0,
    co2_ocean_infusion_factor: float = 1.0,
) -> list[Reaction]:
    """CO2 source and sink terms from Young Table 2 / Eq. 27."""
    f_v = PARAMETERS["f_volcanic_CO2_mol_per_year"]
    f_o = PARAMETERS["f_ocean_CO2_mol_per_year"]
    k_w = PARAMETERS["k_CO2_weathering_per_year"] * co2_sink_factor
    k_a = PARAMETERS["k_ocean_CO2_infusion_per_year"] * co2_sink_factor * co2_ocean_infusion_factor
    source = f_v + f_o if co2_source_override_mol_per_year is None else co2_source_override_mol_per_year
    source_note = (
        "volcanic + ocean CO2 source"
        if co2_source_override_mol_per_year is None
        else "diagnostic CO2 source adjusted to target steady pCO2"
    )
    if co2_source_isotope_mode == "smow":
        r18 = R18_VSMOW
        r17 = R17_VSMOW
    elif co2_source_isotope_mode == "printed_table3":
        r18 = 1.1048007385875107e14 / TABLE3_CO2_TROP
        r17 = 2.053553716895676e13 / TABLE3_CO2_TROP
    else:
        raise ValueError(f"unknown CO2 source isotope mode: {co2_source_isotope_mode}")
    return [
        Reaction("CO2_source", {}, {"CO2_trop": 1}, source, "mol yr^-1", source_note),
        Reaction("CO18O_source", {}, {"CO18O_trop": 1}, source * r18, "mol yr^-1", source_note + f"; source isotope mode={co2_source_isotope_mode}"),
        Reaction("CO17O_source", {}, {"CO17O_trop": 1}, source * r17, "mol yr^-1", source_note + f"; source isotope mode={co2_source_isotope_mode}"),
        Reaction("CO2_weathering", {"CO2_trop": 1}, {"O_geo": 2}, k_w, "yr^-1", f"CO2 drawdown by weathering into geosphere O; co2_sink_factor={co2_sink_factor:g}"),
        Reaction("CO18O_weathering", {"CO18O_trop": 1}, {"O_geo": 1, "O18_geo": 1}, k_w, "yr^-1", f"CO18O drawdown by weathering into geosphere O; co2_sink_factor={co2_sink_factor:g}"),
        Reaction("CO17O_weathering", {"CO17O_trop": 1}, {"O_geo": 1, "O17_geo": 1}, k_w, "yr^-1", f"CO17O drawdown by weathering into geosphere O; co2_sink_factor={co2_sink_factor:g}"),
        Reaction("CO2_ocean_infusion", {"CO2_trop": 1}, {}, k_a, "yr^-1", f"CO2 ocean infusion sink; co2_sink_factor={co2_sink_factor:g}; co2_ocean_infusion_factor={co2_ocean_infusion_factor:g}"),
        Reaction("CO18O_ocean_infusion", {"CO18O_trop": 1}, {}, k_a, "yr^-1", f"CO18O ocean infusion sink; co2_sink_factor={co2_sink_factor:g}; co2_ocean_infusion_factor={co2_ocean_infusion_factor:g}"),
        Reaction("CO17O_ocean_infusion", {"CO17O_trop": 1}, {}, k_a, "yr^-1", f"CO17O ocean infusion sink; co2_sink_factor={co2_sink_factor:g}; co2_ocean_infusion_factor={co2_ocean_infusion_factor:g}"),
    ]


def oxygen_biosphere_reactions(
    gpp_scale: float = 1.0,
    rp_o2: float | None = None,
    co2_photo_sink_factor: float = 1.0,
    co2_photo_sink_mode: str = "smow",
    alpha_respiration_18: float | None = None,
    beta_respiration_17: float | None = None,
    evapotranspiration_alpha_18: float | None = None,
    evapotranspiration_beta_17: float | None = None,
) -> list[Reaction]:
    """Respiration/photosynthesis terms for tropospheric O2.

    `rp_o2` is the 16O2 photosynthetic production flux in mol/year. If omitted,
    it is set to `kr * O2_trop(Table 3)`. `gpp_scale` follows Young's fixed-O2
    GPP experiments by scaling respiration and photosynthesis together.

    Young prints Eq. 27 only for C16O2. The analogous photosynthetic sinks for
    CO18O/CO17O are ambiguous in the paper text. The default `smow` mode keeps
    the conservative first-pass SMOW scaling. The diagnostic `o2_source_water`
    mode uses the same source-water isotope flux as photosynthetic O2
    production; this nearly balances printed Table 3 CO2 isotopologues.
    """
    kr = PARAMETERS["k_respiration_per_year"] * gpp_scale
    alpha18 = PARAMETERS["alpha_respiration_18"] if alpha_respiration_18 is None else alpha_respiration_18
    beta_resp = PARAMETERS["beta_respiration_17"] if beta_respiration_17 is None else beta_respiration_17
    source_alpha18 = (
        PARAMETERS["evapotranspiration_alpha_18"]
        if evapotranspiration_alpha_18 is None
        else evapotranspiration_alpha_18
    )
    source_beta17 = (
        PARAMETERS["evapotranspiration_beta_17"]
        if evapotranspiration_beta_17 is None
        else evapotranspiration_beta_17
    )
    if rp_o2 is None:
        rp_o2 = kr * 3.80e19
    if co2_photo_sink_mode == "smow":
        co18_photo_sink = rp_o2 * co2_photo_sink_factor * R18_VSMOW
        co17_photo_sink = rp_o2 * co2_photo_sink_factor * R17_VSMOW
    elif co2_photo_sink_mode == "o2_source_water":
        co18_photo_sink = rp_o2 * co2_photo_sink_factor * 2.0 * source_alpha18 * R18_VSMOW
        co17_photo_sink = rp_o2 * co2_photo_sink_factor * 2.0 * (source_alpha18**source_beta17) * R17_VSMOW
    else:
        raise ValueError(f"unknown CO2 photosynthesis sink mode: {co2_photo_sink_mode}")
    return [
        Reaction("resp_O2", {"O2_trop": 1}, {"CO2_trop": 1}, kr, "yr^-1", "respiration consumes O2 and produces CO2"),
        Reaction("resp_O18O", {"O18O_trop": 1}, {"CO18O_trop": 1}, alpha18 * kr, "yr^-1"),
        Reaction("resp_O17O", {"O17O_trop": 1}, {"CO17O_trop": 1}, (alpha18**beta_resp) * kr, "yr^-1"),
        Reaction("photo_CO2_sink", {}, {"CO2_trop": -1}, rp_o2, "mol yr^-1", "photosynthetic CO2 sink from Young Eq. 27"),
        Reaction("photo_CO18O_sink", {}, {"CO18O_trop": -1}, co18_photo_sink, "mol yr^-1", f"CO18O photosynthetic sink mode={co2_photo_sink_mode}"),
        Reaction("photo_CO17O_sink", {}, {"CO17O_trop": -1}, co17_photo_sink, "mol yr^-1", f"CO17O photosynthetic sink mode={co2_photo_sink_mode}"),
        Reaction("photo_O2", {}, {"O2_trop": 1}, rp_o2, "mol yr^-1", "photosynthesis 16O2 source"),
        Reaction("photo_O18O", {}, {"O18O_trop": 1}, rp_o2 * 2.0 * source_alpha18 * R18_VSMOW, "mol yr^-1", "photosynthetic source water"),
        Reaction("photo_O17O", {}, {"O17O_trop": 1}, rp_o2 * 2.0 * (source_alpha18**source_beta17) * R17_VSMOW, "mol yr^-1", "photosynthetic source water"),
    ]


def organic_burial_closure_reactions() -> list[Reaction]:
    """Diagnostic modern closure for Young's unprinted organic burial term.

    Young gives korg and states that organic burial is an effective O2
    production feedback depending on 1/nO2, but does not print the exact flux
    equation. This closure only balances the explicit modern O2 weathering sink
    at Table 3 and scales as 1/nO2, matching the qualitative statement. It is
    not enabled by default and should not be treated as the exact Young model.
    """
    k_weather = PARAMETERS["k_O2_weathering_per_year"]
    return [
        InverseSourceReaction(
            "organic_burial_closure_O2",
            "O2_trop",
            {"O2_trop": 1},
            k_weather * TABLE3_O2_TROP * TABLE3_O2_TROP,
            "mol^2 yr^-1",
            "diagnostic closure: balances modern O2 weathering, scales as 1/nO2",
        ),
        InverseSourceReaction(
            "organic_burial_closure_O18O",
            "O2_trop",
            {"O18O_trop": 1},
            k_weather * TABLE3_O18O_TROP * TABLE3_O2_TROP,
            "mol^2 yr^-1",
            "diagnostic closure scaled from modern O18O weathering",
        ),
        InverseSourceReaction(
            "organic_burial_closure_O17O",
            "O2_trop",
            {"O17O_trop": 1},
            k_weather * TABLE3_O17O_TROP * TABLE3_O2_TROP,
            "mol^2 yr^-1",
            "diagnostic closure scaled from modern O17O weathering",
        ),
    ]


def lasaga_ohmoto_burial_closure_reactions() -> list[Reaction]:
    """Organic burial closure with Lasaga-Ohmoto pO2-dependent efficiency.

    The modern flux is normalized to balance the explicit modern O2 weathering
    sink. Isotopologue source ratios are held at the modern atmospheric ratios,
    matching the current Young-style closure convention while replacing only
    the pO2 feedback shape.
    """
    k_weather = PARAMETERS["k_O2_weathering_per_year"]
    return [
        LasagaOhmotoBurialReaction(
            "organic_burial_lasaga_O2",
            {"O2_trop": 1},
            k_weather * TABLE3_O2_TROP,
            "mol yr^-1",
            "literature-improved closure: Lasaga-Ohmoto organic burial pO2 feedback",
        ),
        LasagaOhmotoBurialReaction(
            "organic_burial_lasaga_O18O",
            {"O18O_trop": 1},
            k_weather * TABLE3_O18O_TROP,
            "mol yr^-1",
            "Lasaga-Ohmoto closure scaled from modern O18O weathering",
        ),
        LasagaOhmotoBurialReaction(
            "organic_burial_lasaga_O17O",
            {"O17O_trop": 1},
            k_weather * TABLE3_O17O_TROP,
            "mol yr^-1",
            "Lasaga-Ohmoto closure scaled from modern O17O weathering",
        ),
    ]


def finite_geosphere_organic_burial_reactions() -> list[Reaction]:
    """Candidate finite-geosphere replacement for direct organic burial closure.

    This is a diagnostic variant for Young's unprinted slow geological ODEs.
    The rate law treats `1/nO2` as `1/pO2_PAL`. O_geo is an atom reservoir, so
    the O2-producing branch uses half of O_geo as the substrate flux.
    """
    k = PARAMETERS["k_organic_burial_per_year"]
    return [
        PALInverseGeosphereBurialReaction(
            "organic_burial_geo_O2",
            "O_geo",
            {"O_geo": 2},
            {"O2_trop": 1},
            0.5 * k,
            "yr^-1",
            "diagnostic finite geosphere organic burial, rate=(0.5*korg*O_geo)/pO2_PAL",
        ),
        PALInverseGeosphereBurialReaction(
            "organic_burial_geo_O18O",
            "O18_geo",
            {"O_geo": 1, "O18_geo": 1},
            {"O18O_trop": 1},
            k,
            "yr^-1",
            "diagnostic finite geosphere organic burial, heavy branch scaled by O18_geo/pO2_PAL",
        ),
        PALInverseGeosphereBurialReaction(
            "organic_burial_geo_O17O",
            "O17_geo",
            {"O_geo": 1, "O17_geo": 1},
            {"O17O_trop": 1},
            k,
            "yr^-1",
            "diagnostic finite geosphere organic burial, heavy branch scaled by O17_geo/pO2_PAL",
        ),
    ]


def co2_modern_closure_reactions(
    isotope_mode: str = "printed_table3",
    co2_sink_factor: float = 1.0,
    co2_ocean_infusion_factor: float = 1.0,
) -> list[Reaction]:
    """Diagnostic CO2 closure that balances residual modern CO2 sinks/sources.

    This is a numerical closure for solver development, not a paper-derived
    Young term. It offsets the non-biospheric CO2 imbalance at Table 3 after
    Eq. 27 terms are applied.
    """
    source = PARAMETERS["f_volcanic_CO2_mol_per_year"] + PARAMETERS["f_ocean_CO2_mol_per_year"]
    sink = (
        PARAMETERS["k_CO2_weathering_per_year"]
        + PARAMETERS["k_ocean_CO2_infusion_per_year"] * co2_ocean_infusion_factor
    ) * co2_sink_factor * TABLE3_CO2_TROP
    net_sink_to_balance = sink - source
    if isotope_mode == "rounded_table3":
        r18 = TABLE3_CO18O_TROP / TABLE3_CO2_TROP
        r17 = TABLE3_CO17O_TROP / TABLE3_CO2_TROP
    elif isotope_mode == "printed_table3":
        # Ratios implied by Young's printed delta-prime targets, computed once
        # by model_runner.scaled_table3_state and copied here to avoid a
        # circular import.
        r18 = 1.1048007385875107e14 / TABLE3_CO2_TROP
        r17 = 2.053553716895676e13 / TABLE3_CO2_TROP
    elif isotope_mode == "smow":
        r18 = R18_VSMOW
        r17 = R17_VSMOW
    else:
        raise ValueError(f"unknown CO2 closure isotope mode: {isotope_mode}")
    return [
        Reaction("co2_modern_closure", {}, {"CO2_trop": 1}, net_sink_to_balance, "mol yr^-1", f"diagnostic modern CO2 closure; co2_sink_factor={co2_sink_factor:g}; co2_ocean_infusion_factor={co2_ocean_infusion_factor:g}"),
        Reaction("co18o_modern_closure", {}, {"CO18O_trop": 1}, net_sink_to_balance * r18, "mol yr^-1", f"diagnostic modern CO18O closure isotope_mode={isotope_mode}; co2_sink_factor={co2_sink_factor:g}; co2_ocean_infusion_factor={co2_ocean_infusion_factor:g}"),
        Reaction("co17o_modern_closure", {}, {"CO17O_trop": 1}, net_sink_to_balance * r17, "mol yr^-1", f"diagnostic modern CO17O closure isotope_mode={isotope_mode}; co2_sink_factor={co2_sink_factor:g}; co2_ocean_infusion_factor={co2_ocean_infusion_factor:g}"),
    ]


def diagnostic_closure_reactions(
    mode: str = "none",
    co2_isotope_mode: str = "printed_table3",
    co2_sink_factor: float = 1.0,
    co2_ocean_infusion_factor: float = 1.0,
) -> list[Reaction]:
    if mode == "none":
        return []
    if mode == "organic_burial":
        return organic_burial_closure_reactions()
    if mode == "lasaga_burial":
        return lasaga_ohmoto_burial_closure_reactions()
    if mode == "finite_geosphere_burial":
        return finite_geosphere_organic_burial_reactions()
    if mode == "modern":
        return organic_burial_closure_reactions() + co2_modern_closure_reactions(isotope_mode=co2_isotope_mode, co2_sink_factor=co2_sink_factor, co2_ocean_infusion_factor=co2_ocean_infusion_factor)
    if mode == "modern_lasaga_burial":
        return lasaga_ohmoto_burial_closure_reactions() + co2_modern_closure_reactions(isotope_mode=co2_isotope_mode, co2_sink_factor=co2_sink_factor, co2_ocean_infusion_factor=co2_ocean_infusion_factor)
    if mode == "modern_finite_geosphere":
        return finite_geosphere_organic_burial_reactions() + co2_modern_closure_reactions(isotope_mode=co2_isotope_mode, co2_sink_factor=co2_sink_factor, co2_ocean_infusion_factor=co2_ocean_infusion_factor)
    raise ValueError(f"unknown closure mode: {mode}")


def geological_oxygen_reactions() -> list[Reaction]:
    """Explicit O2 weathering uptake by the geosphere from Young Section 3.6.

    Young gives `kO2-w = 6.0e-7 yr^-1` and describes this term as O2 weathering
    uptake by the geosphere. No isotope-specific weathering fractionation is
    printed, so the isotopologue reactions use the same first-order constant
    and simply move oxygen atoms into the geosphere bookkeeping reservoirs.

    Organic burial is deliberately not included here yet: Young states that it
    depends on 1/nO2, but the exact flux expression/normalization is not printed.
    """
    k = PARAMETERS["k_O2_weathering_per_year"]
    return [
        Reaction("O2_weathering_geo", {"O2_trop": 1}, {"O_geo": 2}, k, "yr^-1", "O2 weathering uptake by geosphere"),
        Reaction("O18O_weathering_geo", {"O18O_trop": 1}, {"O_geo": 1, "O18_geo": 1}, k, "yr^-1", "O18O weathering uptake by geosphere"),
        Reaction("O17O_weathering_geo", {"O17O_trop": 1}, {"O_geo": 1, "O17_geo": 1}, k, "yr^-1", "O17O weathering uptake by geosphere"),
    ]


def reservoir_reactions(
    gpp_scale: float = 1.0,
    rp_o2: float | None = None,
    closure_mode: str = "none",
    co2_photo_sink_factor: float = 1.0,
    co2_photo_sink_mode: str = "smow",
    co2_closure_isotope_mode: str = "printed_table3",
    co2_source_override_mol_per_year: float | None = None,
    co2_source_isotope_mode: str = "smow",
    co2_sink_factor: float = 1.0,
    co2_ocean_infusion_factor: float = 1.0,
    r8_biosphere_bookkeeping: bool = False,
    r8_rate_factor: float = 1.0,
    alpha_respiration_18: float | None = None,
    beta_respiration_17: float | None = None,
    evapotranspiration_alpha_18: float | None = None,
    evapotranspiration_beta_17: float | None = None,
) -> list[Reaction]:
    return (
        transport_reactions()
        + r8_co2_h2o_exchange_reactions(
            bookkeep_biosphere=r8_biosphere_bookkeeping,
            r8_rate_factor=r8_rate_factor,
        )
        + co2_source_sink_reactions(
            co2_source_override_mol_per_year=co2_source_override_mol_per_year,
            co2_source_isotope_mode=co2_source_isotope_mode,
            co2_sink_factor=co2_sink_factor,
            co2_ocean_infusion_factor=co2_ocean_infusion_factor,
        )
        + oxygen_biosphere_reactions(
            gpp_scale=gpp_scale,
            rp_o2=rp_o2,
            co2_photo_sink_factor=co2_photo_sink_factor,
            co2_photo_sink_mode=co2_photo_sink_mode,
            alpha_respiration_18=alpha_respiration_18,
            beta_respiration_17=beta_respiration_17,
            evapotranspiration_alpha_18=evapotranspiration_alpha_18,
            evapotranspiration_beta_17=evapotranspiration_beta_17,
        )
        + geological_oxygen_reactions()
        + diagnostic_closure_reactions(
            closure_mode,
            co2_isotope_mode=co2_closure_isotope_mode,
            co2_sink_factor=co2_sink_factor,
            co2_ocean_infusion_factor=co2_ocean_infusion_factor,
        )
    )


def stratosphere_photolysis_reactions() -> list[Reaction]:
    """Unambiguous first-order stratospheric photolysis reactions R1 and R3."""
    j_r1 = per_second_to_per_year(PARAMETERS["J_R1a_per_s"])
    j_r3a = per_second_to_per_year(PARAMETERS["J_R3a_per_s"])
    j_r3f = per_second_to_per_year(PARAMETERS["J_R3f_per_s"])
    return [
        Reaction("R1a", {"O2_strat": 1}, {"O_strat": 2}, j_r1, "yr^-1", "O2 + hv -> O + O"),
        Reaction("R1b", {"O17O_strat": 1}, {"O_strat": 1, "O17_strat": 1}, j_r1, "yr^-1", "O17O + hv -> O + 17O"),
        Reaction("R1c", {"O18O_strat": 1}, {"O_strat": 1, "O18_strat": 1}, j_r1, "yr^-1", "O18O + hv -> O + 18O"),
        Reaction("R3a", {"O3_strat": 1}, {"O2_strat": 1, "O_strat": 1}, j_r3a, "yr^-1", "O3 + hv -> O2 + O"),
        Reaction("R3b", {"OO18O_strat": 1}, {"O2_strat": 1, "O18_strat": 1}, j_r3a / 3.0, "yr^-1", "OO18O + hv -> O2 + 18O"),
        Reaction("R3c", {"OO18O_strat": 1}, {"O18O_strat": 1, "O_strat": 1}, 2.0 * j_r3a / 3.0, "yr^-1", "OO18O + hv -> O18O + O"),
        Reaction("R3d", {"OO17O_strat": 1}, {"O2_strat": 1, "O17_strat": 1}, j_r3a / 3.0, "yr^-1", "OO17O + hv -> O2 + 17O"),
        Reaction("R3e", {"OO17O_strat": 1}, {"O17O_strat": 1, "O_strat": 1}, 2.0 * j_r3a / 3.0, "yr^-1", "OO17O + hv -> O17O + O"),
        Reaction("R3f", {"O3_strat": 1}, {"O2_strat": 1, "O1D_strat": 1}, j_r3f, "yr^-1", "O3 + hv -> O2 + O(1D)"),
        Reaction("R3g", {"OO18O_strat": 1}, {"O2_strat": 1, "O18_1D_strat": 1}, j_r3f / 3.0, "yr^-1", "OO18O + hv -> O2 + 18O(1D)"),
        Reaction("R3h", {"OO18O_strat": 1}, {"O18O_strat": 1, "O1D_strat": 1}, 2.0 * j_r3f / 3.0, "yr^-1", "OO18O + hv -> O18O + O(1D)"),
        Reaction("R3i", {"OO17O_strat": 1}, {"O2_strat": 1, "O17_1D_strat": 1}, j_r3f / 3.0, "yr^-1", "OO17O + hv -> O2 + 17O(1D)"),
        Reaction("R3j", {"OO17O_strat": 1}, {"O17O_strat": 1, "O1D_strat": 1}, 2.0 * j_r3f / 3.0, "yr^-1", "OO17O + hv -> O17O + O(1D)"),
    ]


REACTION_RECORDS = [
    ReactionRecord("R1a", ("O2_strat",), ("O_strat", "O_strat"), "J_R1a", "Table 2", True),
    ReactionRecord("R1b", ("O17O_strat",), ("O_strat", "O17_strat"), "J_R1a", "Table 2", True),
    ReactionRecord("R1c", ("O18O_strat",), ("O_strat", "O18_strat"), "J_R1a", "Table 2", True),
    ReactionRecord("R2a", ("O_strat", "O2_strat"), ("O3_strat",), "k_R2a", "Table 2", False, "three-body rate already folded into k_R2a using n=8.3e17 cm^-3"),
    ReactionRecord("R2b", ("O18_strat", "O2_strat"), ("OO18O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(18O+O2)) * aMIF", "Table 2 Eq.25"),
    ReactionRecord("R2c", ("O17_strat", "O2_strat"), ("OO17O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(17O+O2)) * aMIF", "Table 2 Eq.25"),
    ReactionRecord("R2d", ("O_strat", "O18O_strat"), ("OO18O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(O+O18O)) * aMIF", "Table 2 Eq.25"),
    ReactionRecord("R2e", ("O_strat", "O17O_strat"), ("OO17O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(O+O17O)) * aMIF", "Table 2 Eq.25"),
    ReactionRecord("R3a", ("O3_strat",), ("O2_strat", "O_strat"), "J_R3a", "Table 2", True),
    ReactionRecord("R3b", ("OO18O_strat",), ("O2_strat", "O18_strat"), "1/3 * J_R3a", "Table 2", True),
    ReactionRecord("R3c", ("OO18O_strat",), ("O18O_strat", "O_strat"), "2/3 * J_R3a", "Table 2", True),
    ReactionRecord("R3d", ("OO17O_strat",), ("O2_strat", "O17_strat"), "1/3 * J_R3a", "Table 2", True),
    ReactionRecord("R3e", ("OO17O_strat",), ("O17O_strat", "O_strat"), "2/3 * J_R3a", "Table 2", True),
    ReactionRecord("R3f", ("O3_strat",), ("O2_strat", "O1D_strat"), "J_R3f", "Table 2", True),
    ReactionRecord("R3g", ("OO18O_strat",), ("O2_strat", "O18_1D_strat"), "1/3 * J_R3f", "Table 2", True),
    ReactionRecord("R3h", ("OO18O_strat",), ("O18O_strat", "O1D_strat"), "2/3 * J_R3f", "Table 2", True),
    ReactionRecord("R3i", ("OO17O_strat",), ("O2_strat", "O17_1D_strat"), "1/3 * J_R3f", "Table 2", True),
    ReactionRecord("R3j", ("OO17O_strat",), ("O17O_strat", "O1D_strat"), "2/3 * J_R3f", "Table 2", True),
    ReactionRecord("R4a", ("O3_strat", "O_strat"), ("O2_strat", "O2_strat"), "k_R4a", "Table 2"),
    ReactionRecord("R4b", ("OO18O_strat", "O_strat"), ("O2_strat", "O18O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(OO18O+O))", "Table 2 Eq.25"),
    ReactionRecord("R4c", ("OO17O_strat", "O_strat"), ("O2_strat", "O17O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(OO17O+O))", "Table 2 Eq.25"),
    ReactionRecord("R4d", ("O3_strat", "O18_strat"), ("O2_strat", "O18O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(O3+18O))", "Table 2 Eq.25"),
    ReactionRecord("R4e", ("O3_strat", "O17_strat"), ("O2_strat", "O17O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(O3+17O))", "Table 2 Eq.25"),
    ReactionRecord("R4f", ("O3_strat", "O1D_strat"), ("O2_strat", "O2_strat"), "k_R4f", "Table 2"),
    ReactionRecord("R4g", ("OO18O_strat", "O1D_strat"), ("O2_strat", "O18O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(OO18O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4h", ("OO17O_strat", "O1D_strat"), ("O2_strat", "O17O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(OO17O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4i", ("O3_strat", "O18_1D_strat"), ("O2_strat", "O18O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(O3+18O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4j", ("O3_strat", "O17_1D_strat"), ("O2_strat", "O17O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(O3+17O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4k", ("O3_strat", "O1D_strat"), ("O2_strat", "O_strat", "O_strat"), "k_R4f", "Table 2"),
    ReactionRecord("R4l", ("OO18O_strat", "O1D_strat"), ("O2_strat", "O_strat", "O18_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO18O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4m", ("OO18O_strat", "O1D_strat"), ("O18O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO18O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4n", ("OO17O_strat", "O1D_strat"), ("O2_strat", "O_strat", "O17_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO17O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4o", ("OO17O_strat", "O1D_strat"), ("O17O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO17O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4p", ("O3_strat", "O18_1D_strat"), ("O2_strat", "O_strat", "O18_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+18O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4q", ("O3_strat", "O18_1D_strat"), ("O18O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+18O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4r", ("O3_strat", "O17_1D_strat"), ("O2_strat", "O_strat", "O17_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+17O1D))", "Table 2 Eq.25"),
    ReactionRecord("R4s", ("O3_strat", "O17_1D_strat"), ("O17O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+17O1D))", "Table 2 Eq.25"),
    ReactionRecord("R5a", ("O1D_strat",), ("O_strat",), "k_R5a * nM", "Table 2", False, "pseudo-first-order if nM fixed; paper adjusts k so k[O2]=k5a[M]"),
    ReactionRecord("R5b", ("O18_1D_strat",), ("O18_strat",), "k_R5a * nM", "Table 2"),
    ReactionRecord("R5c", ("O17_1D_strat",), ("O17_strat",), "k_R5a * nM", "Table 2"),
    ReactionRecord("R6a", ("O2_strat", "O18_strat"), ("O18O_strat", "O_strat"), "k_R6 * sqrt(mu(O2+O)/mu(O2+18O))", "Table 2 Eq.25"),
    ReactionRecord("R6b", ("O18O_strat", "O_strat"), ("O2_strat", "O18_strat"), "1/2 * k_R6 * sqrt(mu(O2+O)/mu(O18O+O))", "Table 2 Eq.25"),
    ReactionRecord("R6c", ("O2_strat", "O17_strat"), ("O17O_strat", "O_strat"), "k_R6 * sqrt(mu(O2+O)/mu(O2+17O))", "Table 2 Eq.25"),
    ReactionRecord("R6d", ("O17O_strat", "O_strat"), ("O2_strat", "O17_strat"), "1/2 * k_R6 * sqrt(mu(O2+O)/mu(O17O+O))", "Table 2 Eq.25"),
    ReactionRecord("R7a", ("CO2_strat", "O1D_strat"), ("CO2_strat", "O_strat"), "k_R7a", "Table 2"),
    ReactionRecord("R7b", ("CO2_strat", "O18_1D_strat"), ("CO2_strat", "O18_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+18O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7c", ("CO2_strat", "O18_1D_strat"), ("CO18O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+18O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7d", ("CO18O_strat", "O1D_strat"), ("CO2_strat", "O18_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO18O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7e", ("CO18O_strat", "O1D_strat"), ("CO18O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO18O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7f", ("CO2_strat", "O17_1D_strat"), ("CO2_strat", "O17_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+17O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7g", ("CO2_strat", "O17_1D_strat"), ("CO17O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+17O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7h", ("CO17O_strat", "O1D_strat"), ("CO2_strat", "O17_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO17O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R7i", ("CO17O_strat", "O1D_strat"), ("CO17O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO17O+O1D))", "Table 2 Eq.25"),
    ReactionRecord("R8a", ("CO2_trop",), ("CO18O_trop",), "k_R8b * alpha_CO2_H2O_18 * (1.00525 * R18_SMOW)", "Table 2", False, "water reservoir folded into rate constant"),
    ReactionRecord("R8b", ("CO18O_trop",), ("CO2_trop",), "k_R8b", "Table 2"),
    ReactionRecord("R8c", ("CO2_trop",), ("CO17O_trop",), "k_R8b * alpha_CO2_H2O_18**0.528 * (1.00525**0.528 * R17_SMOW)", "Table 2", False, "paper text also mentions b=0.528 for equilibrium and 0.520 for evapotranspiration source water; verify exponent mix"),
    ReactionRecord("R8d", ("CO17O_trop",), ("CO2_trop",), "k_R8b", "Table 2"),
    ReactionRecord("k_ST_O2", ("O2_strat",), ("O2_trop",), "k_ST", "Section 3.3", False, "stratosphere to troposphere mixing; isotopologues analogous"),
    ReactionRecord("k_TS_O2", ("O2_trop",), ("O2_strat",), "k_TS", "Section 3.3", False, "troposphere to stratosphere mixing; isotopologues analogous"),
    ReactionRecord("resp_O2", ("O2_trop",), ("CO2_trop",), "k_respiration", "Sections 3.4/3.5", False, "respiration consumes O2 and produces CO2; isotope-specific rates use alpha_respiration"),
    ReactionRecord("photo_O2", tuple(), ("O2_trop",), "r_p", "Section 3.4", False, "photosynthesis flux; isotope-specific source-water terms required"),
    ReactionRecord("CO2_volcanic", tuple(), ("CO2_trop",), "f_volcanic_CO2", "Table 2", False, "isotopologues scaled by SMOW ratios"),
    ReactionRecord("CO2_ocean_effusion", tuple(), ("CO2_trop",), "f_ocean_CO2", "Table 2", False, "isotopologues scaled by SMOW ratios"),
    ReactionRecord("CO2_weathering", ("CO2_trop",), tuple(), "k_CO2_weathering", "Table 2"),
    ReactionRecord("CO2_ocean_infusion", ("CO2_trop",), tuple(), "k_ocean_CO2_infusion", "Table 2"),
    ReactionRecord("O2_weathering", ("O2_trop",), ("O_geo",), "k_O2_weathering", "Table 2/Section 3.6", False, "geosphere stoichiometry/isotopologues need definition"),
    ReactionRecord("organic_burial", ("O_geo",), ("O2_trop",), "k_organic_burial / nO2", "Table 2/Section 3.6", False, "paper says burial depends on 1/nO2; exact equation not printed"),
]


TODO_REACTIONS = [
    TodoReaction("R8a-R8d", "Need SMOW ratios and water isotope terms encoded exactly.", "Young Table 2"),
    TodoReaction("respiration/photosynthesis", "Need full reservoir stoichiometry, source-water ratios, and rp/kr controls.", "Young Sections 3.4, 3.6"),
    TodoReaction("organic burial", "Need exact 1/nO2 organic burial flux expression and normalization.", "Young Section 3.6"),
    TodoReaction("transport", "Need stratosphere/troposphere transfer reactions for all relevant isotopologues.", "Young Section 3.3"),
    TodoReaction("CO2 sources/sinks", "Need isotopologue-specific volcanic/ocean/weathering terms scaled by SMOW ratios.", "Young Section 3.5"),
]


def executable_reactions(
    r5_collision_partner_moles: float | None = None,
    r5_collision_partner_mode: str = "fixed",
    r5_co2_partner_factor: float = 0.0,
    gpp_scale: float = 1.0,
    rp_o2: float | None = None,
    closure_mode: str = "none",
    co2_photo_sink_factor: float = 1.0,
    co2_photo_sink_mode: str = "smow",
    co2_closure_isotope_mode: str = "printed_table3",
    co2_source_override_mol_per_year: float | None = None,
    co2_source_isotope_mode: str = "smow",
    co2_sink_factor: float = 1.0,
    co2_ocean_infusion_factor: float = 1.0,
    r8_biosphere_bookkeeping: bool = False,
    r8_rate_factor: float = 1.0,
    a_mif: float | None = None,
    alpha_respiration_18: float | None = None,
    beta_respiration_17: float | None = None,
    evapotranspiration_alpha_18: float | None = None,
    evapotranspiration_beta_17: float | None = None,
) -> list[Reaction]:
    return (
        stratosphere_photolysis_reactions()
        + reduced_mass_stratosphere_reactions(
            r5_collision_partner_moles=r5_collision_partner_moles,
            r5_collision_partner_mode=r5_collision_partner_mode,
            r5_co2_partner_factor=r5_co2_partner_factor,
            a_mif=a_mif,
        )
        + reservoir_reactions(
            gpp_scale=gpp_scale,
            rp_o2=rp_o2,
            closure_mode=closure_mode,
            co2_photo_sink_factor=co2_photo_sink_factor,
            co2_photo_sink_mode=co2_photo_sink_mode,
            co2_closure_isotope_mode=co2_closure_isotope_mode,
            co2_source_override_mol_per_year=co2_source_override_mol_per_year,
            co2_source_isotope_mode=co2_source_isotope_mode,
            co2_sink_factor=co2_sink_factor,
            co2_ocean_infusion_factor=co2_ocean_infusion_factor,
            r8_biosphere_bookkeeping=r8_biosphere_bookkeeping,
            r8_rate_factor=r8_rate_factor,
            alpha_respiration_18=alpha_respiration_18,
            beta_respiration_17=beta_respiration_17,
            evapotranspiration_alpha_18=evapotranspiration_alpha_18,
            evapotranspiration_beta_17=evapotranspiration_beta_17,
        )
    )


if __name__ == "__main__":
    print(f"executable reactions: {len(executable_reactions())}")
    print(f"reaction records: {len(REACTION_RECORDS)}")
    print(f"todo reaction groups: {len(TODO_REACTIONS)}")
