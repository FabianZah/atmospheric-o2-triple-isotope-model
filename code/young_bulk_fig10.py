"""Source-derived reduced Young et al. (2014) Fig. 10 response.

This module is intentionally specific to the Young-reproduction branch. It
uses the printed homogeneous stratospheric R7 ledger and global biological
turnover; the updated model retains its vertically resolved R7 column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from global_o2_isotope_reservoir import (
    GlobalO2Reservoir,
    IsotopologueTendency,
    OxygenAtomFlux,
    YoungBiologicalO2Budget,
    frozen_photochemical_steady_state,
    isotopologue_tendency_from_atom_flux,
    sum_tendencies,
)
from isotopes import R17_VSMOW, R18_VSMOW
from reactions import derivative
from young_model_inventory import (
    PARAMETERS,
    SPECIES,
    TABLE3_ISOTOPE_TARGETS,
    TABLE3_MOLE_TARGETS,
)
from young_reactions import reduced_mass_stratosphere_reactions


YOUNG_FIG10_INITIAL_PCO2_PPM = 294.0
YOUNG_FIG10_FINAL_PCO2_PPM = 400.0


@dataclass(frozen=True)
class YoungBulkFig10Trajectory:
    time_years: np.ndarray
    states: tuple[GlobalO2Reservoir, ...]
    initial_equilibrium: GlobalO2Reservoir
    final_equilibrium: GlobalO2Reservoir
    r7_294: IsotopologueTendency
    r7_400: IsotopologueTendency

    @property
    def cap_delta17_shift_permil(self) -> np.ndarray:
        initial = self.initial_equilibrium.cap_delta17_prime_permil
        return np.asarray(
            [state.cap_delta17_prime_permil - initial for state in self.states]
        )

    @property
    def delta18_shift_permil(self) -> np.ndarray:
        initial = self.initial_equilibrium.delta18_prime_permil
        return np.asarray(
            [state.delta18_prime_permil - initial for state in self.states]
        )


def young_table3_r7_state():
    species = tuple(item.key for item in SPECIES)
    index = {name: position for position, name in enumerate(species)}
    state = np.ones(len(species), dtype=float)
    for name, value in TABLE3_MOLE_TARGETS.items():
        if name in index:
            state[index[name]] = value
    # Printed rare inventories are rounded too coarsely for the isotope ledger.
    # Reconstruct the four internal ratios from the printed logarithmic values.
    state[index["O17_1D_strat"]] = (
        state[index["O1D_strat"]]
        * R17_VSMOW
        * np.exp(TABLE3_ISOTOPE_TARGETS["d17_O1D_permil"] / 1000.0)
    )
    state[index["O18_1D_strat"]] = (
        state[index["O1D_strat"]]
        * R18_VSMOW
        * np.exp(TABLE3_ISOTOPE_TARGETS["d18_O1D_permil"] / 1000.0)
    )
    state[index["CO17O_strat"]] = (
        state[index["CO2_strat"]]
        * R17_VSMOW
        * np.exp(TABLE3_ISOTOPE_TARGETS["d17_CO2_strat_permil"] / 1000.0)
    )
    state[index["CO18O_strat"]] = (
        state[index["CO2_strat"]]
        * R18_VSMOW
        * np.exp(TABLE3_ISOTOPE_TARGETS["d18_CO2_strat_permil"] / 1000.0)
    )
    reactions = tuple(
        reaction
        for reaction in reduced_mass_stratosphere_reactions()
        if reaction.key.startswith("R7")
    )
    return species, index, state, reactions


def young_table3_r7_rates() -> dict[str, float]:
    _species, index, state, reactions = young_table3_r7_state()
    return {reaction.key: reaction.rate(state, index) for reaction in reactions}


def young_table3_r7_tendencies() -> tuple[
    IsotopologueTendency, IsotopologueTendency
]:
    species, index, state, reactions = young_table3_r7_state()
    tendency = derivative(state, list(reactions), list(species))
    external_atom_flux = OxygenAtomFlux(
        o16=float(tendency[index["O_strat"]] + tendency[index["O1D_strat"]]),
        o17=float(
            tendency[index["O17_strat"]] + tendency[index["O17_1D_strat"]]
        ),
        o18=float(
            tendency[index["O18_strat"]] + tendency[index["O18_1D_strat"]]
        ),
        source="Young Table 3 R7 external-oxygen tendency",
    )
    carbon_atom_flux = OxygenAtomFlux(
        *map(float, -external_atom_flux.values),
        source="Young Table 3 R7 carbon-oxygen tendency",
    )
    return (
        isotopologue_tendency_from_atom_flux(external_atom_flux),
        isotopologue_tendency_from_atom_flux(carbon_atom_flux),
    )


def young_printed_biological_budget() -> YoungBiologicalO2Budget:
    return YoungBiologicalO2Budget(
        photosynthetic_o16o16_mol_per_year=(
            PARAMETERS["k_respiration_per_year"] * TABLE3_MOLE_TARGETS["O2_trop"]
        ),
        respiration_rate_per_year=PARAMETERS["k_respiration_per_year"],
        tropospheric_fraction=(
            TABLE3_MOLE_TARGETS["O2_trop"]
            / (TABLE3_MOLE_TARGETS["O2_trop"] + TABLE3_MOLE_TARGETS["O2_strat"])
        ),
        alpha_respiration_18=PARAMETERS["alpha_respiration_18"],
        beta_respiration_17=0.5149,
        source_alpha_18=PARAMETERS["evapotranspiration_alpha_18"],
        source_beta_17=0.520,
        source="Young et al. (2014) printed global biological isotope budget",
    )


def integrate_young_bulk_fig10(
    time_years: np.ndarray,
    *,
    pco2_initial_ppm: float = YOUNG_FIG10_INITIAL_PCO2_PPM,
    pco2_final_ppm: float = YOUNG_FIG10_FINAL_PCO2_PPM,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
) -> YoungBulkFig10Trajectory:
    times = np.asarray(time_years, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(~np.isfinite(times)):
        raise ValueError("Fig. 10 times must be a finite one-dimensional array")
    if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Fig. 10 times must start at zero and increase strictly")
    if pco2_initial_ppm <= 0.0 or pco2_final_ppm <= 0.0:
        raise ValueError("Fig. 10 pCO2 values must be positive")
    if not np.isclose(pco2_initial_ppm, YOUNG_FIG10_INITIAL_PCO2_PPM):
        raise ValueError(
            "the printed Table 3 bulk R7 ledger is anchored specifically at 294 ppm"
        )

    biology = young_printed_biological_budget()
    r7_294, _carbon = young_table3_r7_tendencies()
    initial = frozen_photochemical_steady_state(
        biology, r7_294, source="reduced Young bulk R7/biology equilibrium"
    )
    ratio = pco2_final_ppm / pco2_initial_ppm
    r7_final = IsotopologueTendency(
        *map(float, ratio * r7_294.values),
        source=(
            "Young Table 3 R7 tendency scaled by homogeneous-box CO2 mass action "
            f"({pco2_final_ppm:g}/{pco2_initial_ppm:g})"
        ),
    )
    final = frozen_photochemical_steady_state(
        biology, r7_final, source="reduced Young bulk final equilibrium"
    )
    inventory_scale = initial.isotopologue_moles

    def rhs(_time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * inventory_scale),
            source="Young bulk Fig. 10 integration state",
        )
        return sum_tendencies(
            (biology.tendency(reservoir), r7_final),
            source="printed Young biology plus mass-action bulk R7",
        ).values / inventory_scale

    solved = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.ones(3),
        t_eval=times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=20.0,
    )
    if not solved.success or solved.y.shape[1] != times.size:
        raise RuntimeError(solved.message)
    states = tuple(
        GlobalO2Reservoir(
            *map(float, inventory), source=f"Young bulk Fig. 10 at {time:g} yr"
        )
        for time, inventory in zip(
            times, solved.y.T * inventory_scale, strict=True
        )
    )
    return YoungBulkFig10Trajectory(
        time_years=times.copy(),
        states=states,
        initial_equilibrium=initial,
        final_equilibrium=final,
        r7_294=r7_294,
        r7_400=r7_final,
    )
