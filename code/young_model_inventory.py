"""Initial structured inventory for reconstructing Young et al. (2014).

This file is deliberately not a working model yet. It stores only values that
are explicit in the paper. Missing formulas or uncertain implementation details
should remain marked as TODO instead of being guessed.
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


@dataclass(frozen=True)
class Species:
    key: str
    reservoir: str
    label: str


SPECIES = [
    Species("O_strat", "stratosphere", "O"),
    Species("O17_strat", "stratosphere", "17O"),
    Species("O18_strat", "stratosphere", "18O"),
    Species("O1D_strat", "stratosphere", "O(1D)"),
    Species("O17_1D_strat", "stratosphere", "17O(1D)"),
    Species("O18_1D_strat", "stratosphere", "18O(1D)"),
    Species("O2_strat", "stratosphere", "O2"),
    Species("O17O_strat", "stratosphere", "O17O"),
    Species("O18O_strat", "stratosphere", "O18O"),
    Species("CO2_strat", "stratosphere", "CO2"),
    Species("CO17O_strat", "stratosphere", "CO17O"),
    Species("CO18O_strat", "stratosphere", "CO18O"),
    Species("O3_strat", "stratosphere", "O3"),
    Species("OO17O_strat", "stratosphere", "OO17O"),
    Species("OO18O_strat", "stratosphere", "OO18O"),
    Species("O2_trop", "troposphere", "O2"),
    Species("O18O_trop", "troposphere", "O18O"),
    Species("O17O_trop", "troposphere", "O17O"),
    Species("CO2_trop", "troposphere", "CO2"),
    Species("CO18O_trop", "troposphere", "CO18O"),
    Species("CO17O_trop", "troposphere", "CO17O"),
    Species("O_bio", "biosphere_hydrosphere", "[O]"),
    Species("O18_bio", "biosphere_hydrosphere", "[18O]"),
    Species("O17_bio", "biosphere_hydrosphere", "[17O]"),
    Species("O_geo", "geosphere", "[O]"),
    Species("O18_geo", "geosphere", "[18O]"),
    Species("O17_geo", "geosphere", "[17O]"),
]


# Explicit constants from Young et al. Table 2 and surrounding text.
PARAMETERS = {
    "temperature_stratosphere_K": 220.0,
    "number_density_25km_cm3": 8.3e17,
    "stratosphere_volume_cm3": 2.8e25,
    "avogadro_over_strat_volume_cm_minus3": 0.0215,
    "moles_stratosphere": 1.8e19,
    "moles_troposphere": 1.8e20,
    "k_ST_per_year": 1.0,
    "k_TS_per_year": 0.1,
    "J_R1a_per_s": 1.109e-12,
    "k_R2a_base_cm3_s": 1.26e-33 * 8.3e17,
    "J_R3a_per_s": 2.96e-4,
    "J_R3f_per_s": 5.01e-4,
    "k_R4a_cm3_s": 6.86e-15,
    "k_R4f_cm3_s": 1.20e-10,
    "k_R5a_cm3_s": 1.4e-10,
    "k_R6_cm3_s": 2.0e-16,
    "k_R7a_cm3_s": 4.46e-11,
    "k_R8b_per_year": 1.0,
    "alpha_CO2_H2O_18": 1.041,
    "beta_CO2_H2O_17": 0.528,
    "k_respiration_per_year": 0.0008,
    "alpha_respiration_18": 1.0 / 1.0182,
    "beta_respiration_17": 0.5149,
    "evapotranspiration_alpha_18": 1.00525,
    "evapotranspiration_beta_17": 0.520,
    "k_O2_weathering_per_year": 6.0e-7,
    "k_organic_burial_per_year": 5.0e-5,
    "k_CO2_weathering_per_year": 0.000208,
    "f_volcanic_CO2_mol_per_year": 4.0e12,
    "f_ocean_CO2_mol_per_year": 5.88e15,
    "k_ocean_CO2_infusion_per_year": 0.1172,
    "a_MIF": 1.065,
}


TABLE3_ISOTOPE_TARGETS = {
    "d18_O_permil": 51.869,
    "d18_O1D_permil": 71.107,
    "d18_O2_strat_permil": 23.212,
    "d18_CO2_strat_permil": 40.345,
    "d18_O3_permil": 76.268,
    "d18_O2_trop_permil": 23.212,
    "d18_CO2_trop_permil": 40.688,
    "d17_O_permil": 26.887,
    "d17_O1D_permil": 64.599,
    "d17_O2_strat_permil": 11.886,
    "d17_CO2_strat_permil": 22.915,
    "d17_O3_permil": 69.766,
    "d17_O2_trop_permil": 11.887,
    "d17_CO2_trop_permil": 21.601,
    "D17_O_permil": -0.500,
    "D17_O1D_permil": 27.054,
    "D17_O2_strat_permil": -0.370,
    "D17_CO2_strat_permil": 1.613,
    "D17_O3_permil": 29.497,
    "D17_O2_trop_permil": -0.410,
    "D17_CO2_trop_permil": 0.118,
}


TABLE3_MOLE_TARGETS = {
    "O_strat": 1.23e9,
    "O17_strat": 4.68e5,
    "O18_strat": 2.64e6,
    "O1D_strat": 3.83e3,
    "O17_1D_strat": 1.51,
    "O18_1D_strat": 8.40,
    "O2_strat": 3.80e18,
    "O17O_strat": 2.85e15,
    "O18O_strat": 1.59e16,
    "CO2_strat": 5.29e15,
    "CO17O_strat": 2.01e12,
    "CO18O_strat": 1.13e13,
    "O3_strat": 1.28e14,
    "OO17O_strat": 1.53e11,
    "OO18O_strat": 8.50e11,
    "O2_trop": 3.80e19,
    "O18O_trop": 1.59e17,
    "O17O_trop": 2.85e16,
    "CO2_trop": 5.29e16,
    "CO18O_trop": 1.13e14,
    "CO17O_trop": 2.00e13,
}


TABLE3_MOLE_FRACTION_TARGETS = {
    "X_O2_trop": 0.212,
    "X_CO2_trop": 2.944e-4,
    "X_CO2_strat": 2.944e-4,
    "X_O3_strat": 7.184e-6,
}


TABLE3_FLUX_TARGETS = {
    "D17_CO2_flux_permil_mol_per_year": 8.55e15,
}


TABLE3_TARGETS = {
    **TABLE3_ISOTOPE_TARGETS,
    **TABLE3_MOLE_FRACTION_TARGETS,
    **TABLE3_FLUX_TARGETS,
}


OPEN_QUESTIONS = [
    "Complete expanded ODEs for all 27 species are not printed in the paper.",
    "Initial conditions are described qualitatively, not tabulated as a full vector.",
    "Reduced-mass implementation needs exact molecular/isotope masses and reactant-pair conventions.",
    "All stoichiometric products for Table 2 must be transcribed and checked reaction by reaction.",
    "Original DLSODE tolerances are stated, but solver-specific behavior may differ in Python.",
]


if __name__ == "__main__":
    print(f"{len(SPECIES)} species")
    print(f"{len(PARAMETERS)} explicit parameters")
    print(f"{len(TABLE3_TARGETS)} validation targets")
