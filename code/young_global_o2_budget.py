"""Young-style biological O2 turnover at a prescribed atmospheric inventory."""

from __future__ import annotations

import numpy as np

from global_o2_isotope_reservoir import IsotopologueTendency, YoungBiologicalO2Budget
from young_model_inventory import PARAMETERS, TABLE3_MOLE_TARGETS


GLOBAL_MAJOR_O2_MOLES_1PAL = (
    TABLE3_MOLE_TARGETS["O2_trop"] + TABLE3_MOLE_TARGETS["O2_strat"]
)
TROPOSPHERIC_FRACTION = (
    TABLE3_MOLE_TARGETS["O2_trop"] / GLOBAL_MAJOR_O2_MOLES_1PAL
)
BASE_PHOTOSYNTHETIC_O16O16_MOL_PER_YEAR = (
    PARAMETERS["k_respiration_per_year"] * TABLE3_MOLE_TARGETS["O2_trop"]
)


def fixed_po2_young_biological_budget(
    *,
    po2_pal: float,
    gpp_percent: float,
    photochemical: IsotopologueTendency,
) -> YoungBiologicalO2Budget:
    """Diagnose respiration to preserve pO2 while retaining Young isotope laws."""

    if not np.isfinite(po2_pal) or po2_pal <= 0.0:
        raise ValueError("pO2 must be finite and positive")
    if not np.isfinite(gpp_percent) or gpp_percent <= 0.0:
        raise ValueError("GPP percentage must be finite and positive")
    production = BASE_PHOTOSYNTHETIC_O16O16_MOL_PER_YEAR * gpp_percent / 100.0
    target_major = GLOBAL_MAJOR_O2_MOLES_1PAL * po2_pal
    respiration_rate = (
        production + photochemical.o16o16
    ) / (TROPOSPHERIC_FRACTION * target_major)
    if respiration_rate <= 0.0:
        raise ValueError("R7 major-isotopologue forcing exceeds gross production")
    return YoungBiologicalO2Budget(
        photosynthetic_o16o16_mol_per_year=production,
        respiration_rate_per_year=respiration_rate,
        tropospheric_fraction=TROPOSPHERIC_FRACTION,
        alpha_respiration_18=PARAMETERS["alpha_respiration_18"],
        beta_respiration_17=0.5149,
        source_alpha_18=PARAMETERS["evapotranspiration_alpha_18"],
        source_beta_17=0.520,
        source=(
            "Young et al. (2014) global biological isotope laws; gross turnover "
            f"={gpp_percent:g}% of the Young scale; respiration rate diagnosed "
            f"to preserve {po2_pal:g} PAL major O2"
        ),
    )
