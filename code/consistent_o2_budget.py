"""Helpers for pO2/GPP/NPP-consistent O2 major-budget experiments.

Young's fixed-pO2 figure calculations vary GPP at fixed NPP/O2 by scaling
photosynthesis and respiration together. If we change pO2 explicitly, keeping
the modern photosynthetic O2 flux is internally inconsistent. These helpers
compute a target photosynthetic flux for a chosen pO2 and respiration/GPP scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from young_model_inventory import PARAMETERS
from young_reactions import TABLE3_O2_TROP, lasaga_ohmoto_burial_relative


@dataclass(frozen=True)
class O2BudgetTerms:
    p_o2_pal: float
    gpp_scale: float
    closure_mode: str
    respiration_o2: float
    weathering_o2: float
    burial_o2: float
    required_rp_o2: float
    net_primary_o2: float
    rp_relative_to_modern: float
    npp_relative_to_modern_gpp: float


def burial_relative_to_modern(p_o2_pal: float, closure_mode: str) -> float:
    if closure_mode in ("organic_burial", "modern", "finite_geosphere_burial", "modern_finite_geosphere"):
        return 1.0 / p_o2_pal
    if closure_mode in ("lasaga_burial", "modern_lasaga_burial"):
        return lasaga_ohmoto_burial_relative(p_o2_pal)
    if closure_mode == "none":
        return 0.0
    raise ValueError(f"unsupported closure mode for O2 budget helper: {closure_mode}")


def consistent_rp_o2(
    p_o2_pal: float,
    gpp_scale: float = 1.0,
    closure_mode: str = "organic_burial",
) -> O2BudgetTerms:
    """Return the photosynthetic O2 flux required for target O2 major balance.

    The balance is for the tropospheric 16O2 major reservoir under the same
    simplified assumptions used by the current Young-style closure:

        rp + burial = respiration + weathering

    Transport cancels when stratospheric/tropospheric O2 scale together, as in
    `scaled_table3_state`. Fast stratospheric photochemistry is not included in
    this low-order major-budget helper.
    """

    o2_target = TABLE3_O2_TROP * p_o2_pal
    modern_weathering = PARAMETERS["k_O2_weathering_per_year"] * TABLE3_O2_TROP
    respiration = PARAMETERS["k_respiration_per_year"] * gpp_scale * o2_target
    weathering = PARAMETERS["k_O2_weathering_per_year"] * o2_target
    burial = modern_weathering * burial_relative_to_modern(p_o2_pal, closure_mode)
    required_rp = respiration + weathering - burial
    modern_rp = PARAMETERS["k_respiration_per_year"] * TABLE3_O2_TROP
    net_primary = required_rp - respiration
    return O2BudgetTerms(
        p_o2_pal=p_o2_pal,
        gpp_scale=gpp_scale,
        closure_mode=closure_mode,
        respiration_o2=respiration,
        weathering_o2=weathering,
        burial_o2=burial,
        required_rp_o2=required_rp,
        net_primary_o2=net_primary,
        rp_relative_to_modern=required_rp / modern_rp,
        npp_relative_to_modern_gpp=net_primary / modern_rp,
    )
