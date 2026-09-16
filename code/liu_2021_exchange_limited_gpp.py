"""Liu et al. (2021) marine-GPP to atmospheric-O2 exchange boundary.

Liu et al. distinguish total marine gross primary production from the O2
that is actually reprocessed across the air-sea interface.  Their archived
photochemical grid stores the latter as ``GPPOXY``.  This module changes
total production without silently relabelling ``GPPOXY`` as total GPP.

The competition law is the standard two-first-order-process result.  Its
single dimensionless competition ratio is diagnosed independently at each
pO2 from Liu's published reference exchange flux, so no response magnitude
is fitted to this project's model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


LIU_MODERN_MARINE_GPP_TMOL_O2_PER_YEAR = 12000.0
LIU_MODERN_ACCESSIBLE_O2_TMOL_PER_YEAR = 10600.0
LIU_MODERN_INTERNAL_RECYCLING_FRACTION = 0.12
LIU_ARCHIVE_MODERN_TOTAL_GPP_FLUX_MOLECULES_CM2_S = 4.50e13
LIU_ARCHIVE_MODERN_ACCESSIBLE_FLUX_MOLECULES_CM2_S = 3.98e13
LIU_REFERENCE_ACCESSIBLE_FLUX_BY_PO2_PAL = {
    0.1: 2.70e13,
    0.3: 3.60e13,
    1.0: LIU_ARCHIVE_MODERN_ACCESSIBLE_FLUX_MOLECULES_CM2_S,
}


@dataclass(frozen=True)
class LiuExchangeLimitedGPP:
    """One total-GPP state mapped onto Liu's atmospheric O2 boundary."""

    gpp_percent_of_liu_modern: float
    total_marine_gpp_tmol_o2_per_year: float
    unconstrained_total_flux_molecules_cm2_s: float
    accessible_o2_flux_molecules_cm2_s: float
    accessible_o2_flux_tmol_per_year: float
    accessible_fraction: float
    internal_recycling_fraction: float
    reference_accessible_flux_molecules_cm2_s: float
    competition_ratio_at_reference_gpp: float


def exchange_limited_gpp(
    *,
    gpp_percent_of_liu_modern: float,
    reference_accessible_flux_molecules_cm2_s: float,
) -> LiuExchangeLimitedGPP:
    """Map total marine GPP to the atmosphere-accessible O2 exchange flux.

    For two competing first-order pathways, air-sea export and internal
    respiratory recycling, the exported fraction is ``1 / (1 + q*s)``.
    Here ``s`` is total GPP relative to Liu's modern marine value and ``q``
    is diagnosed so that ``s=1`` exactly reproduces the archived GPPOXY at
    the selected pO2.  The expression therefore preserves Liu's reference
    grid while exposing the previously unsampled productivity dimension.
    """

    gpp_percent = float(gpp_percent_of_liu_modern)
    reference_accessible = float(reference_accessible_flux_molecules_cm2_s)
    if not math.isfinite(gpp_percent) or gpp_percent <= 0.0:
        raise ValueError("Liu marine GPP percent must be finite and positive")
    if (
        not math.isfinite(reference_accessible)
        or reference_accessible <= 0.0
        or reference_accessible > LIU_ARCHIVE_MODERN_TOTAL_GPP_FLUX_MOLECULES_CM2_S
    ):
        raise ValueError(
            "reference accessible O2 flux must be positive and no greater "
            "than Liu's modern total marine-GPP flux"
        )

    scale = gpp_percent / 100.0
    total_flux = LIU_ARCHIVE_MODERN_TOTAL_GPP_FLUX_MOLECULES_CM2_S * scale
    competition_ratio = (
        LIU_ARCHIVE_MODERN_TOTAL_GPP_FLUX_MOLECULES_CM2_S / reference_accessible - 1.0
    )
    accessible_fraction = 1.0 / (1.0 + competition_ratio * scale)
    accessible_flux = total_flux * accessible_fraction
    # GPPOXY is rounded in the archived text.  Anchor its area/unit conversion
    # to Liu's independently printed 10,600 Tmol yr-1 accessible flux rather
    # than forcing the rounded 3.98/4.50 ratio to equal exactly 0.88.
    conversion = (
        LIU_MODERN_ACCESSIBLE_O2_TMOL_PER_YEAR
        / LIU_ARCHIVE_MODERN_ACCESSIBLE_FLUX_MOLECULES_CM2_S
    )
    return LiuExchangeLimitedGPP(
        gpp_percent_of_liu_modern=gpp_percent,
        total_marine_gpp_tmol_o2_per_year=(
            LIU_MODERN_MARINE_GPP_TMOL_O2_PER_YEAR * scale
        ),
        unconstrained_total_flux_molecules_cm2_s=total_flux,
        accessible_o2_flux_molecules_cm2_s=accessible_flux,
        accessible_o2_flux_tmol_per_year=accessible_flux * conversion,
        accessible_fraction=accessible_fraction,
        internal_recycling_fraction=1.0 - accessible_fraction,
        reference_accessible_flux_molecules_cm2_s=reference_accessible,
        competition_ratio_at_reference_gpp=competition_ratio,
    )


def exchange_limited_gpp_at_supported_po2(
    *,
    po2_pal: float,
    gpp_percent_of_liu_modern: float,
) -> LiuExchangeLimitedGPP:
    """Evaluate Liu's exchange law at an explicitly archived pO2 node.

    No interpolation or extrapolation in pO2 is applied.  The three reference
    fluxes are the GPPOXY values in the archived Liu grid used by this project.
    """

    po2 = float(po2_pal)
    matches = [
        value
        for node, value in LIU_REFERENCE_ACCESSIBLE_FLUX_BY_PO2_PAL.items()
        if math.isclose(po2, node, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        supported = ", ".join(
            f"{value:g}" for value in LIU_REFERENCE_ACCESSIBLE_FLUX_BY_PO2_PAL
        )
        raise ValueError(
            f"Liu exchange mapping is supported only at pO2={supported} PAL"
        )
    return exchange_limited_gpp(
        gpp_percent_of_liu_modern=gpp_percent_of_liu_modern,
        reference_accessible_flux_molecules_cm2_s=matches[0],
    )
