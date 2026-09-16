"""GPP normalization helpers for scenario/UI-facing inputs.

The reduced Young-model machinery uses Young's Table-3 photosynthetic O2 flux
as its internal "100% modern" scale. User-facing scenarios can instead express
GPP as percent of a selected modern PgC/yr reference, then convert back to the
internal Young scale before running the isotope model.
"""

from __future__ import annotations

from dataclasses import dataclass

from young_model_inventory import PARAMETERS


CARBON_MOLAR_MASS_G_PER_MOL = 12.0107
DEFAULT_GPP_NORMALIZATION = "young_2014"
PUBLIC_GPP_POLICY = (
    "Use the Young et al. (2014) internal photosynthetic O2-flux scale as the "
    "default 100% modern GPP convention. Beerling (1999) gives an effectively "
    "equivalent gross O2-production total. Liang et al. (2023) and custom "
    "global values are exposed as explicit sensitivity/reference normalizations "
    "and must be recorded in exports. Adnew et al. (2025) constrain terrestrial "
    "GPP and are not a global normalization."
)
YOUNG_MODERN_GPP_PGC_PER_YEAR = (
    PARAMETERS["k_respiration_per_year"] * 3.80e19 * CARBON_MOLAR_MASS_G_PER_MOL / 1.0e15
)
BEERLING_1999_MODERN_GPP_PMOL_O2_PER_YEAR = 30.6
BEERLING_1999_MODERN_GPP_PGC_PER_YEAR = BEERLING_1999_MODERN_GPP_PMOL_O2_PER_YEAR * CARBON_MOLAR_MASS_G_PER_MOL


@dataclass(frozen=True)
class GppNormalization:
    key: str
    label: str
    modern_pgC_per_year: float
    note: str
    uncertainty_pgC_per_year: float | None = None


GPP_NORMALIZATIONS: dict[str, GppNormalization] = {
    "young_2014": GppNormalization(
        key="young_2014",
        label="Young et al., 2014 gross O2-production scale",
        modern_pgC_per_year=YOUNG_MODERN_GPP_PGC_PER_YEAR,
        note=(
            "Default model convention. This is the internal Young et al., 2014 photosynthetic O2-flux "
            "scale and is effectively equivalent to Beerling, 1999 Table 2 total gross O2 production."
        ),
    ),
    "beerling_1999": GppNormalization(
        key="beerling_1999",
        label="Beerling, 1999 Table 2 total",
        modern_pgC_per_year=BEERLING_1999_MODERN_GPP_PGC_PER_YEAR,
        note=(
            "Terrestrial plus marine gross O2 production from Beerling, 1999 Table 2 "
            "(19.6 + 11.0 = 30.6 Pmol O2/yr), converted stoichiometrically to PgC/yr."
        ),
    ),
    "liang_2023": GppNormalization(
        key="liang_2023",
        label="Liang et al., 2023",
        modern_pgC_per_year=290.0,
        note="Modern terrestrial-plus-marine GPP reference used as a literature update candidate.",
        uncertainty_pgC_per_year=30.0,
    ),
    "custom": GppNormalization(
        key="custom",
        label="Custom",
        modern_pgC_per_year=YOUNG_MODERN_GPP_PGC_PER_YEAR,
        note="User-specified modern GPP normalization.",
    ),
}


def normalization_key(key: str | None) -> str:
    return DEFAULT_GPP_NORMALIZATION if key is None else key


def normalization_role(key: str | None) -> str:
    resolved = normalization_key(key)
    if resolved == DEFAULT_GPP_NORMALIZATION:
        return "default_gross_o2_production_scale"
    if resolved == "beerling_1999":
        return "near_equivalent_gross_o2_production_crosscheck"
    if resolved in {"liang_2023", "custom"}:
        return "explicit_modern_reference_sensitivity"
    return "unknown"


def modern_gpp_pgC_per_year(key: str | None = None, custom_pgC_per_year: float | None = None) -> float:
    resolved = normalization_key(key)
    if resolved == "custom":
        if custom_pgC_per_year is None:
            raise ValueError("custom GPP normalization requires custom_pgC_per_year")
        if custom_pgC_per_year <= 0.0:
            raise ValueError("custom GPP normalization must be positive")
        return float(custom_pgC_per_year)
    if resolved not in GPP_NORMALIZATIONS:
        raise ValueError(f"unknown GPP normalization {resolved!r}; choices: {', '.join(GPP_NORMALIZATIONS)}")
    return GPP_NORMALIZATIONS[resolved].modern_pgC_per_year


def modern_gpp_uncertainty_pgC_per_year(
    key: str | None = None,
    custom_uncertainty_pgC_per_year: float | None = None,
) -> float | None:
    """Return the stated uncertainty of a selected modern GPP reference."""

    resolved = normalization_key(key)
    if resolved == "custom":
        if custom_uncertainty_pgC_per_year is None:
            return None
        if custom_uncertainty_pgC_per_year < 0.0:
            raise ValueError("custom GPP normalization uncertainty must be non-negative")
        return float(custom_uncertainty_pgC_per_year)
    if resolved not in GPP_NORMALIZATIONS:
        raise ValueError(
            f"unknown GPP normalization {resolved!r}; choices: "
            f"{', '.join(GPP_NORMALIZATIONS)}"
        )
    return GPP_NORMALIZATIONS[resolved].uncertainty_pgC_per_year


def requested_gpp_interval_pgC_per_year(
    user_gpp_scale: float,
    key: str | None = None,
    custom_pgC_per_year: float | None = None,
    custom_uncertainty_pgC_per_year: float | None = None,
) -> tuple[float, float]:
    """Convert relative-modern GPP to its absolute literature interval."""

    if user_gpp_scale <= 0.0:
        raise ValueError("relative-modern GPP scale must be positive")
    central = modern_gpp_pgC_per_year(key, custom_pgC_per_year)
    uncertainty = modern_gpp_uncertainty_pgC_per_year(
        key, custom_uncertainty_pgC_per_year
    )
    if uncertainty is None:
        value = float(user_gpp_scale * central)
        return value, value
    lower = user_gpp_scale * (central - uncertainty)
    upper = user_gpp_scale * (central + uncertainty)
    if lower <= 0.0:
        raise ValueError("GPP reference interval must remain positive")
    return float(lower), float(upper)


def internal_young_gpp_scale(
    user_gpp_scale: float,
    key: str | None = None,
    custom_pgC_per_year: float | None = None,
) -> float:
    """Convert a user-facing percent-modern scale to Young's internal scale."""

    return user_gpp_scale * modern_gpp_pgC_per_year(key, custom_pgC_per_year) / YOUNG_MODERN_GPP_PGC_PER_YEAR


def requested_gpp_pgC_per_year(
    user_gpp_scale: float,
    key: str | None = None,
    custom_pgC_per_year: float | None = None,
) -> float:
    return user_gpp_scale * modern_gpp_pgC_per_year(key, custom_pgC_per_year)
