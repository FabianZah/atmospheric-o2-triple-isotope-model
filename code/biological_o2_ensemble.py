"""Source-backed biological O2 isotope conventions for the updated model.

The module keeps biological-process uncertainty separate from uncertainty in
the numerical value assigned to "100% modern GPP".  Callers supply absolute
GPP in Pg C yr-1.  Terrestrial and marine respiration are split in the same
proportion as gross O2 production, which is an explicit steady-state closure
convention rather than an independently observed global sink partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log1p

import numpy as np

from global_o2_isotope_reservoir import (
    BiologicalO2Compartment,
    IsotopologueTendency,
    PartitionedBiologicalO2Budget,
)
from gpp_normalization import YOUNG_MODERN_GPP_PGC_PER_YEAR
from young_global_o2_budget import (
    BASE_PHOTOSYNTHETIC_O16O16_MOL_PER_YEAR,
    GLOBAL_MAJOR_O2_MOLES_1PAL,
    TROPOSPHERIC_FRACTION,
)


LAND_SOURCE_BETA17 = 0.519
LAND_RESPIRATION_EPSILON18_PERMIL = 18.0
MARINE_RESPIRATION_EPSILON18_PERMIL = 18.9
LAND_SOURCE_DELTA18_CONVENTIONAL_INTERVAL_PERMIL = (4.4, 8.7)
PATHWAY_ALLOCATION_KEYS = ("minimum_land_beta", "proportional", "maximum_land_beta")
CENTRAL_BIOLOGICAL_MEMBER_KEY = (
    "liang_2023__barkan_luz_young__proportional__land_source_midpoint"
)
COMPACT_ENVELOPE_MEMBER_KEYS = (
    "bender_1994__barkan_luz_young__maximum_land_beta__land_source_upper",
    "bender_1994__bender_zero__maximum_land_beta__land_source_upper",
    "young_2014__barkan_luz_young__minimum_land_beta__land_source_lower",
    "young_2014__bender_zero__maximum_land_beta__land_source_upper",
    "young_2014__bender_zero__minimum_land_beta__land_source_lower",
    "young_2014__bender_zero__maximum_land_beta__land_source_lower",
    "young_2014__barkan_luz_young__minimum_land_beta__land_source_midpoint",
)


@dataclass(frozen=True)
class ProductionPartition:
    key: str
    land_fraction: float
    provenance: str


@dataclass(frozen=True)
class MarineSourceConvention:
    key: str
    delta18_prime_permil: float
    beta17: float
    provenance: str


@dataclass(frozen=True)
class RespirationPathway:
    key: str
    global_fraction: float
    beta17: float
    provenance: str


@dataclass(frozen=True)
class BiologicalEnsembleMember:
    """One non-fitted terrestrial-marine biological isotope convention."""

    key: str
    production_partition: ProductionPartition
    marine_source: MarineSourceConvention
    pathway_allocation: str
    land_beta17: float
    marine_beta17: float
    land_source_delta18_prime_permil: float
    provenance: str


PRODUCTION_PARTITIONS = (
    ProductionPartition(
        "bender_1994",
        20.4 / (20.4 + 10.6),
        (
            "Bender et al. (1994), Table 1: 20.4e15 mol yr-1 terrestrial "
            "and 10.6e15 mol yr-1 atmosphere-accessible marine O2 production"
        ),
    ),
    ProductionPartition(
        "young_2014",
        0.60,
        "Young et al. (2014), Section 3.4: about 60% terrestrial GPP",
    ),
    ProductionPartition(
        "liang_2023",
        185.0 / 290.0,
        (
            "Liang et al. (2023): central terrestrial and marine GPP values "
            "185 and 105 Pg C yr-1, normalized to 290 Pg C yr-1 total"
        ),
    ),
)


MARINE_SOURCE_CONVENTIONS = (
    MarineSourceConvention(
        "bender_zero",
        0.0,
        0.528,
        (
            "Bender et al. (1994), Table 1: marine photosynthetic O2 has the "
            "isotopic composition of average surface seawater"
        ),
    ),
    MarineSourceConvention(
        "barkan_luz_young",
        2.883,
        0.525,
        (
            "Young et al. (2014), Section 3.4 interpretation of Barkan and "
            "Luz (2011): marine source delta18-prime=2.883 per mil, beta=0.525"
        ),
    ),
)


RESPIRATION_PATHWAYS = (
    RespirationPathway(
        "COX",
        0.68,
        0.516,
        "Young et al. (2014), Section 2.6; Angert et al. (2003)",
    ),
    RespirationPathway(
        "AOX",
        0.08,
        0.514,
        "Young et al. (2014), Section 2.6; Angert et al. (2003)",
    ),
    RespirationPathway(
        "PR",
        0.24,
        0.512,
        (
            "Young et al. (2014), Section 2.6; Angert et al. (2003), revised "
            "photorespiration value from Helman et al. (2005)"
        ),
    ),
)


YOUNG_PATHWAY_WEIGHTED_BETA17 = float(
    sum(item.global_fraction * item.beta17 for item in RESPIRATION_PATHWAYS)
)


def _extreme_land_allocation(
    land_fraction: float, *, maximize_beta: bool
) -> dict[str, float]:
    remaining = float(land_fraction)
    allocation = {item.key: 0.0 for item in RESPIRATION_PATHWAYS}
    ordered = sorted(
        RESPIRATION_PATHWAYS,
        key=lambda item: item.beta17,
        reverse=maximize_beta,
    )
    for pathway in ordered:
        amount = min(pathway.global_fraction, remaining)
        allocation[pathway.key] = amount
        remaining -= amount
    if abs(remaining) > 1.0e-12:
        raise ValueError("land fraction cannot be allocated to respiration pathways")
    return allocation


def _effective_betas(
    land_fraction: float, land_allocation: dict[str, float]
) -> tuple[float, float]:
    ocean_fraction = 1.0 - land_fraction
    land_beta = sum(
        land_allocation[item.key] * item.beta17 for item in RESPIRATION_PATHWAYS
    ) / land_fraction
    marine_beta = sum(
        (item.global_fraction - land_allocation[item.key]) * item.beta17
        for item in RESPIRATION_PATHWAYS
    ) / ocean_fraction
    return float(land_beta), float(marine_beta)


def pathway_beta_samples(land_fraction: float) -> dict[str, tuple[float, float]]:
    """Return the two feasible endpoints and proportional pathway allocation."""

    minimum = _effective_betas(
        land_fraction,
        _extreme_land_allocation(land_fraction, maximize_beta=False),
    )
    maximum = _effective_betas(
        land_fraction,
        _extreme_land_allocation(land_fraction, maximize_beta=True),
    )
    return {
        "minimum_land_beta": minimum,
        "proportional": (
            YOUNG_PATHWAY_WEIGHTED_BETA17,
            YOUNG_PATHWAY_WEIGHTED_BETA17,
        ),
        "maximum_land_beta": maximum,
    }


def land_source_delta18_prime_samples() -> dict[str, float]:
    """Return lower, midpoint, and upper Bender terrestrial-source values."""

    lower, upper = LAND_SOURCE_DELTA18_CONVENTIONAL_INTERVAL_PERMIL
    conventional = {
        "lower": lower,
        "midpoint": 0.5 * (lower + upper),
        "upper": upper,
    }
    return {
        key: float(1000.0 * log1p(value / 1000.0))
        for key, value in conventional.items()
    }


@lru_cache(maxsize=1)
def biological_ensemble_members() -> tuple[BiologicalEnsembleMember, ...]:
    """Return 54 deterministic literature-corner biological conventions."""

    members: list[BiologicalEnsembleMember] = []
    land_sources = land_source_delta18_prime_samples()
    for partition in PRODUCTION_PARTITIONS:
        beta_samples = pathway_beta_samples(partition.land_fraction)
        for marine_source in MARINE_SOURCE_CONVENTIONS:
            for allocation_key, (land_beta, marine_beta) in beta_samples.items():
                weighted_beta = (
                    partition.land_fraction * land_beta
                    + (1.0 - partition.land_fraction) * marine_beta
                )
                if not np.isclose(
                    weighted_beta,
                    YOUNG_PATHWAY_WEIGHTED_BETA17,
                    atol=1.0e-14,
                    rtol=0.0,
                ):
                    raise FloatingPointError("pathway allocation changed global beta")
                for land_source_key, land_source_delta18 in land_sources.items():
                    key = (
                        f"{partition.key}__{marine_source.key}__{allocation_key}__"
                        f"land_source_{land_source_key}"
                    )
                    members.append(
                        BiologicalEnsembleMember(
                            key=key,
                            production_partition=partition,
                            marine_source=marine_source,
                            pathway_allocation=allocation_key,
                            land_beta17=land_beta,
                            marine_beta17=marine_beta,
                            land_source_delta18_prime_permil=land_source_delta18,
                            provenance=(
                                f"{partition.provenance}; {marine_source.provenance}; "
                                "Young et al. (2014) pathway fractions; Bender et al. "
                                "(1994) terrestrial source-water interval"
                            ),
                        )
                    )
    return tuple(members)


def biological_member(key: str) -> BiologicalEnsembleMember:
    """Return one ensemble member by its stable provenance-bearing key."""

    selected = [item for item in biological_ensemble_members() if item.key == key]
    if len(selected) != 1:
        raise KeyError(key)
    return selected[0]


def central_biological_member() -> BiologicalEnsembleMember:
    """Return the non-fitted central updated biological convention."""

    return biological_member(CENTRAL_BIOLOGICAL_MEMBER_KEY)


def compact_biological_envelope_members() -> tuple[BiologicalEnsembleMember, ...]:
    """Return members validated to bound the complete literature-corner set."""

    return tuple(biological_member(key) for key in COMPACT_ENVELOPE_MEMBER_KEYS)


def fixed_po2_partitioned_biological_budget(
    member: BiologicalEnsembleMember,
    *,
    po2_pal: float,
    gpp_pgC_per_year: float,
    photochemical: IsotopologueTendency,
    marine_accessible_fraction: float = 1.0,
    marine_accessibility_source: str | None = None,
) -> PartitionedBiologicalO2Budget:
    """Build one partitioned budget with exact major-O2 steady-state closure."""

    values = np.asarray(
        (po2_pal, gpp_pgC_per_year, marine_accessible_fraction), dtype=float
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "pO2, absolute GPP, and marine accessibility must be finite and positive"
        )
    if marine_accessible_fraction > 1.0:
        raise ValueError("marine accessible fraction cannot exceed one")
    if marine_accessible_fraction != 1.0 and not marine_accessibility_source:
        raise ValueError("non-unity marine accessibility requires provenance")
    gross_production = (
        BASE_PHOTOSYNTHETIC_O16O16_MOL_PER_YEAR
        * gpp_pgC_per_year
        / YOUNG_MODERN_GPP_PGC_PER_YEAR
    )
    land = member.production_partition.land_fraction
    land_production = land * gross_production
    marine_production = (
        (1.0 - land) * gross_production * marine_accessible_fraction
    )
    production = land_production + marine_production
    target_major = GLOBAL_MAJOR_O2_MOLES_1PAL * po2_pal
    total_respiration_rate = (
        production + photochemical.o16o16
    ) / (TROPOSPHERIC_FRACTION * target_major)
    if total_respiration_rate <= 0.0:
        raise ValueError("photochemical major-isotopologue forcing exceeds GPP")

    land_accessible_fraction = land_production / production
    marine_accessible_partition = marine_production / production
    common = {
        "tropospheric_fraction": TROPOSPHERIC_FRACTION,
    }
    terrestrial = BiologicalO2Compartment(
        key="terrestrial",
        photosynthetic_o16o16_mol_per_year=land_production,
        respiration_rate_per_year=(
            land_accessible_fraction * total_respiration_rate
        ),
        alpha_respiration_18=1.0
        / (1.0 + LAND_RESPIRATION_EPSILON18_PERMIL / 1000.0),
        beta_respiration_17=member.land_beta17,
        source_alpha_18=exp(member.land_source_delta18_prime_permil / 1000.0),
        source_beta_17=LAND_SOURCE_BETA17,
        source=f"{member.provenance}; terrestrial compartment",
        **common,
    )
    marine = BiologicalO2Compartment(
        key="marine",
        photosynthetic_o16o16_mol_per_year=marine_production,
        respiration_rate_per_year=(
            marine_accessible_partition * total_respiration_rate
        ),
        alpha_respiration_18=1.0
        / (1.0 + MARINE_RESPIRATION_EPSILON18_PERMIL / 1000.0),
        beta_respiration_17=member.marine_beta17,
        source_alpha_18=exp(member.marine_source.delta18_prime_permil / 1000.0),
        source_beta_17=member.marine_source.beta17,
        source=f"{member.provenance}; marine compartment",
        **common,
    )
    return PartitionedBiologicalO2Budget(
        compartments=(terrestrial, marine),
        source=(
            f"source-backed partitioned biological O2 budget; member={member.key}; "
            "respiration split equals atmosphere-accessible production split and "
            "total respiration is diagnosed only for prescribed major O2; "
            f"marine_accessible_fraction={marine_accessible_fraction:g}; "
            f"marine_accessibility_source={marine_accessibility_source or 'unity baseline'}"
        ),
    )
