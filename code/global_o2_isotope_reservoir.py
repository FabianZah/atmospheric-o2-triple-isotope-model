"""Conservative bridge from resolved R7 chemistry to global atmospheric O2.

The fast atmospheric column transfers individual oxygen atoms between O/O(1D)
and CO2. Atmospheric O2 is instead represented here by one globally mixed,
slow reservoir. The bridge converts isotope-atom fluxes into the three
Young-compatible O2 isotopologue inventories without assigning a local O2
anomaly or a fitted exchange factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, log
from typing import Iterable

import numpy as np

from isotopes import R17_VSMOW, R18_VSMOW


OXYGEN_ISOTOPES = (16, 17, 18)


@dataclass(frozen=True)
class OxygenAtomFlux:
    """Flux of 16O, 17O, and 18O atoms in mol atoms yr-1."""

    o16: float
    o17: float
    o18: float
    source: str

    def __post_init__(self) -> None:
        values = np.asarray((self.o16, self.o17, self.o18), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("oxygen-atom fluxes must be finite")
        if not self.source:
            raise ValueError("oxygen-atom flux requires provenance")

    @property
    def values(self) -> np.ndarray:
        return np.asarray((self.o16, self.o17, self.o18), dtype=float)

    @property
    def total(self) -> float:
        return float(self.o16 + self.o17 + self.o18)


@dataclass(frozen=True)
class R7GlobalTransfer:
    """Equal-and-opposite R7 atom fluxes for carbon oxygen and global O2."""

    carbon_oxygen: OxygenAtomFlux
    global_o2: OxygenAtomFlux
    closure_residual: OxygenAtomFlux
    maximum_relative_closure_residual: float


@dataclass(frozen=True)
class GlobalO2Reservoir:
    """Globally mixed O2 in the singly substituted Young convention.

    ``o16o16`` is the major O2 isotopologue. ``o16o17`` and ``o16o18`` are
    singly substituted trace isotopologues. Ratios are therefore evaluated as
    rare isotopologue / (2 * major isotopologue), matching the existing Young
    reconstruction and Table 3 diagnostics.
    """

    o16o16: float
    o16o17: float
    o16o18: float
    source: str

    def __post_init__(self) -> None:
        values = self.isotopologue_moles
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("global O2 isotopologue inventories must be positive")
        if not self.source:
            raise ValueError("global O2 reservoir requires provenance")

    @classmethod
    def from_prime_composition(
        cls,
        *,
        major_o2_moles: float,
        delta18_prime_permil: float,
        cap_delta17_prime_permil: float,
        source: str,
    ) -> "GlobalO2Reservoir":
        if not np.isfinite(major_o2_moles) or major_o2_moles <= 0.0:
            raise ValueError("major O2 inventory must be finite and positive")
        delta17_prime = cap_delta17_prime_permil + 0.528 * delta18_prime_permil
        ratio17 = R17_VSMOW * exp(delta17_prime / 1000.0)
        ratio18 = R18_VSMOW * exp(delta18_prime_permil / 1000.0)
        return cls(
            o16o16=float(major_o2_moles),
            o16o17=float(2.0 * major_o2_moles * ratio17),
            o16o18=float(2.0 * major_o2_moles * ratio18),
            source=source,
        )

    @property
    def isotopologue_moles(self) -> np.ndarray:
        return np.asarray((self.o16o16, self.o16o17, self.o16o18), dtype=float)

    @property
    def atom_moles(self) -> np.ndarray:
        return np.asarray(
            (
                2.0 * self.o16o16 + self.o16o17 + self.o16o18,
                self.o16o17,
                self.o16o18,
            ),
            dtype=float,
        )

    @property
    def ratio17(self) -> float:
        return float(self.o16o17 / (2.0 * self.o16o16))

    @property
    def ratio18(self) -> float:
        return float(self.o16o18 / (2.0 * self.o16o16))

    @property
    def delta17_prime_permil(self) -> float:
        return float(1000.0 * log(self.ratio17 / R17_VSMOW))

    @property
    def delta18_prime_permil(self) -> float:
        return float(1000.0 * log(self.ratio18 / R18_VSMOW))

    @property
    def cap_delta17_prime_permil(self) -> float:
        return float(
            self.delta17_prime_permil - 0.528 * self.delta18_prime_permil
        )


@dataclass(frozen=True)
class IsotopologueTendency:
    """O16O16, O16O17, and O16O18 tendency in mol molecules yr-1."""

    o16o16: float
    o16o17: float
    o16o18: float
    source: str

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.values)):
            raise ValueError("O2 isotopologue tendencies must be finite")
        if not self.source:
            raise ValueError("O2 isotopologue tendency requires provenance")

    @property
    def values(self) -> np.ndarray:
        return np.asarray((self.o16o16, self.o16o17, self.o16o18), dtype=float)

    @property
    def atom_values(self) -> np.ndarray:
        return np.asarray(
            (
                2.0 * self.o16o16 + self.o16o17 + self.o16o18,
                self.o16o17,
                self.o16o18,
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class AffinePhotochemicalO2Feedback:
    """Resolved photochemical tendency affine in global O2 isotope ratios."""

    reference_ratio17: float
    reference_ratio18: float
    reference_tendency: IsotopologueTendency
    ratio_jacobian_mol_per_year: np.ndarray
    source: str

    def __post_init__(self) -> None:
        references = np.asarray(
            (self.reference_ratio17, self.reference_ratio18), dtype=float
        )
        jacobian = np.asarray(self.ratio_jacobian_mol_per_year, dtype=float)
        if np.any(~np.isfinite(references)) or np.any(references <= 0.0):
            raise ValueError("feedback reference isotope ratios must be positive")
        if jacobian.shape != (3, 2) or not np.all(np.isfinite(jacobian)):
            raise ValueError("feedback ratio Jacobian must be finite with shape (3, 2)")
        if not self.source:
            raise ValueError("photochemical feedback requires provenance")
        object.__setattr__(self, "ratio_jacobian_mol_per_year", jacobian.copy())

    def tendency(self, reservoir: GlobalO2Reservoir) -> IsotopologueTendency:
        ratio_offset = np.asarray(
            (
                reservoir.ratio17 - self.reference_ratio17,
                reservoir.ratio18 - self.reference_ratio18,
            ),
            dtype=float,
        )
        values = (
            self.reference_tendency.values
            + self.ratio_jacobian_mol_per_year @ ratio_offset
        )
        return IsotopologueTendency(
            *map(float, values),
            source=f"{self.source}; evaluated at global O2 isotope ratios",
        )


@dataclass(frozen=True)
class AffineGlobalO2Forcing:
    """One constant forcing interval for the slow global O2 reservoir."""

    duration_years: float
    biological: "YoungBiologicalO2Budget | PartitionedBiologicalO2Budget"
    photochemical: IsotopologueTendency
    label: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_years) or self.duration_years < 0.0:
            raise ValueError("forcing duration must be finite and non-negative")
        if not self.label:
            raise ValueError("global O2 forcing interval requires a label")


@dataclass(frozen=True)
class YoungBiologicalO2Budget:
    """Printed Young photosynthesis and respiration law on the global box."""

    photosynthetic_o16o16_mol_per_year: float
    respiration_rate_per_year: float
    tropospheric_fraction: float
    alpha_respiration_18: float
    beta_respiration_17: float
    source_alpha_18: float
    source_beta_17: float
    source: str

    def __post_init__(self) -> None:
        positive = (
            self.photosynthetic_o16o16_mol_per_year,
            self.respiration_rate_per_year,
            self.tropospheric_fraction,
            self.alpha_respiration_18,
            self.beta_respiration_17,
            self.source_alpha_18,
            self.source_beta_17,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("Young biological-budget parameters must be positive")
        if self.tropospheric_fraction > 1.0:
            raise ValueError("tropospheric O2 fraction cannot exceed one")
        if not self.source:
            raise ValueError("Young biological budget requires provenance")

    @property
    def effective_respiration_rates_per_year(self) -> np.ndarray:
        base = self.respiration_rate_per_year * self.tropospheric_fraction
        return np.asarray(
            (
                base,
                base * self.alpha_respiration_18**self.beta_respiration_17,
                base * self.alpha_respiration_18,
            ),
            dtype=float,
        )

    @property
    def photosynthetic_sources_mol_per_year(self) -> np.ndarray:
        major = self.photosynthetic_o16o16_mol_per_year
        return np.asarray(
            (
                major,
                major
                * 2.0
                * self.source_alpha_18**self.source_beta_17
                * R17_VSMOW,
                major * 2.0 * self.source_alpha_18 * R18_VSMOW,
            ),
            dtype=float,
        )

    def tendency(self, reservoir: GlobalO2Reservoir) -> IsotopologueTendency:
        values = (
            self.photosynthetic_sources_mol_per_year
            - self.effective_respiration_rates_per_year
            * reservoir.isotopologue_moles
        )
        return IsotopologueTendency(
            *map(float, values),
            source=f"{self.source}; global homogeneous reduction",
        )

    def scaled_turnover(
        self, scale: float, *, source: str
    ) -> "YoungBiologicalO2Budget":
        """Scale gross production and respiration as one O2-turnover budget.

        Scaling only one side would change the major atmospheric O2 inventory
        and would not represent a GPP sensitivity at fixed NPP/pO2 in the
        Young et al. sense. This operation preserves their source/sink ratio
        while changing the competition between biological turnover and the
        stratospheric isotope pump.
        """

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("biological turnover scale must be finite and positive")
        if not source:
            raise ValueError("scaled biological turnover requires provenance")
        return YoungBiologicalO2Budget(
            photosynthetic_o16o16_mol_per_year=(
                scale * self.photosynthetic_o16o16_mol_per_year
            ),
            respiration_rate_per_year=scale * self.respiration_rate_per_year,
            tropospheric_fraction=self.tropospheric_fraction,
            alpha_respiration_18=self.alpha_respiration_18,
            beta_respiration_17=self.beta_respiration_17,
            source_alpha_18=self.source_alpha_18,
            source_beta_17=self.source_beta_17,
            source=source,
        )

    def with_source_alpha_18(
        self, source_alpha_18: float, *, source: str
    ) -> "YoungBiologicalO2Budget":
        """Return the same budget with an explicitly sourced O2 source alpha."""

        if not np.isfinite(source_alpha_18) or source_alpha_18 <= 0.0:
            raise ValueError("source alpha18 must be finite and positive")
        if not source:
            raise ValueError("updated source alpha18 requires provenance")
        return YoungBiologicalO2Budget(
            photosynthetic_o16o16_mol_per_year=(
                self.photosynthetic_o16o16_mol_per_year
            ),
            respiration_rate_per_year=self.respiration_rate_per_year,
            tropospheric_fraction=self.tropospheric_fraction,
            alpha_respiration_18=self.alpha_respiration_18,
            beta_respiration_17=self.beta_respiration_17,
            source_alpha_18=source_alpha_18,
            source_beta_17=self.source_beta_17,
            source=source,
        )


@dataclass(frozen=True)
class BiologicalO2Compartment:
    """One source-backed photosynthesis/respiration compartment."""

    key: str
    photosynthetic_o16o16_mol_per_year: float
    respiration_rate_per_year: float
    tropospheric_fraction: float
    alpha_respiration_18: float
    beta_respiration_17: float
    source_alpha_18: float
    source_beta_17: float
    source: str

    def __post_init__(self) -> None:
        values = (
            self.photosynthetic_o16o16_mol_per_year,
            self.respiration_rate_per_year,
            self.tropospheric_fraction,
            self.alpha_respiration_18,
            self.beta_respiration_17,
            self.source_alpha_18,
            self.source_beta_17,
        )
        if not self.key or not self.source:
            raise ValueError("biological compartment requires a key and provenance")
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
            raise ValueError("biological compartment parameters must be positive")
        if self.tropospheric_fraction > 1.0:
            raise ValueError("tropospheric O2 fraction cannot exceed one")

    @property
    def effective_respiration_rates_per_year(self) -> np.ndarray:
        base = self.respiration_rate_per_year * self.tropospheric_fraction
        return np.asarray(
            (
                base,
                base * self.alpha_respiration_18**self.beta_respiration_17,
                base * self.alpha_respiration_18,
            ),
            dtype=float,
        )

    @property
    def photosynthetic_sources_mol_per_year(self) -> np.ndarray:
        major = self.photosynthetic_o16o16_mol_per_year
        return np.asarray(
            (
                major,
                major
                * 2.0
                * self.source_alpha_18**self.source_beta_17
                * R17_VSMOW,
                major * 2.0 * self.source_alpha_18 * R18_VSMOW,
            ),
            dtype=float,
        )

    def tendency(self, reservoir: GlobalO2Reservoir) -> IsotopologueTendency:
        values = (
            self.photosynthetic_sources_mol_per_year
            - self.effective_respiration_rates_per_year
            * reservoir.isotopologue_moles
        )
        return IsotopologueTendency(
            *map(float, values), source=f"{self.source}; {self.key} compartment"
        )


@dataclass(frozen=True)
class PartitionedBiologicalO2Budget:
    """Affine global O2 budget retaining explicit biological compartments."""

    compartments: tuple[BiologicalO2Compartment, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.compartments:
            raise ValueError("partitioned biological budget requires compartments")
        if len({item.key for item in self.compartments}) != len(self.compartments):
            raise ValueError("biological compartment keys must be unique")
        if not self.source:
            raise ValueError("partitioned biological budget requires provenance")

    @property
    def photosynthetic_sources_mol_per_year(self) -> np.ndarray:
        return np.sum(
            np.asarray(
                [item.photosynthetic_sources_mol_per_year for item in self.compartments]
            ),
            axis=0,
        )

    @property
    def effective_respiration_rates_per_year(self) -> np.ndarray:
        return np.sum(
            np.asarray(
                [item.effective_respiration_rates_per_year for item in self.compartments]
            ),
            axis=0,
        )

    def tendency(self, reservoir: GlobalO2Reservoir) -> IsotopologueTendency:
        return sum_tendencies(
            [item.tendency(reservoir) for item in self.compartments],
            source=f"{self.source}; summed biological compartments",
        )

    def scaled_turnover(
        self, scale: float, *, source: str
    ) -> "PartitionedBiologicalO2Budget":
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("biological turnover scale must be finite and positive")
        if not source:
            raise ValueError("scaled biological turnover requires provenance")
        return PartitionedBiologicalO2Budget(
            compartments=tuple(
                BiologicalO2Compartment(
                    key=item.key,
                    photosynthetic_o16o16_mol_per_year=(
                        scale * item.photosynthetic_o16o16_mol_per_year
                    ),
                    respiration_rate_per_year=scale * item.respiration_rate_per_year,
                    tropospheric_fraction=item.tropospheric_fraction,
                    alpha_respiration_18=item.alpha_respiration_18,
                    beta_respiration_17=item.beta_respiration_17,
                    source_alpha_18=item.source_alpha_18,
                    source_beta_17=item.source_beta_17,
                    source=item.source,
                )
                for item in self.compartments
            ),
            source=source,
        )

    def scaled_photosynthesis(
        self, scale: float, *, source: str
    ) -> "PartitionedBiologicalO2Budget":
        """Scale gross photosynthetic sources while retaining respiration."""

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("photosynthesis scale must be finite and positive")
        if not source:
            raise ValueError("scaled photosynthesis requires provenance")
        return PartitionedBiologicalO2Budget(
            compartments=tuple(
                BiologicalO2Compartment(
                    key=item.key,
                    photosynthetic_o16o16_mol_per_year=(
                        scale * item.photosynthetic_o16o16_mol_per_year
                    ),
                    respiration_rate_per_year=item.respiration_rate_per_year,
                    tropospheric_fraction=item.tropospheric_fraction,
                    alpha_respiration_18=item.alpha_respiration_18,
                    beta_respiration_17=item.beta_respiration_17,
                    source_alpha_18=item.source_alpha_18,
                    source_beta_17=item.source_beta_17,
                    source=item.source,
                )
                for item in self.compartments
            ),
            source=source,
        )

    def with_compartment_source_alpha_18(
        self, key: str, source_alpha_18: float, *, source: str
    ) -> "PartitionedBiologicalO2Budget":
        if key not in {item.key for item in self.compartments}:
            raise KeyError(key)
        if not np.isfinite(source_alpha_18) or source_alpha_18 <= 0.0:
            raise ValueError("source alpha18 must be finite and positive")
        if not source:
            raise ValueError("updated source alpha18 requires provenance")
        return PartitionedBiologicalO2Budget(
            compartments=tuple(
                BiologicalO2Compartment(
                    key=item.key,
                    photosynthetic_o16o16_mol_per_year=(
                        item.photosynthetic_o16o16_mol_per_year
                    ),
                    respiration_rate_per_year=item.respiration_rate_per_year,
                    tropospheric_fraction=item.tropospheric_fraction,
                    alpha_respiration_18=item.alpha_respiration_18,
                    beta_respiration_17=item.beta_respiration_17,
                    source_alpha_18=(
                        source_alpha_18 if item.key == key else item.source_alpha_18
                    ),
                    source_beta_17=item.source_beta_17,
                    source=item.source,
                )
                for item in self.compartments
            ),
            source=source,
        )


def isotopologue_tendency_from_atom_flux(
    atom_flux: OxygenAtomFlux,
) -> IsotopologueTendency:
    """Map an exact atom flux into the singly substituted O2 state."""

    major = 0.5 * (atom_flux.o16 - atom_flux.o17 - atom_flux.o18)
    tendency = IsotopologueTendency(
        o16o16=float(major),
        o16o17=float(atom_flux.o17),
        o16o18=float(atom_flux.o18),
        source=f"isotopologue mapping of {atom_flux.source}",
    )
    residual = tendency.atom_values - atom_flux.values
    scale = max(float(np.max(np.abs(atom_flux.values))), 1.0)
    if float(np.max(np.abs(residual))) > 64.0 * np.finfo(float).eps * scale:
        raise FloatingPointError("atom-to-isotopologue mapping failed conservation")
    return tendency


def sum_tendencies(
    tendencies: Iterable[IsotopologueTendency], *, source: str
) -> IsotopologueTendency:
    values = np.sum(np.asarray([item.values for item in tendencies]), axis=0)
    return IsotopologueTendency(*map(float, values), source=source)


def global_o2_tendency_from_affine_co2_rare_tendency(
    rare_co2_tendency_mol_per_year: np.ndarray,
    *,
    source: str,
) -> IsotopologueTendency:
    """Close a CO17O/CO18O R7 tendency against globally mixed O2.

    The rows must be CO17O and CO18O tendencies from an R7-only affine
    operator. Every gained rare CO2 molecule replaces one major CO2 molecule,
    so carbon-side 16O changes by ``-(d17 + d18)``. The external oxygen pool
    receives the exact opposite atom flux, which is then mapped to the global
    singly substituted O2 state.
    """

    tendency = np.asarray(rare_co2_tendency_mol_per_year, dtype=float)
    if tendency.ndim != 2 or tendency.shape[0] != 2:
        raise ValueError("rare CO2 tendency must have CO17O and CO18O rows")
    if not np.all(np.isfinite(tendency)):
        raise ValueError("rare CO2 tendency must be finite")
    if not source:
        raise ValueError("affine CO2-to-O2 closure requires provenance")
    d17 = float(fsum(float(value) for value in tendency[0]))
    d18 = float(fsum(float(value) for value in tendency[1]))
    carbon = np.asarray((-(d17 + d18), d17, d18), dtype=float)
    global_o2 = OxygenAtomFlux(
        *map(float, -carbon),
        source=f"equal-and-opposite external oxygen from {source}",
    )
    return isotopologue_tendency_from_atom_flux(global_o2)


def cap_delta17_tendency_permil_per_year(
    reservoir: GlobalO2Reservoir,
    tendency: IsotopologueTendency,
) -> float:
    """Instantaneous logarithmic Delta-prime-17O tendency."""

    d_major, d17, d18 = tendency.values
    d_delta17 = 1000.0 * (d17 / reservoir.o16o17 - d_major / reservoir.o16o16)
    d_delta18 = 1000.0 * (d18 / reservoir.o16o18 - d_major / reservoir.o16o16)
    return float(d_delta17 - 0.528 * d_delta18)


def frozen_photochemical_steady_state(
    biological: YoungBiologicalO2Budget | PartitionedBiologicalO2Budget,
    photochemical: IsotopologueTendency,
    *,
    source: str,
) -> GlobalO2Reservoir:
    """Solve the affine global box for a fixed photochemical transfer flux."""

    numerator = biological.photosynthetic_sources_mol_per_year + photochemical.values
    rates = biological.effective_respiration_rates_per_year
    if np.any(numerator <= 0.0):
        raise ValueError("fixed photochemical flux overwhelms an O2 source term")
    steady = numerator / rates
    return GlobalO2Reservoir(*map(float, steady), source=source)


def source_alpha18_for_target_delta18_prime(
    biological: YoungBiologicalO2Budget,
    photochemical: IsotopologueTendency,
    *,
    target_delta18_prime_permil: float,
) -> float:
    """Solve the photosynthetic source alpha18 from an O2 Dole-effect target.

    The solution is analytical for the affine global reservoir and uses the
    measured atmospheric delta-prime-18O independently of Delta-prime-17O.
    It therefore supports a two-isotope closure test without tuning to the
    triple-isotope anomaly.
    """

    if not np.isfinite(target_delta18_prime_permil):
        raise ValueError("target delta18-prime must be finite")
    production = biological.photosynthetic_o16o16_mol_per_year
    photo_major, _photo17, photo18 = photochemical.values
    target_ratio = R18_VSMOW * exp(target_delta18_prime_permil / 1000.0)
    numerator = (
        2.0
        * biological.alpha_respiration_18
        * (production + photo_major)
        * target_ratio
        - photo18
    )
    source_alpha = numerator / (2.0 * production * R18_VSMOW)
    if not np.isfinite(source_alpha) or source_alpha <= 0.0:
        raise ValueError("Dole-effect closure requires a non-positive source alpha18")
    return float(source_alpha)


def compartment_source_alpha18_for_target_delta18_prime(
    biological: PartitionedBiologicalO2Budget,
    photochemical: IsotopologueTendency,
    *,
    compartment_key: str,
    target_delta18_prime_permil: float,
) -> float:
    """Solve one compartment source alpha18 from the atmospheric Dole effect."""

    if not np.isfinite(target_delta18_prime_permil):
        raise ValueError("target delta18-prime must be finite")
    selected = [
        item for item in biological.compartments if item.key == compartment_key
    ]
    if len(selected) != 1:
        raise KeyError(compartment_key)
    target = selected[0]
    production = biological.photosynthetic_sources_mol_per_year[0]
    target_production = target.photosynthetic_o16o16_mol_per_year
    other_source18 = sum(
        item.photosynthetic_sources_mol_per_year[2]
        for item in biological.compartments
        if item.key != compartment_key
    )
    major_rate, _rate17, rate18 = biological.effective_respiration_rates_per_year
    photo_major, _photo17, photo18 = photochemical.values
    target_ratio = R18_VSMOW * exp(target_delta18_prime_permil / 1000.0)
    required_total_source18 = (
        2.0
        * target_ratio
        * (production + photo_major)
        * rate18
        / major_rate
        - photo18
    )
    source_alpha = (
        required_total_source18 - other_source18
    ) / (2.0 * target_production * R18_VSMOW)
    if not np.isfinite(source_alpha) or source_alpha <= 0.0:
        raise ValueError("partitioned Dole closure requires a positive source alpha18")
    return float(source_alpha)


def propagate_affine_global_o2(
    initial: GlobalO2Reservoir,
    biological: YoungBiologicalO2Budget | PartitionedBiologicalO2Budget,
    photochemical: IsotopologueTendency,
    *,
    duration_years: float,
    source: str,
) -> GlobalO2Reservoir:
    """Propagate a constant global O2 source/sink budget exactly.

    For each represented isotopologue the governing equation is
    ``dN/dt = S_bio + S_photo - k_resp N``.  The analytical exponential
    solution avoids time-step error over the multi-kyr atmospheric response.
    """

    if not np.isfinite(duration_years) or duration_years < 0.0:
        raise ValueError("propagation duration must be finite and non-negative")
    if not source:
        raise ValueError("propagated global O2 reservoir requires provenance")
    rates = biological.effective_respiration_rates_per_year
    total_source = (
        biological.photosynthetic_sources_mol_per_year + photochemical.values
    )
    if np.any(total_source <= 0.0):
        raise ValueError("combined forcing overwhelms an O2 source term")
    equilibrium = total_source / rates
    decay = np.exp(-rates * duration_years)
    propagated = equilibrium + (initial.isotopologue_moles - equilibrium) * decay
    if np.any(propagated <= 0.0):
        raise ValueError("global O2 propagation produced a non-positive inventory")
    return GlobalO2Reservoir(*map(float, propagated), source=source)


def propagate_piecewise_global_o2(
    initial: GlobalO2Reservoir,
    intervals: Iterable[AffineGlobalO2Forcing],
    *,
    source: str,
) -> tuple[GlobalO2Reservoir, ...]:
    """Propagate consecutive constant intervals and retain every boundary state."""

    if not source:
        raise ValueError("piecewise global O2 propagation requires provenance")
    states = [initial]
    current = initial
    for index, interval in enumerate(intervals, start=1):
        current = propagate_affine_global_o2(
            current,
            interval.biological,
            interval.photochemical,
            duration_years=interval.duration_years,
            source=f"{source}; interval {index}: {interval.label}",
        )
        states.append(current)
    return tuple(states)


def r7_global_transfer_from_tendency(
    tendency_mol_per_year: np.ndarray,
    *,
    species_names: tuple[str, ...],
    cell_mask: np.ndarray | None = None,
    source: str,
) -> R7GlobalTransfer:
    """Reduce a reaction-resolved R7 tendency to conservative atom fluxes."""

    tendency = np.asarray(tendency_mol_per_year, dtype=float)
    if tendency.ndim != 2 or tendency.shape[0] != len(species_names):
        raise ValueError("R7 tendency must have one row per named species")
    if not np.all(np.isfinite(tendency)):
        raise ValueError("R7 tendency must be finite")
    index = {name: position for position, name in enumerate(species_names)}
    required = (
        "CO2",
        "CO17O",
        "CO18O",
        "O",
        "O17",
        "O18",
        "O1D",
        "O17_1D",
        "O18_1D",
    )
    missing = tuple(name for name in required if name not in index)
    if missing:
        raise ValueError(f"R7 tendency is missing species: {missing}")
    if cell_mask is None:
        mask = np.ones(tendency.shape[1], dtype=bool)
    else:
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != (tendency.shape[1],):
            raise ValueError("R7 cell mask must align with the tendency grid")
        if not np.any(mask):
            raise ValueError("R7 cell mask cannot exclude every cell")

    def total(name: str) -> float:
        return float(fsum(float(value) for value in tendency[index[name], mask]))

    carbon_values = np.asarray(
        (
            fsum((2.0 * total("CO2"), total("CO17O"), total("CO18O"))),
            total("CO17O"),
            total("CO18O"),
        )
    )
    external_values = np.asarray(
        (
            fsum((total("O"), total("O1D"))),
            fsum((total("O17"), total("O17_1D"))),
            fsum((total("O18"), total("O18_1D"))),
        )
    )
    residual = carbon_values + external_values
    scale = max(
        float(np.max(np.abs(carbon_values))),
        float(np.max(np.abs(external_values))),
        1.0,
    )
    maximum_relative = float(np.max(np.abs(residual)) / scale)
    carbon = OxygenAtomFlux(*map(float, carbon_values), source=f"carbon side: {source}")
    global_o2 = OxygenAtomFlux(
        *map(float, external_values),
        source=f"O/O(1D) side assigned to globally mixed O2: {source}",
    )
    closure = OxygenAtomFlux(*map(float, residual), source=f"R7 closure: {source}")
    return R7GlobalTransfer(
        carbon_oxygen=carbon,
        global_o2=global_o2,
        closure_residual=closure,
        maximum_relative_closure_residual=maximum_relative,
    )
