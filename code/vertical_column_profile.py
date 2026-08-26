"""Source-backed vertical-profile definitions for the isotope column.

This module deliberately separates constraints reported in the literature from
the numerical grid eventually used by the model. It does not infer missing
layer edges, temperatures, pressures, or eddy diffusivities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerticalDomainReference:
    """A published description of an atmospheric model domain."""

    name: str
    coordinate: str
    lower_altitude_km: float
    upper_altitude_km: float
    reported_vertical_count: int
    source: str
    interpretation_note: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.coordinate or not self.source:
            raise ValueError("domain references require a name, coordinate, and source")
        if self.lower_altitude_km < 0.0:
            raise ValueError("lower altitude must be non-negative")
        if self.upper_altitude_km <= self.lower_altitude_km:
            raise ValueError("upper altitude must exceed lower altitude")
        if self.reported_vertical_count <= 0:
            raise ValueError("reported vertical count must be positive")


@dataclass(frozen=True)
class AltitudeConstraint:
    """An altitude range or transition explicitly described by a source."""

    name: str
    lower_altitude_km: float
    upper_altitude_km: float
    role: str
    source: str

    def __post_init__(self) -> None:
        if not self.name or not self.role or not self.source:
            raise ValueError("altitude constraints require a name, role, and source")
        if self.lower_altitude_km < 0.0:
            raise ValueError("lower altitude must be non-negative")
        if self.upper_altitude_km < self.lower_altitude_km:
            raise ValueError("upper altitude must not be below lower altitude")


@dataclass(frozen=True)
class VerticalCell:
    """One finite-volume cell with externally supplied physical properties."""

    lower_altitude_km: float
    upper_altitude_km: float
    air_moles: float
    number_density_molecules_cm3: float
    temperature_k: float
    pressure_center_bar: float
    eddy_diffusivity_cm2_per_s: float
    pressure_bottom_bar: float | None = None
    pressure_top_bar: float | None = None

    def __post_init__(self) -> None:
        if self.lower_altitude_km < 0.0:
            raise ValueError("cell lower altitude must be non-negative")
        if self.upper_altitude_km <= self.lower_altitude_km:
            raise ValueError("cell upper altitude must exceed lower altitude")
        if self.air_moles <= 0.0:
            raise ValueError("cell air inventory must be positive")
        if self.number_density_molecules_cm3 <= 0.0:
            raise ValueError("cell number density must be positive")
        if self.temperature_k <= 0.0:
            raise ValueError("cell temperature must be positive")
        if self.pressure_center_bar <= 0.0:
            raise ValueError("cell-center pressure must be positive")
        if self.eddy_diffusivity_cm2_per_s < 0.0:
            raise ValueError("eddy diffusivity must be non-negative")
        boundaries = (self.pressure_bottom_bar, self.pressure_top_bar)
        if (boundaries[0] is None) != (boundaries[1] is None):
            raise ValueError("pressure boundaries must either both be supplied or both be omitted")
        if boundaries[0] is not None:
            if boundaries[0] <= boundaries[1] or boundaries[1] <= 0.0:
                raise ValueError("cell pressure must decrease upward and remain positive")
            if not boundaries[0] > self.pressure_center_bar > boundaries[1]:
                raise ValueError("cell-center pressure must lie between its boundaries")


@dataclass(frozen=True)
class ValidatedVerticalProfile:
    """A contiguous finite-volume profile with complete provenance."""

    name: str
    cells: tuple[VerticalCell, ...]
    atmospheric_state_source: str
    eddy_diffusivity_source: str

    def __post_init__(self) -> None:
        if not self.name or not self.atmospheric_state_source or not self.eddy_diffusivity_source:
            raise ValueError("profiles require names and physical-data provenance")
        if not self.cells:
            raise ValueError("a vertical profile must contain at least one cell")
        for lower, upper in zip(self.cells[:-1], self.cells[1:]):
            if not np.isclose(
                lower.upper_altitude_km,
                upper.lower_altitude_km,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("vertical cells must be contiguous")
            if lower.pressure_center_bar <= upper.pressure_center_bar:
                raise ValueError("cell-center pressure must decrease upward")
            if lower.pressure_top_bar is not None and upper.pressure_bottom_bar is not None:
                if not np.isclose(
                    lower.pressure_top_bar,
                    upper.pressure_bottom_bar,
                    rtol=1.0e-12,
                    atol=0.0,
                ):
                    raise ValueError("pressure boundaries must be continuous")

    @property
    def lower_altitude_km(self) -> float:
        return self.cells[0].lower_altitude_km

    @property
    def upper_altitude_km(self) -> float:
        return self.cells[-1].upper_altitude_km

    @property
    def air_moles(self) -> np.ndarray:
        return np.asarray([cell.air_moles for cell in self.cells], dtype=float)

    def whole_cell_indices(self, lower_altitude_km: float, upper_altitude_km: float) -> np.ndarray:
        """Select a diagnostic interval only when it follows exact cell edges."""

        edges = np.asarray(
            [self.cells[0].lower_altitude_km]
            + [cell.upper_altitude_km for cell in self.cells],
            dtype=float,
        )
        lower_match = np.flatnonzero(np.isclose(edges, lower_altitude_km, rtol=0.0, atol=1.0e-12))
        upper_match = np.flatnonzero(np.isclose(edges, upper_altitude_km, rtol=0.0, atol=1.0e-12))
        if lower_match.size != 1 or upper_match.size != 1:
            raise ValueError("diagnostic bounds must coincide with exact vertical cell edges")
        first = int(lower_match[0])
        stop = int(upper_match[0])
        if stop <= first:
            raise ValueError("diagnostic upper bound must exceed lower bound")
        return np.arange(first, stop, dtype=int)

    def air_weighted_mean(
        self,
        values: np.ndarray,
        lower_altitude_km: float,
        upper_altitude_km: float,
    ) -> float:
        """Return an air-inventory-weighted mean over exact complete cells."""

        data = np.asarray(values, dtype=float)
        if data.shape != (len(self.cells),):
            raise ValueError("values must contain one entry per vertical cell")
        selected = self.whole_cell_indices(lower_altitude_km, upper_altitude_km)
        weights = self.air_moles[selected]
        return float(np.average(data[selected], weights=weights))


LIANG_2006_DOMAIN = VerticalDomainReference(
    name="Liang 2006 isotope photochemistry column",
    coordinate="altitude",
    lower_altitude_km=0.0,
    upper_altitude_km=130.0,
    reported_vertical_count=66,
    source="Liang et al. (2006), Section 3, paragraph 10",
    interpretation_note=(
        "The paper calls these 66 layers evenly distributed from the surface to "
        "130 km but does not state the cell-edge convention in the extracted text."
    ),
)

LIANG_2008_CORE_DOMAIN = VerticalDomainReference(
    name="Caltech/JPL 2-D chemistry transport core domain",
    coordinate="logarithmic pressure",
    lower_altitude_km=0.0,
    upper_altitude_km=80.0,
    reported_vertical_count=40,
    source="Liang et al. (2008), Section 4, paragraph 22",
    interpretation_note="The reported model top is approximate.",
)

LIANG_2008_EXTENDED_DOMAIN = VerticalDomainReference(
    name="Caltech/JPL 2-D chemistry transport extended domain",
    coordinate="logarithmic pressure",
    lower_altitude_km=0.0,
    upper_altitude_km=130.0,
    reported_vertical_count=66,
    source="Liang et al. (2008), Section 4, paragraph 22",
    interpretation_note="Extended configuration used for mesospheric studies.",
)

YOUNG_BULK_STRATOSPHERE = AltitudeConstraint(
    name="Young bulk stratospheric CO2 isotope diagnostic",
    lower_altitude_km=10.0,
    upper_altitude_km=60.0,
    role="Column-density-weighted comparison interval for Delta-prime-17O of CO2",
    source="Young et al. (2014), Equation 28",
)

ISOTOPE_PROCESS_CONSTRAINTS = (
    AltitudeConstraint(
        name="stratospheric CO2 isotope comparison",
        lower_altitude_km=30.0,
        upper_altitude_km=60.0,
        role="Range used for modeled and measured CO2 three-isotope slopes",
        source="Liang et al. (2007), Results",
    ),
    AltitudeConstraint(
        name="O(1D) abundance maximum",
        lower_altitude_km=45.0,
        upper_altitude_km=45.0,
        role="Approximate altitude of peak O(1D) and quoted chemistry/transport timescales",
        source="Liang et al. (2007), transport discussion",
    ),
    AltitudeConstraint(
        name="CO2 isotope-slope transition",
        lower_altitude_km=55.0,
        upper_altitude_km=55.0,
        role="Approximate transition in the modeled CO2 three-isotope slope",
        source="Liang et al. (2007), Figure 3 caption",
    ),
    AltitudeConstraint(
        name="O2 photolysis dominance",
        lower_altitude_km=70.0,
        upper_altitude_km=70.0,
        role="Approximate lower altitude above which O2 photolysis dominates O(1D) production",
        source="Liang et al. (2007), Sources of O(1D)",
    ),
    AltitudeConstraint(
        name="homopause transition",
        lower_altitude_km=90.0,
        upper_altitude_km=90.0,
        role="Approximate altitude above which molecular diffusion dominates",
        source="Liang et al. (2007), Figure 3 caption; Liang et al. (2008), Section 5",
    ),
)


def published_vertical_references() -> tuple[VerticalDomainReference, ...]:
    return (
        LIANG_2006_DOMAIN,
        LIANG_2008_CORE_DOMAIN,
        LIANG_2008_EXTENDED_DOMAIN,
    )
