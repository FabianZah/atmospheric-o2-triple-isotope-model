"""Published geometry and transport constraints for the Caltech/JPL 2-D CTM.

This module encodes only quantities printed in Morgan et al. (2004) Appendix A
or in the cited transport comparison by Jiang et al. (2004). It does not infer
the unreported latitude-pressure fields for residual circulation, Kyy, or Kzz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL = 0.0289647
EARTH_RADIUS_M = 6.371e6
STANDARD_GRAVITY_M_PER_S2 = 9.80665


@dataclass(frozen=True)
class LatitudePressureGridReference:
    name: str
    latitude_box_count: int
    vertical_layer_count: int
    surface_pressure_mbar: float
    top_pressure_mbar: float
    scale_height_km: float
    source: str

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise ValueError("grid reference requires a name and source")
        if self.latitude_box_count <= 0 or self.vertical_layer_count <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.surface_pressure_mbar <= self.top_pressure_mbar:
            raise ValueError("surface pressure must exceed top pressure")
        if self.top_pressure_mbar <= 0.0 or self.scale_height_km <= 0.0:
            raise ValueError("top pressure and scale height must be positive")

    @property
    def latitude_edges_degrees(self) -> np.ndarray:
        return np.linspace(-90.0, 90.0, self.latitude_box_count + 1)

    @property
    def latitude_centers_degrees(self) -> np.ndarray:
        edges = self.latitude_edges_degrees
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def altitude_edges_km(self) -> np.ndarray:
        model_top_km = self.scale_height_km * np.log(
            self.surface_pressure_mbar / self.top_pressure_mbar
        )
        return np.linspace(0.0, model_top_km, self.vertical_layer_count + 1)

    @property
    def altitude_centers_km(self) -> np.ndarray:
        edges = self.altitude_edges_km
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def pressure_edges_mbar(self) -> np.ndarray:
        return self.surface_pressure_mbar * np.exp(
            -self.altitude_edges_km / self.scale_height_km
        )

    @property
    def pressure_centers_mbar(self) -> np.ndarray:
        return self.surface_pressure_mbar * np.exp(
            -self.altitude_centers_km / self.scale_height_km
        )

    def tropopause_pressure_mbar(self, latitude_degrees: float) -> float:
        """Return Morgan's season-independent, latitude-belt tropopause."""

        latitude = abs(float(latitude_degrees))
        if latitude > 90.0:
            raise ValueError("latitude must lie between -90 and 90 degrees")
        if latitude <= 30.0:
            return 100.0
        if latitude <= 60.0:
            return 200.0
        return 300.0


@dataclass(frozen=True)
class CirculationMassFluxConstraint:
    key: str
    mass_flux_kg_per_s: float
    surface: str
    source: str
    interpretation: str

    def __post_init__(self) -> None:
        if not self.key or not self.surface or not self.source or not self.interpretation:
            raise ValueError("circulation constraints require complete provenance")
        if self.mass_flux_kg_per_s <= 0.0:
            raise ValueError("circulation mass flux must be positive")

    def air_flux_mol_per_year(
        self,
        molar_mass_kg_per_mol: float = STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    ) -> float:
        if molar_mass_kg_per_mol <= 0.0:
            raise ValueError("air molar mass must be positive")
        return (
            self.mass_flux_kg_per_s
            * SECONDS_PER_YEAR
            / molar_mass_kg_per_mol
        )


MORGAN_2004_GRID = LatitudePressureGridReference(
    name="Caltech/JPL 2-D chemistry-transport grid",
    latitude_box_count=18,
    vertical_layer_count=40,
    surface_pressure_mbar=1000.0,
    top_pressure_mbar=0.01,
    scale_height_km=80.0 / np.log(1000.0 / 0.01),
    source="Morgan et al. (2004), Appendix A, paragraphs 44-46",
)


MORGAN_2004_CONTINUITY_TERMS = (
    "time tendency of mixing ratio",
    "meridional advection",
    "vertical advection in log-pressure altitude",
    "meridional eddy diffusion Kyy",
    "vertical eddy diffusion Kzz",
    "net chemical production divided by ambient number density",
)


SHIA_1989_TROPOPAUSE_FLUX = CirculationMassFluxConstraint(
    key="shia_1989_annual_mean_tropopause_flux",
    mass_flux_kg_per_s=9.3e9,
    surface="model tropopause",
    source="Jiang et al. (2004), paragraph 13; comparison to Shia et al. (1989)",
    interpretation=(
        "Annual-mean circulation flux reported as consistent with bomb-14C data."
    ),
)


JIANG_2004_400K_FLUX = CirculationMassFluxConstraint(
    key="jiang_2004_annual_mean_400k_flux",
    mass_flux_kg_per_s=14.0e9,
    surface="400 K isentropic surface near the tropical tropopause",
    source="Jiang et al. (2004), paragraph 13",
    interpretation=(
        "Annual-mean isentropic circulation flux derived from NCEP2 transport fields."
    ),
)


CALTECH_JPL_CIRCULATION_SCALE_CONSTRAINTS = (
    SHIA_1989_TROPOPAUSE_FLUX,
    JIANG_2004_400K_FLUX,
)


def circulation_scale_rows() -> tuple[dict[str, float | str], ...]:
    """Return citation-ready circulation constraints and diagnostic conversions."""

    return tuple(
        {
            "key": constraint.key,
            "mass_flux_kg_per_s": constraint.mass_flux_kg_per_s,
            "air_flux_mol_per_year": constraint.air_flux_mol_per_year(),
            "surface": constraint.surface,
            "source": constraint.source,
            "interpretation": constraint.interpretation,
            "conversion_air_molar_mass_kg_per_mol": (
                STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL
            ),
        }
        for constraint in CALTECH_JPL_CIRCULATION_SCALE_CONSTRAINTS
    )


def hydrostatic_air_moles(
    latitude_edges_degrees: np.ndarray,
    pressure_edges_pa: np.ndarray,
    earth_radius_m: float = EARTH_RADIUS_M,
    gravity_m_per_s2: float = STANDARD_GRAVITY_M_PER_S2,
    molar_mass_kg_per_mol: float = STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL,
) -> np.ndarray:
    """Return dry-air moles in latitude-pressure finite-volume cells.

    Latitude edges must increase south to north. Pressure edges must decrease
    from the lower to upper atmosphere. The hydrostatic cell mass is
    ``area * pressure_thickness / gravity``. Output uses
    ``(latitude, vertical)`` order.
    """

    latitude = np.asarray(latitude_edges_degrees, dtype=float)
    pressure = np.asarray(pressure_edges_pa, dtype=float)
    if latitude.ndim != 1 or len(latitude) < 2:
        raise ValueError("latitude edges must be a one-dimensional edge array")
    if pressure.ndim != 1 or len(pressure) < 2:
        raise ValueError("pressure edges must be a one-dimensional edge array")
    if not np.all(np.isfinite(latitude)) or not np.all(np.diff(latitude) > 0.0):
        raise ValueError("latitude edges must be finite and strictly increasing")
    if latitude[0] < -90.0 or latitude[-1] > 90.0:
        raise ValueError("latitude edges must remain between -90 and 90 degrees")
    if not np.all(np.isfinite(pressure)) or not np.all(np.diff(pressure) < 0.0):
        raise ValueError("pressure edges must be finite and strictly decreasing")
    if pressure[-1] < 0.0:
        raise ValueError("pressure must be non-negative")
    if earth_radius_m <= 0.0 or gravity_m_per_s2 <= 0.0 or molar_mass_kg_per_mol <= 0.0:
        raise ValueError("physical conversion constants must be positive")

    latitude_radians = np.deg2rad(latitude)
    band_areas_m2 = (
        2.0
        * np.pi
        * earth_radius_m**2
        * np.diff(np.sin(latitude_radians))
    )
    pressure_thickness_pa = -np.diff(pressure)
    air_mass_kg = (
        band_areas_m2[:, None]
        * pressure_thickness_pa[None, :]
        / gravity_m_per_s2
    )
    return air_mass_kg / molar_mass_kg_per_mol
