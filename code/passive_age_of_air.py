"""Passive mean age of air on the ERA5 latitude-pressure transport grid.

This module combines two independently sourced transport components:

* the 2010-2019 ERA5 transformed-Eulerian-mean mass streamfunction from
  Serva (2022), and
* the pinned Photochem v0.6.7 ModernEarth vertical eddy-diffusion profile.

No isotope chemistry or fitted Young-model parameter enters this calculation.
Tropospheric cells are held at age zero using the latitude-dependent
tropopause definition printed by Morgan et al. (2004). The steady mean age in
the remaining cells solves

    L Gamma = -1,

where L is the conservative transport operator written for mixing ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from conservative_circulation_transport import (
    ConservativeCirculationNetwork,
    circulation_network_with_lower_reservoir_row,
    combined_transport_matrix_per_year,
    latitude_vertical_cell_name,
)
from conservative_column_transport import (
    AtmosphericLayer,
    ConservativeColumn,
    ExchangeInterface,
)
from convective_plume_transport import ConvectivePlumeGrid
from era5_kyy_reference import Era5KyyClimatology
from era5_tem_reference import Era5TemClimatology
from meridional_diffusion_transport import meridional_eddy_diffusion_operator
from meridional_transport_reference import (
    EARTH_RADIUS_M,
    MORGAN_2004_GRID,
    SECONDS_PER_YEAR,
    STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    hydrostatic_air_moles,
)
from photochem_profile import AVOGADRO_PER_MOL
from vertical_column_profile import ValidatedVerticalProfile


@dataclass(frozen=True)
class UpperBoundaryClosure:
    """Magnitude of the ERA5 transport removed at the selected model top."""

    pressure_pa: float
    maximum_streamfunction_kg_per_s: float
    maximum_domain_streamfunction_kg_per_s: float
    maximum_relative_to_domain: float
    summed_absolute_normal_flux_kg_per_s: float
    summed_absolute_lower_normal_flux_kg_per_s: float
    normal_flux_relative_to_lower: float


@dataclass(frozen=True)
class PassiveAgeTransport:
    """Fully assembled transport operator and its finite-volume coordinates."""

    latitude_edges_degrees: np.ndarray
    pressure_edges_pa: np.ndarray
    altitude_centers_km: np.ndarray
    air_moles: np.ndarray
    circulation: ConservativeCirculationNetwork
    vertical_diffusion: ConservativeColumn
    meridional_diffusion: ConservativeColumn | None
    upper_boundary_closure: UpperBoundaryClosure
    tropopause_pressure_pa: np.ndarray
    reset_mask: np.ndarray
    convection: ConvectivePlumeGrid | None = None

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_edges_degrees, dtype=float)
        pressure = np.asarray(self.pressure_edges_pa, dtype=float)
        altitude = np.asarray(self.altitude_centers_km, dtype=float)
        air = np.asarray(self.air_moles, dtype=float)
        tropopause = np.asarray(self.tropopause_pressure_pa, dtype=float)
        reset = np.asarray(self.reset_mask, dtype=bool)
        expected = (len(latitude) - 1, len(pressure) - 1)
        if air.shape != expected or reset.shape != expected:
            raise ValueError("air inventory and reset mask must match the grid cells")
        if altitude.shape != (expected[1],):
            raise ValueError("altitude centers must match the vertical cells")
        if tropopause.shape != (expected[0],):
            raise ValueError("tropopause pressure must match the latitude cells")
        if self.circulation.air_moles.shape != (air.size,):
            raise ValueError("circulation operator size does not match the age grid")
        if self.vertical_diffusion.air_moles.shape != (air.size,):
            raise ValueError("diffusion operator size does not match the age grid")
        if not np.allclose(
            self.circulation.air_moles,
            air.ravel(),
            rtol=1.0e-13,
            atol=0.0,
        ):
            raise ValueError("circulation inventory order does not match the age grid")
        if not np.allclose(
            self.vertical_diffusion.air_moles,
            air.ravel(),
            rtol=1.0e-13,
            atol=0.0,
        ):
            raise ValueError("diffusion inventory order does not match the age grid")
        if self.meridional_diffusion is not None:
            if self.meridional_diffusion.air_moles.shape != (air.size,):
                raise ValueError("Kyy operator size does not match the age grid")
            if not np.allclose(
                self.meridional_diffusion.air_moles,
                air.ravel(),
                rtol=1.0e-13,
                atol=0.0,
            ):
                raise ValueError("Kyy inventory order does not match the age grid")
        if self.convection is not None:
            if self.convection.air_moles.shape != (air.size,):
                raise ValueError("convection operator size does not match the age grid")
            if not np.allclose(
                self.convection.air_moles,
                air.ravel(),
                rtol=1.0e-13,
                atol=0.0,
            ):
                raise ValueError("convection inventory order does not match the age grid")

    @property
    def latitude_centers_degrees(self) -> np.ndarray:
        edges = np.asarray(self.latitude_edges_degrees, dtype=float)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def pressure_centers_pa(self) -> np.ndarray:
        edges = np.asarray(self.pressure_edges_pa, dtype=float)
        return np.sqrt(edges[:-1] * edges[1:])

    def inventory_transport_matrix_per_year(self) -> np.ndarray:
        operators = [self.circulation, self.vertical_diffusion]
        if self.meridional_diffusion is not None:
            operators.append(self.meridional_diffusion)
        if self.convection is not None:
            operators.append(self.convection)
        return combined_transport_matrix_per_year(*operators)

    def mixing_ratio_transport_matrix_per_year(self) -> np.ndarray:
        inventory = self.inventory_transport_matrix_per_year()
        air = np.asarray(self.air_moles, dtype=float).ravel()
        return (inventory * air[None, :]) / air[:, None]


@dataclass(frozen=True)
class PassiveAgeResult:
    """Steady mean-age field and linear-system diagnostics."""

    transport: PassiveAgeTransport
    mean_age_years: np.ndarray
    active_equation_max_residual_years_per_year: float
    condition_number: float

    def age_at_altitude(
        self,
        altitude_km: float,
        latitude_degrees: float = 0.0,
    ) -> float:
        """Linearly interpolate age in altitude and then in latitude."""

        altitude = np.asarray(self.transport.altitude_centers_km, dtype=float)
        latitude = self.transport.latitude_centers_degrees
        field = np.asarray(self.mean_age_years, dtype=float)
        if not altitude[0] <= altitude_km <= altitude[-1]:
            raise ValueError("requested altitude lies outside the age grid")
        if not latitude[0] <= latitude_degrees <= latitude[-1]:
            raise ValueError("requested latitude lies outside the age grid")
        vertical = np.asarray(
            [
                np.interp(altitude_km, altitude, field[index, :])
                for index in range(len(latitude))
            ],
            dtype=float,
        )
        return float(np.interp(latitude_degrees, latitude, vertical))


def _exact_pressure_slice(
    pressure_edges_pa: np.ndarray,
    bottom_pressure_pa: float,
    top_pressure_pa: float,
) -> slice:
    pressure = np.asarray(pressure_edges_pa, dtype=float)
    bottom = np.flatnonzero(
        np.isclose(pressure, bottom_pressure_pa, rtol=0.0, atol=1.0e-12)
    )
    top = np.flatnonzero(
        np.isclose(pressure, top_pressure_pa, rtol=0.0, atol=1.0e-12)
    )
    if bottom.size != 1 or top.size != 1:
        raise ValueError("age-domain boundaries must be native ERA5 pressure levels")
    first = int(bottom[0])
    last = int(top[0])
    if last <= first:
        raise ValueError("top pressure must be above the bottom pressure")
    return slice(first, last + 1)


def _upper_boundary_closure(
    raw_streamfunction_kg_per_s: np.ndarray,
    pressure_pa: float,
) -> UpperBoundaryClosure:
    streamfunction = np.asarray(raw_streamfunction_kg_per_s, dtype=float)
    top_normal_flux = np.diff(streamfunction[:, -1])
    lower_normal_flux = np.diff(streamfunction[:, 0])
    top_absolute = float(np.sum(np.abs(top_normal_flux)))
    lower_absolute = float(np.sum(np.abs(lower_normal_flux)))
    domain_maximum = float(np.max(np.abs(streamfunction)))
    top_maximum = float(np.max(np.abs(streamfunction[:, -1])))
    if domain_maximum == 0.0:
        maximum_relative = 0.0
    else:
        maximum_relative = top_maximum / domain_maximum
    if lower_absolute == 0.0:
        normal_relative = 0.0 if top_absolute == 0.0 else float("inf")
    else:
        normal_relative = top_absolute / lower_absolute
    return UpperBoundaryClosure(
        pressure_pa=float(pressure_pa),
        maximum_streamfunction_kg_per_s=top_maximum,
        maximum_domain_streamfunction_kg_per_s=domain_maximum,
        maximum_relative_to_domain=maximum_relative,
        summed_absolute_normal_flux_kg_per_s=top_absolute,
        summed_absolute_lower_normal_flux_kg_per_s=lower_absolute,
        normal_flux_relative_to_lower=normal_relative,
    )


def _interpolate_photochem_profile(
    profile: ValidatedVerticalProfile,
    target_pressure_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_pressure_pa = np.asarray(
        [cell.pressure_center_bar * 1.0e5 for cell in profile.cells],
        dtype=float,
    )
    source_altitude_km = np.asarray(
        [
            0.5 * (cell.lower_altitude_km + cell.upper_altitude_km)
            for cell in profile.cells
        ],
        dtype=float,
    )
    source_number_density = np.asarray(
        [cell.number_density_molecules_cm3 for cell in profile.cells],
        dtype=float,
    )
    source_kzz = np.asarray(
        [cell.eddy_diffusivity_cm2_per_s for cell in profile.cells],
        dtype=float,
    )
    target = np.asarray(target_pressure_pa, dtype=float)
    log_source_pressure = np.log(source_pressure_pa[::-1])
    log_target_pressure = np.log(target)
    if (
        np.min(log_target_pressure) < log_source_pressure[0]
        or np.max(log_target_pressure) > log_source_pressure[-1]
    ):
        raise ValueError("target pressure lies outside the Photochem profile")
    altitude = np.interp(
        log_target_pressure,
        log_source_pressure,
        source_altitude_km[::-1],
    )
    number_density = np.exp(
        np.interp(
            log_target_pressure,
            log_source_pressure,
            np.log(source_number_density[::-1]),
        )
    )
    kzz = np.exp(
        np.interp(
            log_target_pressure,
            log_source_pressure,
            np.log(source_kzz[::-1]),
        )
    )
    return altitude, number_density, kzz


def _vertical_diffusion_operator(
    air_moles: np.ndarray,
    latitude_edges_degrees: np.ndarray,
    pressure_edges_pa: np.ndarray,
    profile: ValidatedVerticalProfile,
) -> tuple[ConservativeColumn, np.ndarray]:
    pressure_edges = np.asarray(pressure_edges_pa, dtype=float)
    pressure_centers = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    altitude_centers, _, _ = _interpolate_photochem_profile(
        profile,
        pressure_centers,
    )
    interface_altitude, interface_number_density, interface_kzz = (
        _interpolate_photochem_profile(profile, pressure_edges[1:-1])
    )

    latitude_radians = np.deg2rad(latitude_edges_degrees)
    band_area_factor = 2.0 * np.pi * np.diff(np.sin(latitude_radians))
    layers = tuple(
        AtmosphericLayer(
            latitude_vertical_cell_name(latitude_index, vertical_index),
            float(air_moles[latitude_index, vertical_index]),
        )
        for latitude_index in range(air_moles.shape[0])
        for vertical_index in range(air_moles.shape[1])
    )
    interfaces: list[ExchangeInterface] = []
    source = (
        f"{profile.eddy_diffusivity_source}; finite-volume pressure-grid "
        "mapping in passive_age_of_air"
    )
    for latitude_index, area_factor in enumerate(band_area_factor):
        for vertical_edge in range(1, air_moles.shape[1]):
            interface_index = vertical_edge - 1
            radius_cm = (
                EARTH_RADIUS_M * 100.0
                + interface_altitude[interface_index] * 1.0e5
            )
            area_cm2 = area_factor * radius_cm**2
            center_spacing_cm = (
                altitude_centers[vertical_edge]
                - altitude_centers[vertical_edge - 1]
            ) * 1.0e5
            gross_air_flux = (
                interface_kzz[interface_index]
                * interface_number_density[interface_index]
                * area_cm2
                / center_spacing_cm
                * SECONDS_PER_YEAR
                / AVOGADRO_PER_MOL
            )
            interfaces.append(
                ExchangeInterface(
                    first_layer=latitude_vertical_cell_name(
                        latitude_index,
                        vertical_edge - 1,
                    ),
                    second_layer=latitude_vertical_cell_name(
                        latitude_index,
                        vertical_edge,
                    ),
                    gross_air_flux_mol_per_year=float(gross_air_flux),
                    source=source,
                )
            )
    return ConservativeColumn(layers=layers, interfaces=tuple(interfaces)), altitude_centers


def build_passive_age_transport(
    climatology: Era5TemClimatology,
    photochem_profile: ValidatedVerticalProfile,
    *,
    kyy_climatology: Era5KyyClimatology | None = None,
    bottom_pressure_pa: float = 30000.0,
    top_pressure_pa: float = 3.0,
    lower_reservoir_pressure_pa: float = 100000.0,
) -> PassiveAgeTransport:
    """Build ERA5 circulation, Photochem Kzz, and optional ERA5 Kyy."""

    latitude_edges, native_pressure, native_streamfunction = (
        climatology.ten_degree_native_pressure_nodes()
    )
    selected = _exact_pressure_slice(
        native_pressure,
        bottom_pressure_pa,
        top_pressure_pa,
    )
    atmospheric_pressure_edges = native_pressure[selected].copy()
    raw_streamfunction = native_streamfunction[:, selected].copy()

    # Meridional mass flux through a pole is exactly zero by symmetry. ERA5
    # interpolation leaves small nonzero endpoint values, so enforce this
    # physical boundary before measuring and closing the upper truncation.
    raw_streamfunction[0, :] = 0.0
    raw_streamfunction[-1, :] = 0.0
    closure = _upper_boundary_closure(
        raw_streamfunction,
        atmospheric_pressure_edges[-1],
    )
    closed_streamfunction = raw_streamfunction.copy()
    closed_streamfunction[:, -1] = 0.0

    atmospheric_air = hydrostatic_air_moles(
        latitude_edges,
        atmospheric_pressure_edges,
    )
    lower_pressure_edges = np.asarray(
        [lower_reservoir_pressure_pa, atmospheric_pressure_edges[0]],
        dtype=float,
    )
    lower_air = hydrostatic_air_moles(
        latitude_edges,
        lower_pressure_edges,
    )[:, 0]
    # DynVarMIP Eq. A8 defines positive Psi as northward transport integrated
    # from p=0 down to the stated pressure. The generic conservative-network
    # corner convention is the negative of that definition: its northward
    # face transport is Psi(upper)-Psi(lower), and its upward face transport
    # is Psi(south)-Psi(north). Convert the sign once at this adapter boundary.
    streamfunction_mol_per_year = (
        -closed_streamfunction
        * SECONDS_PER_YEAR
        / STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL
    )
    circulation = circulation_network_with_lower_reservoir_row(
        atmospheric_air_moles=atmospheric_air,
        lower_reservoir_air_moles=lower_air,
        streamfunction_mol_per_year=streamfunction_mol_per_year,
        source=(
            "Serva (2022), ERA5 TEM 2010-2019 month-duration-weighted "
            "climatology, DOI 10.5281/zenodo.7081721; DynVarMIP mass "
            "streamfunction sign from Gerber and Manzini (2016), Eq. A8"
        ),
    )

    pressure_edges = np.concatenate(
        ([lower_reservoir_pressure_pa], atmospheric_pressure_edges)
    )
    air_moles = np.concatenate((lower_air[:, None], atmospheric_air), axis=1)
    vertical_diffusion, altitude_centers = _vertical_diffusion_operator(
        air_moles,
        latitude_edges,
        pressure_edges,
        photochem_profile,
    )
    meridional_diffusion = None
    if kyy_climatology is not None:
        kyy_field = kyy_climatology.model_face_field(
            latitude_edges_degrees=latitude_edges,
            pressure_edges_pa=pressure_edges,
        )
        meridional_diffusion = meridional_eddy_diffusion_operator(
            air_moles=air_moles,
            latitude_edges_degrees=latitude_edges,
            pressure_edges_pa=pressure_edges,
            kyy_m2_per_s=kyy_field,
            source=kyy_climatology.source,
        )
    latitude_centers = 0.5 * (latitude_edges[:-1] + latitude_edges[1:])
    tropopause_pressure_pa = np.asarray(
        [
            MORGAN_2004_GRID.tropopause_pressure_mbar(latitude) * 100.0
            for latitude in latitude_centers
        ],
        dtype=float,
    )
    pressure_centers = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    reset_mask = (
        pressure_centers[None, :]
        >= tropopause_pressure_pa[:, None]
    )
    return PassiveAgeTransport(
        latitude_edges_degrees=latitude_edges,
        pressure_edges_pa=pressure_edges,
        altitude_centers_km=altitude_centers,
        air_moles=air_moles,
        circulation=circulation,
        vertical_diffusion=vertical_diffusion,
        meridional_diffusion=meridional_diffusion,
        upper_boundary_closure=closure,
        tropopause_pressure_pa=tropopause_pressure_pa,
        reset_mask=reset_mask,
    )


def solve_passive_mean_age(
    transport: PassiveAgeTransport,
) -> PassiveAgeResult:
    """Solve the steady mean-age equation with zero-age tropospheric air."""

    operator = transport.mixing_ratio_transport_matrix_per_year()
    reset = np.asarray(transport.reset_mask, dtype=bool).ravel()
    active = ~reset
    active_operator = operator[np.ix_(active, active)]
    active_age = np.linalg.solve(
        active_operator,
        -np.ones(int(np.count_nonzero(active)), dtype=float),
    )
    if np.min(active_age) < -1.0e-10:
        raise ValueError("passive-age solution contains negative ages")
    age = np.zeros(operator.shape[0], dtype=float)
    age[active] = active_age
    equation_residual = (
        np.ones(int(np.count_nonzero(active)), dtype=float)
        + operator[active, :] @ age
    )
    return PassiveAgeResult(
        transport=transport,
        mean_age_years=age.reshape(transport.air_moles.shape),
        active_equation_max_residual_years_per_year=float(
            np.max(np.abs(equation_residual))
        ),
        condition_number=float(np.linalg.cond(active_operator)),
    )


def select_passive_transport_components(
    transport: PassiveAgeTransport,
    *,
    include_circulation: bool = True,
    include_vertical_diffusion: bool = True,
    include_meridional_diffusion: bool = True,
    include_convection: bool = True,
    reset_mask: np.ndarray | None = None,
) -> PassiveAgeTransport:
    """Return the same grid with explicitly selected transport components.

    Excluded operators are replaced by exact zero operators on the same cells.
    This is intended for attribution experiments; it never rescales a retained
    coefficient.
    """

    if not any(
        (
            include_circulation,
            include_vertical_diffusion,
            include_meridional_diffusion
            and transport.meridional_diffusion is not None,
            include_convection and transport.convection is not None,
        )
    ):
        raise ValueError("at least one available transport component is required")
    circulation = transport.circulation
    if not include_circulation:
        circulation = ConservativeCirculationNetwork(
            layers=transport.circulation.layers,
            fluxes=(),
        )
    vertical_diffusion = transport.vertical_diffusion
    if not include_vertical_diffusion:
        vertical_diffusion = ConservativeColumn(
            layers=transport.vertical_diffusion.layers,
            interfaces=(),
        )
    meridional_diffusion = transport.meridional_diffusion
    if not include_meridional_diffusion:
        meridional_diffusion = None
    convection = transport.convection
    if not include_convection:
        convection = None
    selected_reset = (
        np.asarray(transport.reset_mask, dtype=bool).copy()
        if reset_mask is None
        else np.asarray(reset_mask, dtype=bool).copy()
    )
    if selected_reset.shape != transport.reset_mask.shape:
        raise ValueError("selected reset mask must match the transport grid")
    return replace(
        transport,
        circulation=circulation,
        vertical_diffusion=vertical_diffusion,
        meridional_diffusion=meridional_diffusion,
        convection=convection,
        reset_mask=selected_reset,
    )


def select_vertical_diffusion_altitude_domain(
    transport: PassiveAgeTransport,
    *,
    minimum_interface_altitude_km: float | None = None,
    maximum_interface_altitude_km: float | None = None,
) -> PassiveAgeTransport:
    """Retain complete Kzz interfaces inside one altitude interval.

    The function is for mechanism attribution. It does not modify any retained
    exchange coefficient. Interface altitude is the midpoint of its adjacent
    finite-volume cell centers, and the interval is half-open at the upper
    bound so complementary domains partition the original operator exactly.
    """

    lower = (
        -float("inf")
        if minimum_interface_altitude_km is None
        else float(minimum_interface_altitude_km)
    )
    upper = (
        float("inf")
        if maximum_interface_altitude_km is None
        else float(maximum_interface_altitude_km)
    )
    if not lower < upper:
        raise ValueError("vertical-diffusion altitude bounds must increase")

    altitude = np.asarray(transport.altitude_centers_km, dtype=float)
    interface_altitude = 0.5 * (altitude[:-1] + altitude[1:])
    allowed_pairs: set[tuple[str, str]] = set()
    for latitude_index in range(transport.air_moles.shape[0]):
        for vertical_edge, height_km in enumerate(interface_altitude, start=1):
            if lower <= height_km < upper:
                allowed_pairs.add(
                    (
                        latitude_vertical_cell_name(
                            latitude_index,
                            vertical_edge - 1,
                        ),
                        latitude_vertical_cell_name(
                            latitude_index,
                            vertical_edge,
                        ),
                    )
                )

    retained = tuple(
        interface
        for interface in transport.vertical_diffusion.interfaces
        if (interface.first_layer, interface.second_layer) in allowed_pairs
    )
    vertical_diffusion = ConservativeColumn(
        layers=transport.vertical_diffusion.layers,
        interfaces=retained,
    )
    return replace(transport, vertical_diffusion=vertical_diffusion)


def select_vertical_diffusion_below_local_tropopause(
    transport: PassiveAgeTransport,
) -> PassiveAgeTransport:
    """Retain unchanged Kzz interfaces connecting two tropospheric cells.

    Cell classification follows ``transport.reset_mask`` so the selector uses
    the same discretized local tropopause as the tracer boundary condition.
    No retained exchange coefficient is rescaled.
    """

    tropospheric = np.asarray(transport.reset_mask, dtype=bool)
    allowed_pairs: set[tuple[str, str]] = set()
    for latitude_index in range(tropospheric.shape[0]):
        for vertical_edge in range(1, tropospheric.shape[1]):
            if (
                tropospheric[latitude_index, vertical_edge - 1]
                and tropospheric[latitude_index, vertical_edge]
            ):
                allowed_pairs.add(
                    (
                        latitude_vertical_cell_name(
                            latitude_index, vertical_edge - 1
                        ),
                        latitude_vertical_cell_name(latitude_index, vertical_edge),
                    )
                )
    retained = tuple(
        interface
        for interface in transport.vertical_diffusion.interfaces
        if (interface.first_layer, interface.second_layer) in allowed_pairs
    )
    vertical_diffusion = ConservativeColumn(
        layers=transport.vertical_diffusion.layers,
        interfaces=retained,
    )
    return replace(transport, vertical_diffusion=vertical_diffusion)
