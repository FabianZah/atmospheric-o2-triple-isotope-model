"""Decode one ERA5 monthly convection bundle into conservative zonal inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from era5_convection_reference import (
    ECMWF_STANDARD_GRAVITY_M_S2,
    ERA5_CONVECTION_PARAMETERS,
    global_cell_areas_from_centers_m2,
    l137_geopotential_and_thickness,
    l137_half_level_pressure_pa,
    zonal_detrainment_to_mol_per_year,
    zonal_mass_flux_to_mol_per_year,
)
from meridional_transport_reference import (
    STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL,
)
from conservative_column_transport import AtmosphericLayer
from convective_plume_transport import ConvectivePlumeColumn, ConvectivePlumeGrid
from conservative_circulation_transport import latitude_vertical_cell_name
from meridional_transport_reference import EARTH_RADIUS_M, SECONDS_PER_YEAR


@dataclass(frozen=True)
class Era5ZonalConvectionMonth:
    """Bottom-to-top plume inputs integrated over each latitude band."""

    latitude_degrees: np.ndarray
    layer_air_moles: np.ndarray
    updraught_interface_mol_per_year: np.ndarray
    downdraught_interface_mol_per_year: np.ndarray
    updraught_detrainment_mol_per_year: np.ndarray
    downdraught_detrainment_mol_per_year: np.ndarray
    updraught_negative_entrainment_mol_per_year: np.ndarray
    downdraught_negative_entrainment_mol_per_year: np.ndarray
    source_files: tuple[str, str, str]

    @property
    def updraught_negative_fraction(self) -> float:
        negative = float(np.sum(self.updraught_negative_entrainment_mol_per_year))
        positive = np.maximum(
            self.updraught_interface_mol_per_year[1:]
            - self.updraught_interface_mol_per_year[:-1]
            + self.updraught_detrainment_mol_per_year,
            0.0,
        )
        return negative / float(np.sum(positive))

    @property
    def downdraught_negative_fraction(self) -> float:
        negative = float(np.sum(self.downdraught_negative_entrainment_mol_per_year))
        positive = np.maximum(
            self.downdraught_interface_mol_per_year[:-1]
            - self.downdraught_interface_mol_per_year[1:]
            + self.downdraught_detrainment_mol_per_year,
            0.0,
        )
        return negative / float(np.sum(positive))

    @property
    def updraught_detrainment_correction_fraction(self) -> float:
        return float(np.sum(self.updraught_negative_entrainment_mol_per_year)) / float(
            np.sum(self.updraught_detrainment_mol_per_year)
        )

    @property
    def downdraught_detrainment_correction_fraction(self) -> float:
        return float(np.sum(self.downdraught_negative_entrainment_mol_per_year)) / float(
            np.sum(self.downdraught_detrainment_mol_per_year)
        )

    def to_continuity_projected_plume_grid(self) -> ConvectivePlumeGrid:
        """Build a conservative plume grid using the minimum detrainment correction.

        Fixed monthly-mean interface fluxes can imply tiny negative entrainment
        after horizontal interpolation and mean-state thickness conversion. The
        one-sided projection adds exactly that deficit to layer detrainment,
        making diagnosed entrainment zero only where required. No isotope or
        transport scaling factor is introduced.
        """

        corrected_up_detrainment = (
            self.updraught_detrainment_mol_per_year
            + self.updraught_negative_entrainment_mol_per_year
        )
        corrected_down_detrainment = (
            self.downdraught_detrainment_mol_per_year
            + self.downdraught_negative_entrainment_mol_per_year
        )
        columns = []
        for latitude_index, latitude in enumerate(self.latitude_degrees):
            layers = tuple(
                AtmosphericLayer(
                    name=latitude_vertical_cell_name(latitude_index, layer_index),
                    air_moles=float(self.layer_air_moles[layer_index, latitude_index]),
                )
                for layer_index in range(self.layer_air_moles.shape[0])
            )
            try:
                column = ConvectivePlumeColumn(
                    layers=layers,
                    updraught_interface_flux_mol_per_year=(
                        self.updraught_interface_mol_per_year[:, latitude_index]
                    ),
                    downdraught_interface_flux_mol_per_year=(
                        self.downdraught_interface_mol_per_year[:, latitude_index]
                    ),
                    updraught_detrainment_mol_per_year=(
                        corrected_up_detrainment[:, latitude_index]
                    ),
                    downdraught_detrainment_mol_per_year=(
                        corrected_down_detrainment[:, latitude_index]
                    ),
                    source=(
                        "ERA5 monthly mean convective plume fields; minimum "
                        "non-negative-entrainment continuity projection"
                    ),
                )
            except ValueError as exc:
                raise ValueError(
                    "ERA5 convection column failed at latitude index "
                    f"{latitude_index} ({latitude:g} degrees): {exc}"
                ) from exc
            columns.append(column)
        return ConvectivePlumeGrid(tuple(columns))


def mean_era5_zonal_convection_months(
    months: tuple[Era5ZonalConvectionMonth, ...],
    *,
    weights: np.ndarray | None = None,
) -> Era5ZonalConvectionMonth:
    """Average compatible monthly fields and rediagnose continuity deficits."""

    if not months:
        raise ValueError("at least one ERA5 convection month is required")
    reference = months[0]
    for month in months[1:]:
        if not np.array_equal(month.latitude_degrees, reference.latitude_degrees):
            raise ValueError("ERA5 convection months use different latitude grids")
        if not np.allclose(
            month.layer_air_moles,
            reference.layer_air_moles,
            rtol=1.0e-13,
            atol=0.0,
        ):
            raise ValueError("ERA5 convection months use different target inventories")
    if weights is None:
        normalized = np.full(len(months), 1.0 / len(months))
    else:
        normalized = np.asarray(weights, dtype=float)
        if normalized.shape != (len(months),) or np.any(normalized < 0.0):
            raise ValueError("climatology weights must be non-negative and match months")
        if not np.isfinite(np.sum(normalized)) or np.sum(normalized) <= 0.0:
            raise ValueError("climatology weights must have a positive finite sum")
        normalized = normalized / np.sum(normalized)

    def weighted(attribute: str) -> np.ndarray:
        arrays = np.stack(
            [np.asarray(getattr(month, attribute), dtype=float) for month in months]
        )
        return np.tensordot(normalized, arrays, axes=(0, 0))

    up_interface = weighted("updraught_interface_mol_per_year")
    down_interface = weighted("downdraught_interface_mol_per_year")
    up_det = weighted("updraught_detrainment_mol_per_year")
    down_det = weighted("downdraught_detrainment_mol_per_year")
    up_entrainment = up_interface[1:] - up_interface[:-1] + up_det
    down_entrainment = down_interface[:-1] - down_interface[1:] + down_det
    return Era5ZonalConvectionMonth(
        latitude_degrees=reference.latitude_degrees.copy(),
        layer_air_moles=reference.layer_air_moles.copy(),
        updraught_interface_mol_per_year=up_interface,
        downdraught_interface_mol_per_year=down_interface,
        updraught_detrainment_mol_per_year=up_det,
        downdraught_detrainment_mol_per_year=down_det,
        updraught_negative_entrainment_mol_per_year=np.maximum(-up_entrainment, 0.0),
        downdraught_negative_entrainment_mol_per_year=np.maximum(
            -down_entrainment, 0.0
        ),
        source_files=tuple(
            source for month in months for source in month.source_files
        ),
    )


@dataclass(frozen=True)
class TwoReservoirConvectiveExchange:
    """Exact uniform-composition lumping of a conservative plume grid."""

    group_air_moles: np.ndarray
    inventory_matrix_per_year: np.ndarray
    bidirectional_air_flux_mol_per_year: float

    @property
    def first_group_turnover_years(self) -> float:
        return float(self.group_air_moles[0] / self.bidirectional_air_flux_mol_per_year)

    @property
    def second_group_turnover_years(self) -> float:
        return float(self.group_air_moles[1] / self.bidirectional_air_flux_mol_per_year)


def aggregate_plume_grid_to_two_reservoirs(
    plume_grid: ConvectivePlumeGrid,
    first_group_mask: np.ndarray,
) -> TwoReservoirConvectiveExchange:
    """Lump cell transport while preserving group-uniform mixing ratios."""

    air = np.asarray(plume_grid.air_moles, dtype=float)
    first = np.asarray(first_group_mask, dtype=bool).ravel()
    if first.shape != air.shape or not np.any(first) or np.all(first):
        raise ValueError("two-reservoir mask must split the complete plume grid")
    group = np.where(first, 0, 1)
    totals = np.asarray([np.sum(air[group == index]) for index in (0, 1)])
    aggregation = np.vstack([group == index for index in (0, 1)]).astype(float)
    prolongation = np.column_stack(
        [
            np.where(group == index, air / totals[index], 0.0)
            for index in (0, 1)
        ]
    )
    matrix = (
        aggregation
        @ plume_grid.transport_matrix_per_year()
        @ prolongation
    )
    forward_flux = float(matrix[1, 0] * totals[0])
    reverse_flux = float(matrix[0, 1] * totals[1])
    scale = max(abs(forward_flux), abs(reverse_flux), 1.0)
    if (
        np.any(np.diag(matrix) > 1.0e-12)
        or np.any(matrix[[1, 0], [0, 1]] < -1.0e-12)
        or np.max(np.abs(np.sum(matrix, axis=0))) > 1.0e-12
        or abs(forward_flux - reverse_flux) > 1.0e-12 * scale
    ):
        raise ValueError("two-reservoir plume reduction failed conservation checks")
    return TwoReservoirConvectiveExchange(
        group_air_moles=totals,
        inventory_matrix_per_year=matrix,
        bidirectional_air_flux_mol_per_year=0.5 * (forward_flux + reverse_flux),
    )


def _require_variable(dataset, name: str, parameter_id: int, units: str) -> np.ndarray:
    if name not in dataset:
        raise ValueError(f"ERA5 file is missing {name}")
    variable = dataset[name]
    if int(variable.attrs.get("GRIB_paramId", -1)) != parameter_id:
        raise ValueError(f"ERA5 {name} parameter ID does not match {parameter_id}")
    if variable.attrs.get("units") != units:
        raise ValueError(f"ERA5 {name} units do not match {units}")
    expected_dimensions = ("valid_time", "model_level", "latitude", "longitude")
    if variable.dims != expected_dimensions or variable.shape[0] != 1:
        raise ValueError(f"ERA5 {name} must contain one monthly model-level field")
    values = np.asarray(variable.values[0], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"ERA5 {name} contains non-finite values")
    return values


def load_era5_zonal_convection_month(
    convection_path: Path,
    state_path: Path,
    surface_state_path: Path,
) -> Era5ZonalConvectionMonth:
    """Load, validate, and conservatively integrate one monthly ERA5 bundle."""

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("ERA5 NetCDF decoding requires xarray and h5netcdf") from exc

    convection_path = Path(convection_path)
    state_path = Path(state_path)
    surface_state_path = Path(surface_state_path)
    with (
        xr.open_dataset(convection_path, engine="h5netcdf") as convection,
        xr.open_dataset(state_path, engine="h5netcdf") as state,
        xr.open_dataset(surface_state_path, engine="h5netcdf") as surface,
    ):
        level = np.asarray(convection.model_level.values, dtype=float)
        if not np.array_equal(level, np.arange(1.0, 138.0)):
            raise ValueError("ERA5 convection levels must run from 1 to 137")
        latitude = np.asarray(convection.latitude.values, dtype=float)
        longitude = np.asarray(convection.longitude.values, dtype=float)
        for dataset in (state, surface):
            if not np.array_equal(dataset.latitude.values, latitude) or not np.array_equal(
                dataset.longitude.values, longitude
            ):
                raise ValueError("ERA5 bundle grids do not match")

        updraught = _require_variable(
            convection, "avg_umf", ERA5_CONVECTION_PARAMETERS["mumf"], "kg m**-2 s**-1"
        )
        downdraught_signed = _require_variable(
            convection, "avg_dmf", ERA5_CONVECTION_PARAMETERS["mdmf"], "kg m**-2 s**-1"
        )
        up_detrainment = _require_variable(
            convection, "avg_udr", ERA5_CONVECTION_PARAMETERS["mudr"], "kg m**-3 s**-1"
        )
        down_detrainment = _require_variable(
            convection, "avg_ddr", ERA5_CONVECTION_PARAMETERS["mddr"], "kg m**-3 s**-1"
        )
        temperature = _require_variable(state, "t", 130, "K")
        humidity = _require_variable(state, "q", 133, "kg kg**-1")
        surface_geopotential = _require_variable(surface, "z", 129, "m**2 s**-2")[0]
        log_surface_pressure = _require_variable(surface, "lnsp", 152, "Numeric")[0]

        if np.any(updraught < 0.0) or np.any(up_detrainment < 0.0) or np.any(down_detrainment < 0.0):
            raise ValueError("ERA5 updraught and detrainment fields must be non-negative")
        # ERA5 downdraught flux is archived negative. Tiny opposite-sign values
        # arise at interpolation precision and carry no resolved downward mass.
        downdraught = np.maximum(-downdraught_signed, 0.0)
        surface_pressure = np.exp(log_surface_pressure)
        _, _, thickness = l137_geopotential_and_thickness(
            temperature,
            humidity,
            surface_pressure,
            surface_geopotential,
        )
        area = global_cell_areas_from_centers_m2(latitude, longitude)

        up_zonal = zonal_mass_flux_to_mol_per_year(updraught, area)
        down_zonal = zonal_mass_flux_to_mol_per_year(downdraught, area)
        up_det_zonal = zonal_detrainment_to_mol_per_year(up_detrainment, thickness, area)
        down_det_zonal = zonal_detrainment_to_mol_per_year(down_detrainment, thickness, area)

        pressure_thickness = np.diff(
            l137_half_level_pressure_pa(surface_pressure),
            axis=0,
        )
        total_air_mass = pressure_thickness * area[None, :, :] / ECMWF_STANDARD_GRAVITY_M_S2
        dry_air_mass = total_air_mass * (1.0 - humidity)
        layer_air_moles = np.sum(dry_air_mass, axis=-1) / STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL

    # Archive levels 1..137 represent half-levels 1..137; half-level 0 is zero.
    # Reverse both dimensions to the plume operator's south-to-north,
    # bottom-to-top convention.
    up_interface = np.concatenate((np.zeros((1, len(latitude))), up_zonal), axis=0)[::-1, ::-1]
    down_interface = np.concatenate((np.zeros((1, len(latitude))), down_zonal), axis=0)[::-1, ::-1]
    up_det = up_det_zonal[::-1, ::-1]
    down_det = down_det_zonal[::-1, ::-1]
    air_moles = layer_air_moles[::-1, ::-1]

    # Return unresolved downdraught at the lower boundary to the lowest layer.
    down_det[0] += down_interface[0]
    down_interface[0] = 0.0
    up_entrainment = up_interface[1:] - up_interface[:-1] + up_det
    down_entrainment = down_interface[:-1] - down_interface[1:] + down_det
    return Era5ZonalConvectionMonth(
        latitude_degrees=latitude[::-1].copy(),
        layer_air_moles=air_moles,
        updraught_interface_mol_per_year=up_interface,
        downdraught_interface_mol_per_year=down_interface,
        updraught_detrainment_mol_per_year=up_det,
        downdraught_detrainment_mol_per_year=down_det,
        updraught_negative_entrainment_mol_per_year=np.maximum(-up_entrainment, 0.0),
        downdraught_negative_entrainment_mol_per_year=np.maximum(-down_entrainment, 0.0),
        source_files=(str(convection_path), str(state_path), str(surface_state_path)),
    )


def _target_source_cell_areas_m2(
    source_latitude_centers_degrees: np.ndarray,
    source_longitude_centers_degrees: np.ndarray,
    target_latitude_edges_degrees: np.ndarray,
) -> np.ndarray:
    """Area overlap for target latitude bands and source grid cells."""

    source_latitude = np.asarray(source_latitude_centers_degrees, dtype=float)
    source_longitude = np.asarray(source_longitude_centers_degrees, dtype=float)
    target_edges = np.asarray(target_latitude_edges_degrees, dtype=float)
    if not np.all(np.diff(source_latitude) < 0.0):
        raise ValueError("ERA5 source latitude must be north-to-south")
    if not np.all(np.diff(target_edges) > 0.0):
        raise ValueError("target latitude edges must increase south-to-north")
    if target_edges[0] != -90.0 or target_edges[-1] != 90.0:
        raise ValueError("target convection grid must be global")
    longitude_spacing = np.diff(source_longitude)
    if not np.allclose(longitude_spacing, longitude_spacing[0]):
        raise ValueError("ERA5 convection longitude must be regular")
    longitude_width_rad = np.deg2rad(longitude_spacing[0])

    source_ascending = source_latitude[::-1]
    source_edges_ascending = np.concatenate(
        (
            [-90.0],
            0.5 * (source_ascending[:-1] + source_ascending[1:]),
            [90.0],
        )
    )
    source_edges = source_edges_ascending[::-1]
    area = np.zeros(
        (len(target_edges) - 1, len(source_latitude), len(source_longitude)),
        dtype=float,
    )
    for target_index, (target_south, target_north) in enumerate(
        zip(target_edges[:-1], target_edges[1:], strict=True)
    ):
        for source_index in range(len(source_latitude)):
            source_north = source_edges[source_index]
            source_south = source_edges[source_index + 1]
            overlap_south = max(target_south, source_south)
            overlap_north = min(target_north, source_north)
            if overlap_north <= overlap_south:
                continue
            cell_area = (
                EARTH_RADIUS_M**2
                * longitude_width_rad
                * (
                    np.sin(np.deg2rad(overlap_north))
                    - np.sin(np.deg2rad(overlap_south))
                )
            )
            area[target_index, source_index, :] = cell_area
    return area


def _interpolate_interface_field_at_pressure(
    pressure_pa: float,
    half_level_pressure_pa: np.ndarray,
    half_level_field: np.ndarray,
) -> np.ndarray:
    """Log-pressure interpolation of one half-level field at every grid cell."""

    pressure = float(pressure_pa)
    result = np.zeros(half_level_field.shape[1:], dtype=float)
    for latitude_index in range(result.shape[0]):
        for longitude_index in range(result.shape[1]):
            native_pressure = half_level_pressure_pa[1:, latitude_index, longitude_index]
            if pressure < native_pressure[0] or pressure > native_pressure[-1]:
                continue
            result[latitude_index, longitude_index] = np.interp(
                np.log(pressure),
                np.log(native_pressure),
                half_level_field[1:, latitude_index, longitude_index],
            )
    return result


def load_era5_convection_on_reduced_grid(
    convection_path: Path,
    state_path: Path,
    surface_state_path: Path,
    *,
    target_latitude_edges_degrees: np.ndarray,
    target_pressure_edges_pa: np.ndarray,
    target_air_moles: np.ndarray,
) -> Era5ZonalConvectionMonth:
    """Conservatively reduce native ERA5 plumes onto a latitude-pressure grid."""

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("ERA5 NetCDF decoding requires xarray and h5netcdf") from exc

    target_latitude_edges = np.asarray(target_latitude_edges_degrees, dtype=float)
    target_pressure_edges = np.asarray(target_pressure_edges_pa, dtype=float)
    target_air = np.asarray(target_air_moles, dtype=float)
    expected_shape = (len(target_latitude_edges) - 1, len(target_pressure_edges) - 1)
    if target_air.shape != expected_shape or np.any(target_air <= 0.0):
        raise ValueError("target air inventory must match the latitude-pressure cells")
    if not np.all(np.diff(target_pressure_edges) < 0.0):
        raise ValueError("target pressure edges must decrease bottom-to-top")

    convection_path = Path(convection_path)
    state_path = Path(state_path)
    surface_state_path = Path(surface_state_path)
    with (
        xr.open_dataset(convection_path, engine="h5netcdf") as convection,
        xr.open_dataset(state_path, engine="h5netcdf") as state,
        xr.open_dataset(surface_state_path, engine="h5netcdf") as surface,
    ):
        latitude = np.asarray(convection.latitude.values, dtype=float)
        longitude = np.asarray(convection.longitude.values, dtype=float)
        updraught = _require_variable(
            convection, "avg_umf", ERA5_CONVECTION_PARAMETERS["mumf"], "kg m**-2 s**-1"
        )
        down_signed = _require_variable(
            convection, "avg_dmf", ERA5_CONVECTION_PARAMETERS["mdmf"], "kg m**-2 s**-1"
        )
        up_detrainment = _require_variable(
            convection, "avg_udr", ERA5_CONVECTION_PARAMETERS["mudr"], "kg m**-3 s**-1"
        )
        down_detrainment = _require_variable(
            convection, "avg_ddr", ERA5_CONVECTION_PARAMETERS["mddr"], "kg m**-3 s**-1"
        )
        temperature = _require_variable(state, "t", 130, "K")
        humidity = _require_variable(state, "q", 133, "kg kg**-1")
        surface_geopotential = _require_variable(surface, "z", 129, "m**2 s**-2")[0]
        surface_pressure = np.exp(_require_variable(surface, "lnsp", 152, "Numeric")[0])
        half_pressure = l137_half_level_pressure_pa(surface_pressure)
        _, _, thickness = l137_geopotential_and_thickness(
            temperature, humidity, surface_pressure, surface_geopotential
        )
        overlap_area = _target_source_cell_areas_m2(
            latitude, longitude, target_latitude_edges
        )

        native_up_interface = np.concatenate(
            (np.zeros((1, *updraught.shape[1:])), updraught), axis=0
        )
        native_down_interface = np.concatenate(
            (np.zeros((1, *down_signed.shape[1:])), np.maximum(-down_signed, 0.0)),
            axis=0,
        )
        target_count = expected_shape[0]
        layer_count = expected_shape[1]
        up_interface_kg_s = np.zeros((layer_count + 1, target_count), dtype=float)
        down_interface_kg_s = np.zeros_like(up_interface_kg_s)
        up_det_kg_s = np.zeros((layer_count, target_count), dtype=float)
        down_det_kg_s = np.zeros_like(up_det_kg_s)

        # The bottom value is the diagnosed plume launch/return at local surface.
        for target_index in range(target_count):
            area = overlap_area[target_index]
            up_interface_kg_s[0, target_index] = np.sum(native_up_interface[-1] * area)
            surface_down = float(np.sum(native_down_interface[-1] * area))
            down_det_kg_s[0, target_index] += surface_down
        for interface_index, pressure in enumerate(target_pressure_edges[1:-1], start=1):
            up_at_pressure = _interpolate_interface_field_at_pressure(
                pressure, half_pressure, native_up_interface
            )
            down_at_pressure = _interpolate_interface_field_at_pressure(
                pressure, half_pressure, native_down_interface
            )
            up_interface_kg_s[interface_index] = np.sum(
                overlap_area * up_at_pressure[None, :, :], axis=(1, 2)
            )
            down_interface_kg_s[interface_index] = np.sum(
                overlap_area * down_at_pressure[None, :, :], axis=(1, 2)
            )

        native_top = np.maximum(half_pressure[:-1], 0.1)
        native_bottom = half_pressure[1:]
        native_log_thickness = np.log(native_bottom / native_top)
        up_detrainment_per_area = up_detrainment * thickness
        down_detrainment_per_area = down_detrainment * thickness
        for layer_index in range(layer_count):
            target_bottom = target_pressure_edges[layer_index]
            target_top = target_pressure_edges[layer_index + 1]
            overlap_top = np.maximum(native_top, target_top)
            overlap_bottom = np.minimum(native_bottom, target_bottom)
            fraction = np.where(
                overlap_bottom > overlap_top,
                np.log(np.maximum(overlap_bottom, overlap_top) / overlap_top)
                / native_log_thickness,
                0.0,
            )
            for target_index in range(target_count):
                area = overlap_area[target_index]
                up_det_kg_s[layer_index, target_index] += np.sum(
                    up_detrainment_per_area * fraction * area[None, :, :]
                )
                down_det_kg_s[layer_index, target_index] += np.sum(
                    down_detrainment_per_area * fraction * area[None, :, :]
                )

    conversion = SECONDS_PER_YEAR / STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL
    up_interface = up_interface_kg_s * conversion
    down_interface = down_interface_kg_s * conversion
    up_det = up_det_kg_s * conversion
    down_det = down_det_kg_s * conversion
    up_entrainment = up_interface[1:] - up_interface[:-1] + up_det
    down_entrainment = down_interface[:-1] - down_interface[1:] + down_det
    target_latitude = 0.5 * (target_latitude_edges[:-1] + target_latitude_edges[1:])
    return Era5ZonalConvectionMonth(
        latitude_degrees=target_latitude,
        layer_air_moles=target_air.T.copy(),
        updraught_interface_mol_per_year=up_interface,
        downdraught_interface_mol_per_year=down_interface,
        updraught_detrainment_mol_per_year=up_det,
        downdraught_detrainment_mol_per_year=down_det,
        updraught_negative_entrainment_mol_per_year=np.maximum(-up_entrainment, 0.0),
        downdraught_negative_entrainment_mol_per_year=np.maximum(-down_entrainment, 0.0),
        source_files=(str(convection_path), str(state_path), str(surface_state_path)),
    )
