"""ERA5-complete retrieval metadata and conservative unit conversions.

The module has no dependency on ``cdsapi`` or a GRIB decoder. It defines the
source-backed MARS requests and converts already decoded regular-grid fields
to latitude-band air transfers. Native file decoding remains an adapter step
because the archived half-level orientation and downdraught sign must be
verified from real GRIB metadata before a plume operator is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from meridional_transport_reference import (
    EARTH_RADIUS_M,
    SECONDS_PER_YEAR,
    STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL,
)


ERA5_COMPLETE_DATASET = "reanalysis-era5-complete"
ERA5_CONVECTION_PARAMETERS = {
    "mumf": 235009,
    "mdmf": 235010,
    "mudr": 235011,
    "mddr": 235012,
}
ERA5_STATE_PARAMETERS = {
    "temperature": 130,
    "specific_humidity": 133,
}
ERA5_SURFACE_STATE_PARAMETERS = {
    "surface_geopotential": 129,
    "log_surface_pressure": 152,
}
ERA5_CONVECTION_DOCUMENTATION = (
    "https://confluence.ecmwf.int/spaces/CKB/pages/76414402/"
    "ERA5+data+documentation"
)
ERA5_DOWNLOAD_DOCUMENTATION = (
    "https://confluence.ecmwf.int/spaces/CKB/pages/129135000/"
    "How+to+download+ERA5"
)
ERA5_L137_DOCUMENTATION = (
    "https://confluence.ecmwf.int/spaces/UDOC/pages/108117123/"
    "L137+model+level+definitions"
)
ERA5_GEOPOTENTIAL_DOCUMENTATION = (
    "https://confluence.ecmwf.int/pages/viewpage.action?pageId=174862416"
)
ERA5_L137_COEFFICIENTS_FILE = (
    Path(__file__).with_name("data") / "era5_l137_hybrid_coefficients.csv"
)
ECMWF_DRY_AIR_GAS_CONSTANT_J_KG_K = 287.06
ECMWF_STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class Era5RetrievalJob:
    """One tape-efficient monthly ERA5-complete request."""

    name: str
    target: Path
    request: dict[str, str]


def month_starts(start_year: int, end_year: int) -> tuple[date, ...]:
    if not 1940 <= start_year <= end_year:
        raise ValueError("ERA5 year range must increase from 1940 or later")
    return tuple(
        date(year, month, 1)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    )


def forecast_monthly_mean_valid_time(month: date) -> date:
    """Return ERA5 ``moda`` forecast valid time for an averaging month."""

    if month.day != 1:
        raise ValueError("ERA5 monthly means use the first day of each month")
    if month.month == 12:
        return date(month.year + 1, 1, 1)
    return date(month.year, month.month + 1, 1)


def monthly_retrieval_jobs(
    month: date,
    output_directory: Path,
    *,
    grid_degrees: float = 10.0,
    output_format: str = "netcdf",
) -> tuple[Era5RetrievalJob, ...]:
    """Return convection and state requests for one MARS tape month.

    Monthly requests are kept separate because ECMWF recommends retrieving all
    required data from one tape before moving to the next. Forecast mean-rate,
    analysis model-level state, and level-1 surface state use different MARS
    product selections and therefore remain separate jobs.
    """

    if month.day != 1:
        raise ValueError("ERA5 monthly means use the first day of each month")
    if grid_degrees <= 0.0 or 180.0 % grid_degrees != 0.0 or 360.0 % grid_degrees != 0.0:
        raise ValueError("regular ERA5 grid must divide both 180 and 360 degrees")
    if output_format not in {"netcdf", "grib"}:
        raise ValueError("ERA5 retrieval format must be netcdf or grib")

    output = Path(output_directory)
    date_text = month.isoformat()
    stamp = month.strftime("%Y%m")
    common = {
        "class": "ea",
        "expver": "1",
        "stream": "moda",
        "date": date_text,
        "grid": f"{grid_degrees:g}/{grid_degrees:g}",
        "format": output_format,
    }
    suffix = "nc" if output_format == "netcdf" else "grib"
    convection = {
        **common,
        "type": "fc",
        "levtype": "ml",
        "levelist": "1/to/137",
        "param": "/".join(str(value) for value in ERA5_CONVECTION_PARAMETERS.values()),
    }
    state = {
        **common,
        "type": "an",
        "levtype": "ml",
        "levelist": "1/to/137",
        "param": "/".join(str(value) for value in ERA5_STATE_PARAMETERS.values()),
    }
    surface = {
        **common,
        "type": "an",
        "levtype": "ml",
        "levelist": "1",
        "param": "/".join(
            str(value) for value in ERA5_SURFACE_STATE_PARAMETERS.values()
        ),
    }
    return (
        Era5RetrievalJob(
            name=f"convection_{stamp}",
            target=output / f"era5_convection_{stamp}.{suffix}",
            request=convection,
        ),
        Era5RetrievalJob(
            name=f"state_{stamp}",
            target=output / f"era5_state_{stamp}.{suffix}",
            request=state,
        ),
        Era5RetrievalJob(
            name=f"surface_state_{stamp}",
            target=output / f"era5_surface_state_{stamp}.{suffix}",
            request=surface,
        ),
    )


def calendar_month_retrieval_jobs(
    calendar_month: int,
    start_year: int,
    end_year: int,
    output_directory: Path,
    *,
    grid_degrees: float = 10.0,
    output_format: str = "netcdf",
) -> tuple[Era5RetrievalJob, ...]:
    """Batch one calendar month across years into three MARS requests.

    The product selections are identical to :func:`monthly_retrieval_jobs`.
    Batching reduces client submissions while preserving calendar-month
    timestamps for deterministic splitting into the validated monthly format.
    """

    if not 1 <= calendar_month <= 12:
        raise ValueError("calendar month must lie between 1 and 12")
    months = tuple(
        date(year, calendar_month, 1)
        for year in range(start_year, end_year + 1)
    )
    if not months or start_year < 1940 or end_year < start_year:
        raise ValueError("ERA5 year range must increase from 1940 or later")
    template = monthly_retrieval_jobs(
        months[0],
        output_directory,
        grid_degrees=grid_degrees,
        output_format=output_format,
    )
    date_selection = "/".join(month.isoformat() for month in months)
    suffix = "nc" if output_format == "netcdf" else "grib"
    output = Path(output_directory)
    jobs = []
    for job in template:
        product = job.name.rsplit("_", 1)[0]
        request = {**job.request, "date": date_selection}
        jobs.append(
            Era5RetrievalJob(
                name=(
                    f"{product}_climatology_month{calendar_month:02d}_"
                    f"{start_year}_{end_year}"
                ),
                target=(
                    output
                    / (
                        f"era5_{product}_month{calendar_month:02d}_"
                        f"{start_year}_{end_year}.{suffix}"
                    )
                ),
                request=request,
            )
        )
    return tuple(jobs)


def regular_latitude_longitude_edges(
    grid_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_degrees <= 0.0 or 180.0 % grid_degrees != 0.0 or 360.0 % grid_degrees != 0.0:
        raise ValueError("regular grid spacing must divide 180 and 360 degrees")
    latitude = np.arange(-90.0, 90.0 + grid_degrees, grid_degrees, dtype=float)
    longitude = np.arange(0.0, 360.0 + grid_degrees, grid_degrees, dtype=float)
    return latitude, longitude


def spherical_cell_areas_m2(
    latitude_edges_degrees: np.ndarray,
    longitude_edges_degrees: np.ndarray,
) -> np.ndarray:
    """Exact spherical area for each regular or irregular lat-lon cell."""

    latitude = np.asarray(latitude_edges_degrees, dtype=float)
    longitude = np.asarray(longitude_edges_degrees, dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude edges must be one-dimensional")
    if len(latitude) < 2 or len(longitude) < 2:
        raise ValueError("horizontal grid requires at least one cell")
    if not np.all(np.diff(latitude) > 0.0) or not np.all(np.diff(longitude) > 0.0):
        raise ValueError("horizontal coordinate edges must strictly increase")
    if latitude[0] < -90.0 or latitude[-1] > 90.0:
        raise ValueError("latitude edges must lie between -90 and 90 degrees")
    latitude_factor = np.diff(np.sin(np.deg2rad(latitude)))
    longitude_width = np.diff(np.deg2rad(longitude))
    return EARTH_RADIUS_M**2 * latitude_factor[:, None] * longitude_width[None, :]


def global_cell_areas_from_centers_m2(
    latitude_centers_degrees: np.ndarray,
    longitude_centers_degrees: np.ndarray,
) -> np.ndarray:
    """Return global cell areas in the same ordering as grid-point centres."""

    latitude = np.asarray(latitude_centers_degrees, dtype=float)
    longitude = np.asarray(longitude_centers_degrees, dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude centres must be one-dimensional")
    if len(latitude) < 2 or len(longitude) < 2:
        raise ValueError("global centre grid requires at least two points per axis")
    latitude_difference = np.diff(latitude)
    descending_latitude = np.all(latitude_difference < 0.0)
    if not descending_latitude and not np.all(latitude_difference > 0.0):
        raise ValueError("latitude centres must be strictly monotonic")
    latitude_ascending = latitude[::-1] if descending_latitude else latitude
    if not np.allclose(np.diff(latitude_ascending), np.diff(latitude_ascending)[0]):
        raise ValueError("ERA5 adapter currently requires regular latitude spacing")
    if latitude_ascending[0] != -90.0 or latitude_ascending[-1] != 90.0:
        raise ValueError("ERA5 convection centre grid must include both poles")
    longitude_spacing = np.diff(longitude)
    if (
        not np.all(longitude_spacing > 0.0)
        or not np.allclose(longitude_spacing, longitude_spacing[0])
        or not np.isclose(longitude[-1] - longitude[0] + longitude_spacing[0], 360.0)
    ):
        raise ValueError("ERA5 longitude centres must form one regular global cycle")

    latitude_edges = np.concatenate(
        (
            [-90.0],
            0.5 * (latitude_ascending[:-1] + latitude_ascending[1:]),
            [90.0],
        )
    )
    longitude_edges = np.concatenate(
        (
            [longitude[0] - 0.5 * longitude_spacing[0]],
            0.5 * (longitude[:-1] + longitude[1:]),
            [longitude[-1] + 0.5 * longitude_spacing[-1]],
        )
    )
    area = spherical_cell_areas_m2(latitude_edges, longitude_edges)
    return area[::-1] if descending_latitude else area


def zonal_mass_flux_to_mol_per_year(
    mass_flux_kg_m2_s: np.ndarray,
    cell_area_m2: np.ndarray,
) -> np.ndarray:
    """Integrate interface mass-flux density over longitude by latitude band."""

    flux = np.asarray(mass_flux_kg_m2_s, dtype=float)
    area = np.asarray(cell_area_m2, dtype=float)
    if flux.ndim < 2 or flux.shape[-2:] != area.shape:
        raise ValueError("mass flux must end in the latitude-longitude grid shape")
    if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(area)):
        raise ValueError("mass flux and cell area must be finite")
    if np.any(area <= 0.0):
        raise ValueError("horizontal cell areas must be positive")
    kg_per_second = np.sum(flux * area, axis=-1)
    return (
        kg_per_second
        * SECONDS_PER_YEAR
        / STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL
    )


def zonal_detrainment_to_mol_per_year(
    detrainment_kg_m3_s: np.ndarray,
    layer_thickness_m: np.ndarray,
    cell_area_m2: np.ndarray,
) -> np.ndarray:
    """Integrate volumetric detrainment over each layer and longitude."""

    detrainment = np.asarray(detrainment_kg_m3_s, dtype=float)
    thickness = np.asarray(layer_thickness_m, dtype=float)
    area = np.asarray(cell_area_m2, dtype=float)
    if detrainment.shape != thickness.shape:
        raise ValueError("detrainment and layer thickness must share a shape")
    if detrainment.ndim < 2 or detrainment.shape[-2:] != area.shape:
        raise ValueError("detrainment fields must end in the horizontal grid shape")
    if not np.all(np.isfinite(detrainment)) or not np.all(np.isfinite(thickness)):
        raise ValueError("detrainment and layer thickness must be finite")
    if np.any(detrainment < 0.0) or np.any(thickness <= 0.0):
        raise ValueError("detrainment must be non-negative and layer thickness positive")
    kg_per_second = np.sum(
        detrainment * thickness * area,
        axis=-1,
    )
    return (
        kg_per_second
        * SECONDS_PER_YEAR
        / STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL
    )


def l137_hybrid_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Return official ECMWF L137 half-level A [Pa] and B coefficients."""

    table = np.loadtxt(
        ERA5_L137_COEFFICIENTS_FILE,
        delimiter=",",
        skiprows=1,
    )
    if table.shape != (138, 3):
        raise ValueError("ECMWF L137 coefficient table must contain half-levels 0-137")
    expected_levels = np.arange(138, dtype=float)
    if not np.array_equal(table[:, 0], expected_levels):
        raise ValueError("ECMWF L137 half-level indices are incomplete or out of order")
    a_pa = table[:, 1]
    b = table[:, 2]
    if (
        not np.all(np.isfinite(a_pa))
        or not np.all(np.isfinite(b))
        or np.any(a_pa < 0.0)
        or np.any((b < 0.0) | (b > 1.0))
        or a_pa[0] != 0.0
        or b[0] != 0.0
        or a_pa[-1] != 0.0
        or b[-1] != 1.0
    ):
        raise ValueError("ECMWF L137 coefficients failed physical endpoint checks")
    return a_pa.copy(), b.copy()


def l137_half_level_pressure_pa(surface_pressure_pa: np.ndarray) -> np.ndarray:
    """Compute all 138 L137 half-level pressures from surface pressure."""

    surface_pressure = np.asarray(surface_pressure_pa, dtype=float)
    if not np.all(np.isfinite(surface_pressure)) or np.any(surface_pressure <= 0.0):
        raise ValueError("surface pressure must be finite and positive")
    a_pa, b = l137_hybrid_coefficients()
    expand = (slice(None),) + (None,) * surface_pressure.ndim
    pressure = a_pa[expand] + b[expand] * surface_pressure[None, ...]
    if np.any(np.diff(pressure, axis=0) <= 0.0):
        raise ValueError("L137 half-level pressure must increase monotonically downward")
    return pressure


def l137_geopotential_and_thickness(
    temperature_k: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    surface_pressure_pa: np.ndarray,
    surface_geopotential_m2_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce ECMWF's hydrostatic L137 geopotential integration.

    Inputs use top-to-bottom model-level order. Returned arrays are full-level
    geopotential, half-level geopotential, and positive geometric layer
    thickness in metres.
    """

    temperature = np.asarray(temperature_k, dtype=float)
    humidity = np.asarray(specific_humidity_kg_kg, dtype=float)
    surface_pressure = np.asarray(surface_pressure_pa, dtype=float)
    surface_geopotential = np.asarray(surface_geopotential_m2_s2, dtype=float)
    if temperature.shape != humidity.shape or temperature.shape[0] != 137:
        raise ValueError("temperature and humidity must share 137 top-to-bottom levels")
    if temperature.shape[1:] != surface_pressure.shape:
        raise ValueError("surface pressure must match the state-field horizontal shape")
    if surface_geopotential.shape != surface_pressure.shape:
        raise ValueError("surface geopotential must match surface pressure")
    if (
        not np.all(np.isfinite(temperature))
        or not np.all(np.isfinite(humidity))
        or not np.all(np.isfinite(surface_geopotential))
        or np.any(temperature <= 0.0)
        or np.any(humidity < 0.0)
    ):
        raise ValueError("ERA5 temperature, humidity, and geopotential must be physical")

    half_pressure = l137_half_level_pressure_pa(surface_pressure)
    full_geopotential = np.empty_like(temperature)
    half_geopotential = np.empty((138, *surface_pressure.shape), dtype=float)
    half_geopotential[-1] = surface_geopotential
    virtual_temperature = temperature * (1.0 + 0.609133 * humidity)

    for level_index in range(136, -1, -1):
        pressure_top = half_pressure[level_index]
        pressure_bottom = half_pressure[level_index + 1]
        if level_index == 0:
            dlog_pressure = np.log(pressure_bottom / 0.1)
            alpha = np.log(2.0)
        else:
            dlog_pressure = np.log(pressure_bottom / pressure_top)
            alpha = 1.0 - (
                pressure_top
                / (pressure_bottom - pressure_top)
                * dlog_pressure
            )
        rd_tv = ECMWF_DRY_AIR_GAS_CONSTANT_J_KG_K * virtual_temperature[level_index]
        full_geopotential[level_index] = (
            half_geopotential[level_index + 1] + rd_tv * alpha
        )
        half_geopotential[level_index] = (
            half_geopotential[level_index + 1] + rd_tv * dlog_pressure
        )

    thickness_m = (
        half_geopotential[:-1] - half_geopotential[1:]
    ) / ECMWF_STANDARD_GRAVITY_M_S2
    if np.any(thickness_m <= 0.0) or not np.all(np.isfinite(thickness_m)):
        raise ValueError("ECMWF hydrostatic integration produced invalid layer thickness")
    return full_geopotential, half_geopotential, thickness_m
