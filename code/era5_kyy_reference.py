"""Source-backed ERA5 meridional eddy-diffusion diagnostic.

The numerical Kyy field used by Morgan et al. (2004) is not archived with the
paper. This module therefore constructs a separately identified modern
transport candidate from the 2010-2019 ERA5 diagnostics published by Serva
(2022):

* zonal-mean eastward wind,
* zonal-mean temperature, and
* eastward-wind tendency due to Eliassen-Palm flux divergence.

The zonal-mean quasigeostrophic potential-vorticity gradient follows Randel
and Garcia (1994), Equation 2, including their one-pass 1-2-1 meridional
smoothing and printed minimum gradient of 0.5e-11 m-1 s-1. Their Equation 14
then gives the flux-gradient diffusivity. Negative diagnosed diffusivities are
set to zero following Jiang et al. (2004). No upper cap or age-of-air fit is
applied.

This is a pressure-coordinate ERA5 diagnostic in the Fleming et al. (1999)
transport lineage. It is not represented as the unavailable isentropic Kyy
field used by Morgan et al. (2004).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
from scipy.io import netcdf_file

from era5_tem_reference import (
    ERA5_TEM_REFERENCE_DOI,
    ERA5_TEM_REFERENCE_VERSION,
    file_md5,
)
from meridional_transport_reference import EARTH_RADIUS_M


ERA5_KYY_ARCHIVES = {
    "u": {
        "name": "ERA5_mon_u_zm_2010s.zip",
        "md5": "cb0d85ed6f69d9882c3da4c346a142cb",
        "url": (
            "https://zenodo.org/api/records/7081721/files/"
            "ERA5_mon_u_zm_2010s.zip/content"
        ),
        "prefix": "ERA5_mon_u_zm_",
        "units": "m s**-1",
    },
    "t": {
        "name": "ERA5_mon_t_zm_2010s.zip",
        "md5": "eacb720d3af2569e8d6db2bb8ead2537",
        "url": (
            "https://zenodo.org/api/records/7081721/files/"
            "ERA5_mon_t_zm_2010s.zip/content"
        ),
        "prefix": "ERA5_mon_t_zm_",
        "units": "K",
    },
    "utendepfd": {
        "name": "ERA5_mon_utendepfd_2010s.zip",
        "md5": "33336b2667b50c2ff9795e2c72e5ffee",
        "url": (
            "https://zenodo.org/api/records/7081721/files/"
            "ERA5_mon_utendepfd_2010s.zip/content"
        ),
        "prefix": "ERA5_mon_utendepfd_",
        "units": "m s-2",
    },
}

QGPV_GRADIENT_FLOOR_PER_M_S = 0.5e-11
EARTH_ROTATION_PER_S = 7.292115e-5
LOG_PRESSURE_SCALE_HEIGHT_M = 7000.0
REFERENCE_PRESSURE_PA = 100000.0
DRY_AIR_GAS_CONSTANT_J_PER_KG_K = 287.05
DRY_AIR_HEAT_CAPACITY_J_PER_KG_K = 1004.0
STANDARD_GRAVITY_M_PER_S2 = 9.80665


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class _ArchiveClimatology:
    latitude_degrees: np.ndarray
    pressure_pa: np.ndarray
    values: np.ndarray
    total_weight_hours: float
    months: tuple[str, ...]


def _load_archive_climatology(
    archive_path: Path,
    variable: str,
    *,
    verify_checksum: bool,
) -> _ArchiveClimatology:
    metadata = ERA5_KYY_ARCHIVES[variable]
    archive_path = Path(archive_path)
    if archive_path.name != metadata["name"]:
        raise ValueError(
            f"expected {metadata['name']} for {variable}, found {archive_path.name}"
        )
    if verify_checksum:
        actual = file_md5(archive_path)
        if actual != metadata["md5"]:
            raise ValueError(
                f"{variable} archive checksum mismatch: "
                f"expected {metadata['md5']}, found {actual}"
            )

    latitude: np.ndarray | None = None
    pressure: np.ndarray | None = None
    weighted_sum: np.ndarray | None = None
    total_hours = 0.0
    months: list[str] = []
    with ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(str(metadata["prefix"])) and name.endswith(".nc")
        )
        for member in members:
            month = Path(member).stem.removeprefix(str(metadata["prefix"]))
            content = archive.read(member)
            if content.startswith(b"CDF"):
                with netcdf_file(BytesIO(content), "r", mmap=False) as dataset:
                    current_latitude = np.asarray(
                        dataset.variables["lat"][:],
                        dtype=float,
                    ).copy()
                    current_pressure = np.asarray(
                        dataset.variables["plev"][:],
                        dtype=float,
                    ).copy()
                    current_values = np.asarray(
                        dataset.variables[variable][0, :, :, 0],
                        dtype=float,
                    ).copy()
                    bounds = np.asarray(
                        dataset.variables["time_bnds"][0, :],
                        dtype=float,
                    ).copy()
                    units = _decode(
                        getattr(dataset.variables[variable], "units", "")
                    )
            else:
                with h5py.File(BytesIO(content), "r") as dataset:
                    version = _decode(dataset.attrs.get("version", ""))
                    if version != ERA5_TEM_REFERENCE_VERSION:
                        raise ValueError(
                            f"unexpected ERA5 diagnostic version {version!r} "
                            f"in {member}"
                        )
                    current_latitude = np.asarray(
                        dataset["lat"][...],
                        dtype=float,
                    )
                    current_pressure = np.asarray(
                        dataset["plev"][...],
                        dtype=float,
                    )
                    current_values = np.asarray(
                        dataset[variable][0, :, :, 0],
                        dtype=float,
                    )
                    bounds = np.asarray(dataset["time_bnds"][0, :], dtype=float)
                    units = _decode(dataset[variable].attrs.get("units", ""))

            if units != metadata["units"]:
                raise ValueError(
                    f"unexpected {variable} units {units!r} in {member}"
                )
            month_hours = float(bounds[1] - bounds[0])
            if latitude is None:
                latitude = current_latitude
                pressure = current_pressure
                weighted_sum = current_values * month_hours
            else:
                if not np.array_equal(current_latitude, latitude):
                    raise ValueError(f"latitude coordinate changed in {member}")
                if not np.array_equal(current_pressure, pressure):
                    raise ValueError(f"pressure coordinate changed in {member}")
                if current_values.shape != weighted_sum.shape:
                    raise ValueError(f"{variable} shape changed in {member}")
                weighted_sum += current_values * month_hours
            total_hours += month_hours
            months.append(month)

    if latitude is None or pressure is None or weighted_sum is None or not months:
        raise ValueError(f"{archive_path} contains no {variable} monthly fields")
    if months[0] != "201001" or months[-1] != "201912" or len(months) != 120:
        raise ValueError(
            "expected complete January 2010 through December 2019 archive; "
            f"found {len(months)} months from {months[0]} to {months[-1]}"
        )
    return _ArchiveClimatology(
        latitude_degrees=latitude,
        pressure_pa=pressure,
        values=weighted_sum / total_hours,
        total_weight_hours=total_hours,
        months=tuple(months),
    )


def _load_archive_month_of_year_climatologies(
    archive_path: Path,
    variable: str,
    *,
    verify_checksum: bool,
) -> tuple[_ArchiveClimatology, ...]:
    """Load twelve calendar-month means from one Serva diagnostic archive."""

    metadata = ERA5_KYY_ARCHIVES[variable]
    archive_path = Path(archive_path)
    if archive_path.name != metadata["name"]:
        raise ValueError(
            f"expected {metadata['name']} for {variable}, found {archive_path.name}"
        )
    if verify_checksum:
        actual = file_md5(archive_path)
        if actual != metadata["md5"]:
            raise ValueError(
                f"{variable} archive checksum mismatch: "
                f"expected {metadata['md5']}, found {actual}"
            )

    latitude: np.ndarray | None = None
    pressure: np.ndarray | None = None
    weighted_sum: list[np.ndarray | None] = [None] * 12
    total_hours = np.zeros(12, dtype=float)
    months: list[list[str]] = [[] for _ in range(12)]
    with ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(str(metadata["prefix"])) and name.endswith(".nc")
        )
        for member in members:
            month = Path(member).stem.removeprefix(str(metadata["prefix"]))
            month_index = int(month[4:6]) - 1
            content = archive.read(member)
            if content.startswith(b"CDF"):
                with netcdf_file(BytesIO(content), "r", mmap=False) as dataset:
                    current_latitude = np.asarray(
                        dataset.variables["lat"][:],
                        dtype=float,
                    ).copy()
                    current_pressure = np.asarray(
                        dataset.variables["plev"][:],
                        dtype=float,
                    ).copy()
                    current_values = np.asarray(
                        dataset.variables[variable][0, :, :, 0],
                        dtype=float,
                    ).copy()
                    bounds = np.asarray(
                        dataset.variables["time_bnds"][0, :],
                        dtype=float,
                    ).copy()
                    units = _decode(
                        getattr(dataset.variables[variable], "units", "")
                    )
            else:
                with h5py.File(BytesIO(content), "r") as dataset:
                    version = _decode(dataset.attrs.get("version", ""))
                    if version != ERA5_TEM_REFERENCE_VERSION:
                        raise ValueError(
                            f"unexpected ERA5 diagnostic version {version!r} "
                            f"in {member}"
                        )
                    current_latitude = np.asarray(
                        dataset["lat"][...],
                        dtype=float,
                    )
                    current_pressure = np.asarray(
                        dataset["plev"][...],
                        dtype=float,
                    )
                    current_values = np.asarray(
                        dataset[variable][0, :, :, 0],
                        dtype=float,
                    )
                    bounds = np.asarray(dataset["time_bnds"][0, :], dtype=float)
                    units = _decode(dataset[variable].attrs.get("units", ""))
            if units != metadata["units"]:
                raise ValueError(
                    f"unexpected {variable} units {units!r} in {member}"
                )
            if latitude is None:
                latitude = current_latitude
                pressure = current_pressure
            else:
                if not np.array_equal(current_latitude, latitude):
                    raise ValueError(f"latitude coordinate changed in {member}")
                if not np.array_equal(current_pressure, pressure):
                    raise ValueError(f"pressure coordinate changed in {member}")
            month_hours = float(bounds[1] - bounds[0])
            if weighted_sum[month_index] is None:
                weighted_sum[month_index] = current_values * month_hours
            else:
                if current_values.shape != weighted_sum[month_index].shape:
                    raise ValueError(f"{variable} shape changed in {member}")
                weighted_sum[month_index] += current_values * month_hours
            total_hours[month_index] += month_hours
            months[month_index].append(month)

    if latitude is None or pressure is None:
        raise ValueError(f"{archive_path} contains no {variable} monthly fields")
    if any(len(values) != 10 for values in months):
        raise ValueError(
            "expected ten fields for each calendar month; found "
            f"{[len(values) for values in months]}"
        )
    result = []
    for month_index in range(12):
        current_sum = weighted_sum[month_index]
        if current_sum is None:
            raise ValueError(f"missing calendar month {month_index + 1}")
        result.append(
            _ArchiveClimatology(
                latitude_degrees=latitude.copy(),
                pressure_pa=pressure.copy(),
                values=current_sum / total_hours[month_index],
                total_weight_hours=float(total_hours[month_index]),
                months=tuple(months[month_index]),
            )
        )
    return tuple(result)


def _one_two_one_meridional(field: np.ndarray) -> np.ndarray:
    result = np.asarray(field, dtype=float).copy()
    result[:, 1:-1] = (
        field[:, :-2] + 2.0 * field[:, 1:-1] + field[:, 2:]
    ) / 4.0
    return result


def quasigeostrophic_pv_gradient_per_m_s(
    *,
    latitude_degrees: np.ndarray,
    pressure_pa: np.ndarray,
    zonal_wind_m_per_s: np.ndarray,
    temperature_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Randel and Garcia (1994), Equation 2 and static stability.

    Inputs use shape ``(pressure, latitude)``, pressure ordered from high to
    low pressure and latitude from south to north.
    """

    latitude = np.asarray(latitude_degrees, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    wind = np.asarray(zonal_wind_m_per_s, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    expected = (len(pressure), len(latitude))
    if wind.shape != expected or temperature.shape != expected:
        raise ValueError("wind and temperature must have shape (pressure, latitude)")
    if not np.all(np.diff(latitude) > 0.0):
        raise ValueError("latitude must increase south to north")
    if not np.all(np.diff(pressure) < 0.0):
        raise ValueError("pressure must decrease upward")
    if not np.all(np.isfinite(wind)) or not np.all(np.isfinite(temperature)):
        raise ValueError("mean-state fields must be finite")

    phi = np.deg2rad(latitude)
    altitude_m = LOG_PRESSURE_SCALE_HEIGHT_M * np.log(
        REFERENCE_PRESSURE_PA / pressure
    )
    potential_temperature = temperature * (
        REFERENCE_PRESSURE_PA / pressure[:, None]
    ) ** (
        DRY_AIR_GAS_CONSTANT_J_PER_KG_K
        / DRY_AIR_HEAT_CAPACITY_J_PER_KG_K
    )
    static_stability_per_s2 = (
        STANDARD_GRAVITY_M_PER_S2
        / potential_temperature
        * np.gradient(
            potential_temperature,
            altitude_m,
            axis=0,
            edge_order=2,
        )
    )
    if np.any(static_stability_per_s2 <= 0.0):
        raise ValueError(
            "Randel-Garcia QGPV diagnostic requires positive static stability"
        )

    cosine = np.cos(phi)[None, :]
    planetary = 2.0 * EARTH_ROTATION_PER_S / EARTH_RADIUS_M * cosine
    curvature_inner = (
        np.gradient(wind * cosine, phi, axis=1, edge_order=2) / cosine
    )
    curvature = (
        -np.gradient(curvature_inner, phi, axis=1, edge_order=2)
        / EARTH_RADIUS_M**2
    )
    coriolis = (2.0 * EARTH_ROTATION_PER_S * np.sin(phi))[None, :]
    vertical_inner = (
        np.exp(-altitude_m[:, None] / LOG_PRESSURE_SCALE_HEIGHT_M)
        / static_stability_per_s2
        * np.gradient(wind, altitude_m, axis=0, edge_order=2)
    )
    vertical = (
        -(coriolis**2)
        * np.exp(altitude_m[:, None] / LOG_PRESSURE_SCALE_HEIGHT_M)
        * np.gradient(vertical_inner, altitude_m, axis=0, edge_order=2)
    )
    gradient = _one_two_one_meridional(planetary + curvature + vertical)
    return gradient, static_stability_per_s2


@dataclass(frozen=True)
class Era5KyyClimatology:
    """Modern Kyy on ERA5 pressure and non-polar latitude nodes."""

    latitude_degrees: np.ndarray
    pressure_pa: np.ndarray
    kyy_m2_per_s: np.ndarray
    raw_qgpv_gradient_per_m_s: np.ndarray
    ep_flux_divergence_acceleration_m_per_s2: np.ndarray
    static_stability_per_s2: np.ndarray
    zonal_wind_m_per_s: np.ndarray
    temperature_k: np.ndarray
    gradient_floor_per_m_s: float
    gradient_floor_fraction: float
    negative_diffusivity_fraction: float
    month_count: int
    total_weight_hours: float
    source: str

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_degrees, dtype=float)
        pressure = np.asarray(self.pressure_pa, dtype=float)
        expected = (len(latitude), len(pressure))
        fields = (
            self.kyy_m2_per_s,
            self.raw_qgpv_gradient_per_m_s,
            self.ep_flux_divergence_acceleration_m_per_s2,
            self.static_stability_per_s2,
            self.zonal_wind_m_per_s,
            self.temperature_k,
        )
        if any(np.asarray(field).shape != expected for field in fields):
            raise ValueError("ERA5 Kyy fields must have shape (latitude, pressure)")
        if not np.all(np.diff(latitude) > 0.0):
            raise ValueError("Kyy latitude must increase south to north")
        if not np.all(np.diff(pressure) < 0.0):
            raise ValueError("Kyy pressure must decrease upward")
        if not np.all(np.isfinite(self.kyy_m2_per_s)):
            raise ValueError("Kyy must be finite")
        if np.any(np.asarray(self.kyy_m2_per_s) < 0.0):
            raise ValueError("Kyy must be non-negative")
        if not np.all(np.isfinite(self.zonal_wind_m_per_s)):
            raise ValueError("zonal wind must be finite")
        if not np.all(np.isfinite(self.temperature_k)) or np.any(
            np.asarray(self.temperature_k) <= 0.0
        ):
            raise ValueError("temperature must be finite and positive")
        if self.month_count <= 0 or self.total_weight_hours <= 0.0:
            raise ValueError("Kyy climatology requires positive time support")
        if not self.source:
            raise ValueError("Kyy climatology requires provenance")

    def model_face_field(
        self,
        *,
        latitude_edges_degrees: np.ndarray,
        pressure_edges_pa: np.ndarray,
    ) -> np.ndarray:
        """Interpolate Kyy to internal latitude faces and cell pressures."""

        latitude_edges = np.asarray(latitude_edges_degrees, dtype=float)
        pressure_edges = np.asarray(pressure_edges_pa, dtype=float)
        target_latitude = latitude_edges[1:-1]
        target_pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
        if target_latitude[0] < self.latitude_degrees[0] or (
            target_latitude[-1] > self.latitude_degrees[-1]
        ):
            raise ValueError("target latitude faces lie outside the Kyy grid")
        if target_pressure[-1] < self.pressure_pa[-1] or (
            target_pressure[0] > self.pressure_pa[0]
        ):
            raise ValueError("target pressure centers lie outside the Kyy grid")

        at_latitude = np.asarray(
            [
                np.interp(
                    target_latitude,
                    self.latitude_degrees,
                    self.kyy_m2_per_s[:, pressure_index],
                )
                for pressure_index in range(len(self.pressure_pa))
            ],
            dtype=float,
        ).T
        source_log_pressure = np.log(self.pressure_pa[::-1])
        target_log_pressure = np.log(target_pressure[::-1])
        mapped = np.asarray(
            [
                np.interp(
                    target_log_pressure,
                    source_log_pressure,
                    row[::-1],
                )[::-1]
                for row in at_latitude
            ],
            dtype=float,
        )
        return mapped


def _diagnose_kyy(
    loaded: dict[str, _ArchiveClimatology],
    *,
    source_time_description: str,
) -> Era5KyyClimatology:
    """Diagnose one Kyy field from aligned u, T, and EP-force means."""

    reference = loaded["u"]
    for variable in ("t", "utendepfd"):
        current = loaded[variable]
        if not np.array_equal(current.latitude_degrees, reference.latitude_degrees):
            raise ValueError(f"{variable} latitude differs from zonal wind")
        if not np.array_equal(current.pressure_pa, reference.pressure_pa):
            raise ValueError(f"{variable} pressure differs from zonal wind")
        if current.months != reference.months:
            raise ValueError(f"{variable} months differ from zonal wind")
        if current.total_weight_hours != reference.total_weight_hours:
            raise ValueError(f"{variable} time weights differ from zonal wind")

    latitude = reference.latitude_degrees[::-1].copy()
    pressure = reference.pressure_pa[::-1].copy()
    wind = reference.values[::-1, ::-1].copy()
    temperature = loaded["t"].values[::-1, ::-1].copy()
    acceleration = loaded["utendepfd"].values[::-1, ::-1].copy()
    gradient, stability = quasigeostrophic_pv_gradient_per_m_s(
        latitude_degrees=latitude,
        pressure_pa=pressure,
        zonal_wind_m_per_s=wind,
        temperature_k=temperature,
    )

    # The Gerber-DynVarMIP spherical EP diagnostic is singular at cos(phi)=0.
    # The transport grid requires only internal faces, so retain all finite
    # non-polar ERA5 nodes and discard the two mathematical pole values.
    latitude = latitude[1:-1]
    gradient = gradient[:, 1:-1]
    stability = stability[:, 1:-1]
    wind = wind[:, 1:-1]
    temperature = temperature[:, 1:-1]
    acceleration = acceleration[:, 1:-1]
    used_gradient = np.maximum(gradient, QGPV_GRADIENT_FLOOR_PER_M_S)
    raw_diffusivity = -acceleration / used_gradient
    negative = raw_diffusivity < 0.0
    kyy = np.where(negative, 0.0, raw_diffusivity)
    if not np.all(np.isfinite(kyy)):
        raise ValueError("ERA5 Kyy diagnosis produced non-finite values")

    return Era5KyyClimatology(
        latitude_degrees=latitude,
        pressure_pa=pressure,
        kyy_m2_per_s=kyy.T,
        raw_qgpv_gradient_per_m_s=gradient.T,
        ep_flux_divergence_acceleration_m_per_s2=acceleration.T,
        static_stability_per_s2=stability.T,
        zonal_wind_m_per_s=wind.T,
        temperature_k=temperature.T,
        gradient_floor_per_m_s=QGPV_GRADIENT_FLOOR_PER_M_S,
        gradient_floor_fraction=float(
            np.mean(gradient < QGPV_GRADIENT_FLOOR_PER_M_S)
        ),
        negative_diffusivity_fraction=float(np.mean(negative)),
        month_count=len(reference.months),
        total_weight_hours=reference.total_weight_hours,
        source=(
            f"Serva (2022) ERA5 {source_time_description} diagnostics, "
            f"doi:{ERA5_TEM_REFERENCE_DOI}; QGPV gradient and 1-2-1 "
            "smoothing from Randel and Garcia (1994), Eq. 2; gradient "
            "floor from Randel and Garcia (1994), Sec. 2; direct "
            "flux-gradient ratio from their Eq. 14; negative Kyy set to "
            "zero following Jiang et al. (2004), Eq. 2"
        ),
    )


def load_era5_kyy_2010s_climatology(
    external_data_directory: Path,
    *,
    verify_checksum: bool = True,
) -> Era5KyyClimatology:
    """Load the three Serva archives and derive annual-mean ERA5 Kyy."""

    directory = Path(external_data_directory)
    loaded = {
        variable: _load_archive_climatology(
            directory / str(metadata["name"]),
            variable,
            verify_checksum=verify_checksum,
        )
        for variable, metadata in ERA5_KYY_ARCHIVES.items()
    }
    return _diagnose_kyy(
        loaded,
        source_time_description="2010-2019 annual-mean",
    )


def load_era5_kyy_month_of_year_climatologies(
    external_data_directory: Path,
    *,
    verify_checksum: bool = True,
) -> tuple[Era5KyyClimatology, ...]:
    """Derive twelve Kyy fields from calendar-month ERA5 climatologies."""

    directory = Path(external_data_directory)
    loaded = {
        variable: _load_archive_month_of_year_climatologies(
            directory / str(metadata["name"]),
            variable,
            verify_checksum=verify_checksum,
        )
        for variable, metadata in ERA5_KYY_ARCHIVES.items()
    }
    result = []
    for month_index in range(12):
        result.append(
            _diagnose_kyy(
                {
                    variable: climatologies[month_index]
                    for variable, climatologies in loaded.items()
                },
                source_time_description=(
                    f"calendar month {month_index + 1:02d}, 2010-2019"
                ),
            )
        )
    return tuple(result)
