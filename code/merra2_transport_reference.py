"""Checksum-pinned MERRA-2 state and gravity-wave-drag climatologies.

The public daily zonal-mean archives are from Serva (2022), Zenodo record
6959944. They retain the native MERRA-2 gravity-wave-drag tendency DUDTGWD,
unlike the aggregate ERA5 parameterized tendency. Daily fields are averaged
into twelve 2010-2019 calendar-month climatologies without fitted scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np

from era5_kyy_reference import quasigeostrophic_pv_gradient_per_m_s
from era5_tem_reference import file_md5


MERRA2_DAILY_ARCHIVES = {
    "DUDTGWD": {
        "name": "MERRA2_day_DUDTGWD_zm_2010s.zip",
        "md5": "9cfce6764ec688d09458672881f089ca",
        "url": (
            "https://zenodo.org/api/records/6959944/files/"
            "MERRA2_day_DUDTGWD_zm_2010s.zip/content"
        ),
        "prefix": "MERRA2_day_DUDTGWD_zm_",
        "units": "m s-2",
    },
    "u": {
        "name": "MERRA2_day_u_zm_2010s.zip",
        "md5": "c78ca35b1d81d23ee2d8a1aadd571e18",
        "url": (
            "https://zenodo.org/api/records/6959944/files/"
            "MERRA2_day_u_zm_2010s.zip/content"
        ),
        "prefix": "MERRA2_day_u_zm_",
        "units": "m s-1",
    },
    "t": {
        "name": "MERRA2_day_t_zm_2010s.zip",
        "md5": "9e2d1849f36ca1fadfbc8905e73131e4",
        "url": (
            "https://zenodo.org/api/records/6959944/files/"
            "MERRA2_day_t_zm_2010s.zip/content"
        ),
        "prefix": "MERRA2_day_t_zm_",
        "units": "K",
    },
}


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


@dataclass(frozen=True)
class Merra2GravityWaveState:
    """One monthly MERRA-2 climatological atmospheric state."""

    month: int
    latitude_degrees: np.ndarray
    pressure_pa: np.ndarray
    zonal_wind_m_per_s: np.ndarray
    temperature_k: np.ndarray
    static_stability_per_s2: np.ndarray
    gravity_wave_drag_m_per_s2: np.ndarray
    day_count: int
    source: str

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_degrees, dtype=float)
        pressure = np.asarray(self.pressure_pa, dtype=float)
        expected = (len(latitude), len(pressure))
        fields = (
            self.zonal_wind_m_per_s,
            self.temperature_k,
            self.static_stability_per_s2,
            self.gravity_wave_drag_m_per_s2,
        )
        if self.month < 1 or self.month > 12:
            raise ValueError("month must be in 1..12")
        if any(np.asarray(field).shape != expected for field in fields):
            raise ValueError("MERRA-2 fields must have shape (latitude, pressure)")
        if not np.all(np.diff(latitude) > 0.0):
            raise ValueError("latitude must increase south to north")
        if not np.all(np.diff(pressure) < 0.0):
            raise ValueError("pressure must decrease upward")
        if any(not np.all(np.isfinite(field)) for field in fields):
            raise ValueError("MERRA-2 fields must be finite")
        if np.any(self.temperature_k <= 0.0):
            raise ValueError("temperature must be positive")
        if np.any(self.static_stability_per_s2 <= 0.0):
            raise ValueError("static stability must be positive")
        if self.day_count < 280 or self.day_count > 310:
            raise ValueError("calendar-month climatology has unexpected day support")
        if not self.source:
            raise ValueError("MERRA-2 state requires provenance")


@dataclass(frozen=True)
class _MonthlyField:
    latitude_degrees: np.ndarray
    pressure_pa: np.ndarray
    values: tuple[np.ndarray, ...]
    day_counts: tuple[int, ...]


def _load_monthly_field(
    archive_path: Path,
    variable: str,
    *,
    verify_checksum: bool,
) -> _MonthlyField:
    metadata = MERRA2_DAILY_ARCHIVES[variable]
    path = Path(archive_path)
    if path.name != metadata["name"]:
        raise ValueError(f"expected {metadata['name']}, found {path.name}")
    if verify_checksum:
        actual = file_md5(path)
        if actual != metadata["md5"]:
            raise ValueError(
                f"{variable} checksum mismatch: expected {metadata['md5']}, found {actual}"
            )

    latitude = None
    pressure = None
    sums: list[np.ndarray | None] = [None] * 12
    counts = np.zeros(12, dtype=int)
    with ZipFile(path) as archive:
        members = sorted(
            member
            for member in archive.namelist()
            if member.startswith(str(metadata["prefix"])) and member.endswith(".nc")
        )
        if len(members) != 120:
            raise ValueError(f"expected 120 monthly files, found {len(members)}")
        for member in members:
            stamp = Path(member).stem.removeprefix(str(metadata["prefix"]))
            month_index = int(stamp[4:6]) - 1
            with h5py.File(BytesIO(archive.read(member)), "r") as dataset:
                current_latitude = np.asarray(dataset["lat"][...], dtype=float)
                pressure_name = "lev" if "lev" in dataset else "plev"
                current_pressure = np.asarray(dataset[pressure_name][...], dtype=float)
                current_values = np.asarray(dataset[variable][..., 0], dtype=float)
                units = _decode(dataset[variable].attrs.get("units", ""))
                fill_value = float(
                    np.asarray(
                        dataset[variable].attrs.get("_FillValue", np.inf)
                    ).ravel()[0]
                )
            if units != metadata["units"]:
                raise ValueError(f"unexpected {variable} units {units!r} in {member}")
            # DUDTGWD labels lev as hPa, but the numerical coordinate is exactly
            # the same 3-100000 Pa coordinate stored by u and t. Preserve the
            # numeric coordinate and verify cross-variable equality below.
            missing = np.abs(current_values) >= abs(fill_value)
            if np.any(missing):
                all_missing_level = np.all(missing, axis=(0, 2))
                any_missing_level = np.any(missing, axis=(0, 2))
                if not np.array_equal(all_missing_level, any_missing_level):
                    raise ValueError(
                        f"{variable} has partially missing pressure levels in {member}"
                    )
                current_values = current_values.copy()
                current_values[:, all_missing_level, :] = np.nan
            if latitude is None:
                latitude = current_latitude
                pressure = current_pressure
            elif not np.array_equal(current_latitude, latitude) or not np.array_equal(
                current_pressure, pressure
            ):
                raise ValueError(f"coordinate changed in {member}")
            monthly_sum = np.sum(current_values, axis=0)
            if sums[month_index] is None:
                sums[month_index] = monthly_sum
            else:
                sums[month_index] += monthly_sum
            counts[month_index] += current_values.shape[0]

    if latitude is None or pressure is None or any(value is None for value in sums):
        raise ValueError(f"{path} did not provide complete monthly fields")
    values = tuple(
        np.asarray(sums[index], dtype=float) / counts[index] for index in range(12)
    )
    return _MonthlyField(
        latitude_degrees=np.asarray(latitude, dtype=float),
        pressure_pa=np.asarray(pressure, dtype=float),
        values=values,
        day_counts=tuple(int(value) for value in counts),
    )


def load_merra2_gravity_wave_monthly_states(
    external_data_directory: Path,
    *,
    verify_checksum: bool = True,
) -> tuple[Merra2GravityWaveState, ...]:
    """Load matched 2010-2019 MERRA-2 monthly u, T, N2, and DUDTGWD."""

    directory = Path(external_data_directory)
    fields = {
        variable: _load_monthly_field(
            directory / str(metadata["name"]),
            variable,
            verify_checksum=verify_checksum,
        )
        for variable, metadata in MERRA2_DAILY_ARCHIVES.items()
    }
    reference = fields["u"]
    for variable, field in fields.items():
        if not np.array_equal(field.latitude_degrees, reference.latitude_degrees):
            raise ValueError(f"{variable} latitude differs from u")
        if not np.array_equal(field.pressure_pa, reference.pressure_pa):
            raise ValueError(f"{variable} pressure differs from u")
        if field.day_counts != reference.day_counts:
            raise ValueError(f"{variable} day support differs from u")

    valid_pressure = np.ones(len(reference.pressure_pa), dtype=bool)
    for field in fields.values():
        valid_pressure &= np.asarray(
            [
                all(np.all(np.isfinite(month[level])) for month in field.values)
                for level in range(len(reference.pressure_pa))
            ],
            dtype=bool,
        )
    if not np.any(valid_pressure):
        raise ValueError("MERRA-2 fields have no common finite pressure domain")

    # Source files run from the model top downward and north to south. Return
    # the same public orientation as the ERA5 adapters and omit singular poles.
    latitude = reference.latitude_degrees[::-1][1:-1].copy()
    pressure = reference.pressure_pa[valid_pressure][::-1].copy()
    result = []
    for month_index in range(12):
        wind_vertical_latitude = fields["u"].values[month_index][valid_pressure][::-1, ::-1][:, 1:-1]
        temperature_vertical_latitude = fields["t"].values[month_index][valid_pressure][::-1, ::-1][:, 1:-1]
        drag_vertical_latitude = fields["DUDTGWD"].values[month_index][valid_pressure][::-1, ::-1][:, 1:-1]
        _, stability = quasigeostrophic_pv_gradient_per_m_s(
            latitude_degrees=latitude,
            pressure_pa=pressure,
            zonal_wind_m_per_s=wind_vertical_latitude,
            temperature_k=temperature_vertical_latitude,
        )
        result.append(
            Merra2GravityWaveState(
                month=month_index + 1,
                latitude_degrees=latitude,
                pressure_pa=pressure,
                zonal_wind_m_per_s=wind_vertical_latitude.T,
                temperature_k=temperature_vertical_latitude.T,
                static_stability_per_s2=stability.T,
                gravity_wave_drag_m_per_s2=drag_vertical_latitude.T,
                day_count=reference.day_counts[month_index],
                source=(
                    "Serva (2022) MERRA-2 daily zonal means, 2010-2019 "
                    "calendar-month climatology, doi:10.5281/zenodo.6959944; "
                    "DUDTGWD is the native MERRA-2 gravity-wave-drag tendency; "
                    "the fully missing 3, 5, and 7 Pa DUDTGWD levels are excluded"
                ),
            )
        )
    return tuple(result)
