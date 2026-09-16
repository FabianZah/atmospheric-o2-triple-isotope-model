"""Versioned ERA5 transformed-Eulerian-mean streamfunction reference.

The external archive is Serva (2022), version 0.1.1, DOI
10.5281/zenodo.7081721. It is not distributed with this project. The loader
verifies the published archive checksum and computes a month-duration-weighted
2010-2019 climatology directly from the NetCDF4/HDF5 files in the zip archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np


ERA5_TEM_REFERENCE_DOI = "10.5281/zenodo.7081721"
ERA5_TEM_REFERENCE_VERSION = "0.1.1"
ERA5_TEM_ARCHIVE_NAME = "ERA5_mon_psitem_2010s.zip"
ERA5_TEM_ARCHIVE_MD5 = "688c7c9f738c27354486d35cedf50464"
ERA5_TEM_ARCHIVE_URL = (
    "https://zenodo.org/api/records/7081721/files/"
    "ERA5_mon_psitem_2010s.zip/content"
)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class Era5TemClimatology:
    latitude_degrees: np.ndarray
    pressure_pa: np.ndarray
    streamfunction_kg_per_s: np.ndarray
    month_count: int
    total_weight_hours: float
    start_month: str
    end_month: str
    source: str

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_degrees, dtype=float)
        pressure = np.asarray(self.pressure_pa, dtype=float)
        streamfunction = np.asarray(self.streamfunction_kg_per_s, dtype=float)
        if latitude.ndim != 1 or pressure.ndim != 1:
            raise ValueError("latitude and pressure coordinates must be one-dimensional")
        if streamfunction.shape != (len(pressure), len(latitude)):
            raise ValueError("streamfunction shape must be (pressure, latitude)")
        if not np.all(np.isfinite(streamfunction)):
            raise ValueError("streamfunction must contain only finite values")
        if not np.all(np.diff(latitude) < 0.0):
            raise ValueError("ERA5 source latitude must be ordered north to south")
        if not np.all(np.diff(pressure) > 0.0):
            raise ValueError("ERA5 source pressure must increase downward")
        if self.month_count <= 0 or self.total_weight_hours <= 0.0:
            raise ValueError("climatology must contain positive time support")
        if not self.source:
            raise ValueError("climatology requires provenance")

    def ten_degree_native_pressure_nodes(
        self,
        set_physical_pole_values_to_zero: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact 10-degree latitude nodes on native pressure levels.

        Output is ordered from south to north and from high to low pressure, so
        it follows the cell-corner convention of the circulation operator.
        """

        target_latitude = np.arange(-90.0, 91.0, 10.0)
        latitude_indices = []
        for latitude in target_latitude:
            matches = np.flatnonzero(
                np.isclose(
                    self.latitude_degrees,
                    latitude,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            if matches.size != 1:
                raise ValueError(f"ERA5 latitude grid does not contain {latitude:g} degrees")
            latitude_indices.append(int(matches[0]))

        pressure_descending = self.pressure_pa[::-1].copy()
        streamfunction = self.streamfunction_kg_per_s[
            ::-1,
            latitude_indices,
        ].T.copy()
        if set_physical_pole_values_to_zero:
            streamfunction[0, :] = 0.0
            streamfunction[-1, :] = 0.0
        return target_latitude, pressure_descending, streamfunction

    def maximum_absolute_streamfunction(
        self,
        maximum_pressure_pa: float,
        minimum_pressure_pa: float = 0.0,
        exclude_poles: bool = True,
    ) -> dict[str, float]:
        if not maximum_pressure_pa > minimum_pressure_pa >= 0.0:
            raise ValueError("pressure bounds must be ordered and non-negative")
        pressure_mask = (
            (self.pressure_pa <= maximum_pressure_pa)
            & (self.pressure_pa >= minimum_pressure_pa)
        )
        if not np.any(pressure_mask):
            raise ValueError("pressure interval does not include an ERA5 level")
        latitude_slice = slice(1, -1) if exclude_poles else slice(None)
        subset = self.streamfunction_kg_per_s[pressure_mask, latitude_slice]
        local_index = np.unravel_index(np.argmax(np.abs(subset)), subset.shape)
        pressure_indices = np.flatnonzero(pressure_mask)
        latitude_offset = 1 if exclude_poles else 0
        pressure_index = int(pressure_indices[local_index[0]])
        latitude_index = int(local_index[1] + latitude_offset)
        return {
            "streamfunction_kg_per_s": float(
                self.streamfunction_kg_per_s[pressure_index, latitude_index]
            ),
            "absolute_streamfunction_kg_per_s": float(
                abs(self.streamfunction_kg_per_s[pressure_index, latitude_index])
            ),
            "pressure_pa": float(self.pressure_pa[pressure_index]),
            "latitude_degrees": float(self.latitude_degrees[latitude_index]),
        }


def load_era5_tem_2010s_climatology(
    archive_path: Path,
    verify_checksum: bool = True,
) -> Era5TemClimatology:
    archive_path = Path(archive_path)
    if verify_checksum:
        actual_checksum = file_md5(archive_path)
        if actual_checksum != ERA5_TEM_ARCHIVE_MD5:
            raise ValueError(
                "ERA5 TEM archive checksum mismatch: "
                f"expected {ERA5_TEM_ARCHIVE_MD5}, found {actual_checksum}"
            )

    weighted_sum: np.ndarray | None = None
    total_hours = 0.0
    latitude: np.ndarray | None = None
    pressure: np.ndarray | None = None
    months: list[str] = []

    with ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ERA5_mon_psitem_20") and name.endswith(".nc")
        )
        for member in members:
            month = Path(member).stem.removeprefix("ERA5_mon_psitem_")
            with h5py.File(BytesIO(archive.read(member)), "r") as dataset:
                version = _text_attribute(dataset.attrs.get("version", ""))
                if version != ERA5_TEM_REFERENCE_VERSION:
                    raise ValueError(
                        f"unexpected ERA5 TEM file version {version!r} in {member}"
                    )
                units = _text_attribute(dataset["psitem"].attrs.get("units", ""))
                if units != "kg s-1":
                    raise ValueError(
                        f"unexpected streamfunction units {units!r} in {member}"
                    )
                current_latitude = np.asarray(dataset["lat"][...], dtype=float)
                current_pressure = np.asarray(dataset["plev"][...], dtype=float)
                current_streamfunction = np.asarray(
                    dataset["psitem"][0, :, :, 0],
                    dtype=float,
                )
                bounds = np.asarray(dataset["time_bnds"][0, :], dtype=float)
                month_hours = float(bounds[1] - bounds[0])

            if latitude is None:
                latitude = current_latitude
                pressure = current_pressure
                weighted_sum = current_streamfunction * month_hours
            else:
                if not np.array_equal(current_latitude, latitude):
                    raise ValueError(f"latitude coordinate changed in {member}")
                if not np.array_equal(current_pressure, pressure):
                    raise ValueError(f"pressure coordinate changed in {member}")
                if current_streamfunction.shape != weighted_sum.shape:
                    raise ValueError(f"streamfunction shape changed in {member}")
                weighted_sum += current_streamfunction * month_hours
            total_hours += month_hours
            months.append(month)

    if (
        latitude is None
        or pressure is None
        or weighted_sum is None
        or not months
    ):
        raise ValueError("ERA5 TEM archive contains no monthly streamfunction files")
    if months[0] != "201001" or months[-1] != "201912" or len(months) != 120:
        raise ValueError(
            "expected complete January 2010 through December 2019 archive; "
            f"found {len(months)} months from {months[0]} to {months[-1]}"
        )

    return Era5TemClimatology(
        latitude_degrees=latitude,
        pressure_pa=pressure,
        streamfunction_kg_per_s=weighted_sum / total_hours,
        month_count=len(months),
        total_weight_hours=total_hours,
        start_month=months[0],
        end_month=months[-1],
        source=(
            "Serva (2022), ERA5 monthly transformed-Eulerian-mean data "
            f"v{ERA5_TEM_REFERENCE_VERSION}, doi:{ERA5_TEM_REFERENCE_DOI}"
        ),
    )


def load_era5_tem_month_of_year_climatologies(
    archive_path: Path,
    verify_checksum: bool = True,
) -> tuple[Era5TemClimatology, ...]:
    """Return twelve 2010-2019 calendar-month TEM climatologies.

    Each calendar month is averaged over ten years using the exact time bounds
    stored in the Serva archive. The result preserves the January-to-December
    seasonal cycle without introducing an interpolated circulation field.
    """

    archive_path = Path(archive_path)
    if verify_checksum:
        actual_checksum = file_md5(archive_path)
        if actual_checksum != ERA5_TEM_ARCHIVE_MD5:
            raise ValueError(
                "ERA5 TEM archive checksum mismatch: "
                f"expected {ERA5_TEM_ARCHIVE_MD5}, found {actual_checksum}"
            )
    weighted_sum: list[np.ndarray | None] = [None] * 12
    total_hours = np.zeros(12, dtype=float)
    counts = np.zeros(12, dtype=int)
    first_month: list[str | None] = [None] * 12
    last_month: list[str | None] = [None] * 12
    latitude: np.ndarray | None = None
    pressure: np.ndarray | None = None

    with ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ERA5_mon_psitem_20") and name.endswith(".nc")
        )
        for member in members:
            month = Path(member).stem.removeprefix("ERA5_mon_psitem_")
            month_index = int(month[4:6]) - 1
            with h5py.File(BytesIO(archive.read(member)), "r") as dataset:
                version = _text_attribute(dataset.attrs.get("version", ""))
                if version != ERA5_TEM_REFERENCE_VERSION:
                    raise ValueError(
                        f"unexpected ERA5 TEM file version {version!r} in {member}"
                    )
                units = _text_attribute(dataset["psitem"].attrs.get("units", ""))
                if units != "kg s-1":
                    raise ValueError(
                        f"unexpected streamfunction units {units!r} in {member}"
                    )
                current_latitude = np.asarray(dataset["lat"][...], dtype=float)
                current_pressure = np.asarray(dataset["plev"][...], dtype=float)
                current_streamfunction = np.asarray(
                    dataset["psitem"][0, :, :, 0],
                    dtype=float,
                )
                bounds = np.asarray(dataset["time_bnds"][0, :], dtype=float)
                month_hours = float(bounds[1] - bounds[0])
            if latitude is None:
                latitude = current_latitude
                pressure = current_pressure
            else:
                if not np.array_equal(current_latitude, latitude):
                    raise ValueError(f"latitude coordinate changed in {member}")
                if not np.array_equal(current_pressure, pressure):
                    raise ValueError(f"pressure coordinate changed in {member}")
            if weighted_sum[month_index] is None:
                weighted_sum[month_index] = current_streamfunction * month_hours
                first_month[month_index] = month
            else:
                if current_streamfunction.shape != weighted_sum[month_index].shape:
                    raise ValueError(f"streamfunction shape changed in {member}")
                weighted_sum[month_index] += current_streamfunction * month_hours
            total_hours[month_index] += month_hours
            counts[month_index] += 1
            last_month[month_index] = month

    if latitude is None or pressure is None:
        raise ValueError("ERA5 TEM archive contains no monthly streamfunction files")
    if not np.all(counts == 10):
        raise ValueError(
            "expected ten fields for every calendar month; "
            f"found {counts.tolist()}"
        )
    results = []
    for month_index in range(12):
        current_sum = weighted_sum[month_index]
        if current_sum is None:
            raise ValueError(f"missing calendar month {month_index + 1}")
        results.append(
            Era5TemClimatology(
                latitude_degrees=latitude.copy(),
                pressure_pa=pressure.copy(),
                streamfunction_kg_per_s=(
                    current_sum / total_hours[month_index]
                ),
                month_count=int(counts[month_index]),
                total_weight_hours=float(total_hours[month_index]),
                start_month=str(first_month[month_index]),
                end_month=str(last_month[month_index]),
                source=(
                    "Serva (2022), ERA5 calendar-month TEM climatology "
                    f"for month {month_index + 1:02d}, 2010-2019, "
                    f"v{ERA5_TEM_REFERENCE_VERSION}, "
                    f"doi:{ERA5_TEM_REFERENCE_DOI}"
                ),
            )
        )
    return tuple(results)
