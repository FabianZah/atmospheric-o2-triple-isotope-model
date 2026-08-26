"""Portable reduced-grid ERA5 convection climatology artifact."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from era5_convection_adapter import Era5ZonalConvectionMonth


_FIELDS = (
    "updraught_interface_mol_per_year",
    "downdraught_interface_mol_per_year",
    "updraught_detrainment_mol_per_year",
    "downdraught_detrainment_mol_per_year",
    "updraught_negative_entrainment_mol_per_year",
    "downdraught_negative_entrainment_mol_per_year",
)


@dataclass(frozen=True)
class Era5ConvectionClimatology:
    """Calendar-month operators and their day-weighted annual mean."""

    calendar_months: np.ndarray
    monthly: tuple[Era5ZonalConvectionMonth, ...]
    annual_mean: Era5ZonalConvectionMonth
    month_weights: np.ndarray
    source_years: tuple[int, ...]

    def __post_init__(self) -> None:
        calendar_months = np.asarray(self.calendar_months, dtype=int)
        weights = np.asarray(self.month_weights, dtype=float)
        if calendar_months.shape != (len(self.monthly),):
            raise ValueError("calendar months must match monthly operators")
        if weights.shape != calendar_months.shape or np.any(weights <= 0.0):
            raise ValueError("climatology month weights must be positive")
        if len(calendar_months) != len(set(calendar_months.tolist())):
            raise ValueError("calendar months must be unique")

    def save(self, path: Path) -> None:
        """Write a compressed, pickle-free numerical archive."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "calendar_months": np.asarray(self.calendar_months, dtype=int),
            "month_weights": np.asarray(self.month_weights, dtype=float),
            "latitude_degrees": np.asarray(
                self.annual_mean.latitude_degrees, dtype=float
            ),
            "layer_air_moles": np.asarray(self.annual_mean.layer_air_moles, dtype=float),
        }
        for field in _FIELDS:
            arrays[f"monthly_{field}"] = np.stack(
                [np.asarray(getattr(month, field), dtype=float) for month in self.monthly]
            )
            arrays[f"annual_{field}"] = np.asarray(
                getattr(self.annual_mean, field), dtype=float
            )
        metadata = {
            "source_years": list(self.source_years),
            "monthly_source_files": [
                list(month.source_files) for month in self.monthly
            ],
            "annual_source_files": list(self.annual_mean.source_files),
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(target, **arrays)


def load_era5_convection_climatology(path: Path) -> Era5ConvectionClimatology:
    """Load a reduced-grid climatology without requiring raw ERA5 files."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        latitude = np.asarray(archive["latitude_degrees"], dtype=float)
        air = np.asarray(archive["layer_air_moles"], dtype=float)
        calendar_months = np.asarray(archive["calendar_months"], dtype=int)

        def month_from_archive(prefix: str, index: int | None, sources) -> Era5ZonalConvectionMonth:
            values = {}
            for field in _FIELDS:
                array = np.asarray(archive[f"{prefix}_{field}"], dtype=float)
                values[field] = array if index is None else array[index]
            return Era5ZonalConvectionMonth(
                latitude_degrees=latitude.copy(),
                layer_air_moles=air.copy(),
                source_files=tuple(sources),
                **values,
            )

        monthly = tuple(
            month_from_archive(
                "monthly", index, metadata["monthly_source_files"][index]
            )
            for index in range(len(calendar_months))
        )
        annual = month_from_archive(
            "annual", None, metadata["annual_source_files"]
        )
        return Era5ConvectionClimatology(
            calendar_months=calendar_months,
            monthly=monthly,
            annual_mean=annual,
            month_weights=np.asarray(archive["month_weights"], dtype=float),
            source_years=tuple(int(year) for year in metadata["source_years"]),
        )
