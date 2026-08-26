"""Benchmark the passive-tracer bridge against a pinned Photochem profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for subdirectory in ("code", "validation"):
    path_text = str(ROOT / subdirectory)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from photochem_profile import (  # noqa: E402
    PHOTOCHEM_MODERN_EARTH_SHA256,
    eddy_diffusion_column,
    file_sha256,
    load_photochem_v067_modern_earth_profile,
)
from young_model_inventory import PARAMETERS  # noqa: E402


def mean_first_passage_years(
    transport_matrix_per_year: np.ndarray,
    start_layer: int,
    first_target_layer: int,
) -> float:
    """Return the passive-tracer first-passage time to a target altitude."""

    generator = np.asarray(transport_matrix_per_year, dtype=float).T
    transient = np.arange(first_target_layer)
    if start_layer < 0 or start_layer >= first_target_layer:
        raise ValueError("start layer must be below the first target layer")
    times = np.linalg.solve(
        -generator[np.ix_(transient, transient)],
        np.ones(first_target_layer),
    )
    return float(times[start_layer])


def benchmark(atmosphere_path: Path) -> dict[str, float | int | str]:
    profile = load_photochem_v067_modern_earth_profile(atmosphere_path)
    column = eddy_diffusion_column(profile)

    uniform_inventory = 3.0e-4 * column.air_moles
    uniform_tendency = column.derivative(uniform_inventory)
    uniform_scale = (
        float(np.max(np.abs(column.transport_matrix_per_year())))
        * float(np.max(np.abs(uniform_inventory)))
    )

    pulse = np.zeros(len(profile.cells))
    pulse[45] = 1.0
    pulse_tendency = column.derivative(pulse)
    pulse_scale = max(float(np.sum(np.abs(pulse_tendency))), 1.0)

    young_indices = profile.whole_cell_indices(10.0, 60.0)
    young_interval_moles = float(np.sum(profile.air_moles[young_indices]))
    total_moles = float(np.sum(profile.air_moles))
    young_box_moles = float(PARAMETERS["moles_stratosphere"])
    transport_matrix = column.transport_matrix_per_year()
    diffusion_age_to_45km_years = mean_first_passage_years(
        transport_matrix,
        start_layer=10,
        first_target_layer=45,
    )
    liang_age_at_45km_years = 1.0e8 / (365.25 * 24.0 * 3600.0)

    interface_fluxes = np.asarray(
        [
            row["gross_air_flux_mol_per_year"]
            for row in column.interface_rate_constants_per_year()
        ],
        dtype=float,
    )
    return {
        "photochem_profile_sha256": file_sha256(atmosphere_path),
        "expected_sha256": PHOTOCHEM_MODERN_EARTH_SHA256,
        "layer_count": len(profile.cells),
        "lower_altitude_km": profile.lower_altitude_km,
        "upper_altitude_km": profile.upper_altitude_km,
        "total_air_moles_0_100_km": total_moles,
        "air_moles_10_60_km": young_interval_moles,
        "young_stratosphere_box_moles": young_box_moles,
        "ratio_10_60km_to_young_box": young_interval_moles / young_box_moles,
        "kzz_only_first_passage_10_5_to_45km_years": diffusion_age_to_45km_years,
        "liang_reported_age_at_45km_years": liang_age_at_45km_years,
        "kzz_only_to_liang_age_ratio": (
            diffusion_age_to_45km_years / liang_age_at_45km_years
        ),
        "uniform_mixing_ratio_max_abs_tendency_mol_per_year": float(
            np.max(np.abs(uniform_tendency))
        ),
        "uniform_mixing_ratio_relative_roundoff": float(
            np.max(np.abs(uniform_tendency))
        )
        / uniform_scale,
        "pulse_conservation_relative_residual": abs(float(np.sum(pulse_tendency)))
        / pulse_scale,
        "minimum_interface_exchange_mol_per_year": float(np.min(interface_fluxes)),
        "maximum_interface_exchange_mol_per_year": float(np.max(interface_fluxes)),
        "transport_scope": (
            "eddy diffusion only; zero-flux column boundaries; no molecular "
            "diffusion, thermal diffusion, mean advection, or chemistry"
        ),
        "transport_interpretation": (
            "Kzz-only first passage is a diagnostic, not an accepted age-of-air "
            "model; the mismatch identifies missing residual circulation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atmosphere_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "photochem_transport_benchmark.json",
    )
    args = parser.parse_args()

    report = benchmark(args.atmosphere_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
