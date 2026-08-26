"""Convert I-type cosmic spherule Δ′17O to atmospheric O2 Δ′17O.

Uses Zahnow et al. (2025) Eq. 3 / Fischer et al. (2021) calibration:

    Δ′17O_air = Δ′17O_spherule + 0.0285 * δ18O_spherule - 1.005

All isotope values are per mil, and δ18O is reported vs VSMOW.
"""

from __future__ import annotations


# --- path bootstrap (direct execution) ---
import sys as _sys
from pathlib import Path as _Path
_root = next((p for p in _Path(__file__).resolve().parents if (p / ".project-root").exists()), None)
if _root is not None:
    for _sub in ("code", "validation"):
        _p = str(_root / _sub)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
# --- end path bootstrap ---
FISCHER_SLOPE_MAGNITUDE = 0.0285
FISCHER_SLOPE_1SIGMA = 0.003
FISCHER_INTERCEPT_1SIGMA_PERMIL = 0.13
ZAHNOW_EQ3_OFFSET_PERMIL = 1.005


FISCHER_TABLE5_D18O = [
    48.01,
    46.65,
    38.80,
    42.67,
    46.55,
    43.53,
    46.13,
    42.03,
    46.90,
    41.98,
    44.38,
    41.59,
    39.27,
    40.65,
    40.79,
    44.07,
    41.58,
]


def air_d17o_from_spherule(spherule_d17o: float, spherule_d18o: float) -> float:
    return (
        spherule_d17o
        + FISCHER_SLOPE_MAGNITUDE * spherule_d18o
        - ZAHNOW_EQ3_OFFSET_PERMIL
    )


def spherule_d17o_from_air(air_d17o: float, spherule_d18o: float) -> float:
    return (
        air_d17o
        - FISCHER_SLOPE_MAGNITUDE * spherule_d18o
        + ZAHNOW_EQ3_OFFSET_PERMIL
    )


def analytical_air_d17o_sigma(
    spherule_d17o_sigma: float,
    spherule_d18o_sigma: float,
) -> float:
    """Propagate analytical measurement errors through Zahnow Eq. 3."""

    return (
        spherule_d17o_sigma**2
        + (FISCHER_SLOPE_MAGNITUDE * spherule_d18o_sigma) ** 2
    ) ** 0.5


def independent_calibration_sensitivity_envelope(
    spherule_d17o: float,
    spherule_d18o: float,
    *,
    sigma_level: float = 1.0,
) -> tuple[float, float]:
    """Return an independent-parameter sensitivity envelope for the calibration.

    Fischer et al. (2021) report slope and intercept errors but not their
    covariance. The corner envelope is therefore deliberately not called a
    confidence interval and is not combined with analytical measurement error.
    """

    slopes = (
        FISCHER_SLOPE_MAGNITUDE - sigma_level * FISCHER_SLOPE_1SIGMA,
        FISCHER_SLOPE_MAGNITUDE + sigma_level * FISCHER_SLOPE_1SIGMA,
    )
    offsets = (
        ZAHNOW_EQ3_OFFSET_PERMIL - sigma_level * FISCHER_INTERCEPT_1SIGMA_PERMIL,
        ZAHNOW_EQ3_OFFSET_PERMIL + sigma_level * FISCHER_INTERCEPT_1SIGMA_PERMIL,
    )
    values = [
        spherule_d17o + slope * spherule_d18o - offset
        for slope in slopes
        for offset in offsets
    ]
    return min(values), max(values)


def main() -> None:
    mean_d18o = sum(FISCHER_TABLE5_D18O) / len(FISCHER_TABLE5_D18O)
    spherule_d17o = -10.0
    air_d17o = air_d17o_from_spherule(spherule_d17o, mean_d18o)
    print(f"n Table 5 d18O values = {len(FISCHER_TABLE5_D18O)}")
    print(f"mean d18O_spherule = {mean_d18o:.3f} per mil")
    print(f"D17O_spherule = {spherule_d17o:.3f} per mil")
    print(f"D17O_air O2 = {air_d17o:.3f} per mil")


if __name__ == "__main__":
    main()
