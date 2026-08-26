"""Validation targets extracted from Young et al. (2014).

These are not model equations. They are paper-level benchmarks used to judge
whether the reconstructed 27-ODE box model behaves like Young's published
solutions across tables, figures, and textual examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import log
from pathlib import Path

import csv
import numpy as np

from young_model_inventory import TABLE3_ISOTOPE_TARGETS, TABLE3_TARGETS


HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
FIG7_DIGITIZED_CONTOURS = _PROJECT_OUTPUTS / "young_fig7_digitized_contours.csv"


@dataclass(frozen=True)
class ScalarTarget:
    key: str
    value: float
    units: str
    source: str
    tolerance: float | None = None
    note: str = ""


TABLE3_SCALAR_TARGETS = tuple(
    ScalarTarget(key, value, "per mil", "Young Table 3", 0.05)
    for key, value in TABLE3_ISOTOPE_TARGETS.items()
)


def young_fig7_294ppm_d17o(gpp_percent: float) -> float:
    """Young text fit for the 294 ppmv isopleth in Fig. 7.

    OCR line: D17O = -0.7397 + 0.07941 ln(|37.3110 - %GPP|).
    This is a validation target for the published numerical solutions, not an
    equation used by the box model.
    """

    return -0.7397 + 0.07941 * log(abs(37.3110 - gpp_percent))


@lru_cache(maxsize=1)
def load_fig7_digitized_contours() -> dict[int, np.ndarray]:
    """Load first-pass digitized Young Fig. 7 pCO2 contours.

    Columns are pCO2 in ppmv, GPP in percent of modern, and atmospheric O2
    Delta'17O in permil. The 294 ppm contour is cross-checked against Young's
    printed equation during digitization QA.
    """

    contours: dict[int, list[tuple[float, float]]] = {}
    with FIG7_DIGITIZED_CONTOURS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            co2 = int(float(row["co2_ppmv"]))
            contours.setdefault(co2, []).append(
                (float(row["gpp_percent"]), float(row["d17o_o2_permil"]))
            )
    return {
        co2: np.array(sorted(values), dtype=float)
        for co2, values in contours.items()
    }


def fig7_digitized_targets(samples_per_contour: int = 15) -> tuple[tuple[int, float, float], ...]:
    """Return evenly sampled Fig. 7 digitized targets.

    Each tuple is `(pCO2_ppmv, GPP_percent, target_Delta17O_permil)`. Sampling
    is even along the visible GPP span of each contour, so the score covers the
    full figure instead of a few selected points.
    """

    targets: list[tuple[int, float, float]] = []
    for co2, curve in load_fig7_digitized_contours().items():
        gpp = curve[:, 0]
        d17o = curve[:, 1]
        grid = np.linspace(float(gpp.min()), float(gpp.max()), samples_per_contour)
        values = np.interp(grid, gpp, d17o)
        targets.extend((co2, float(x), float(y)) for x, y in zip(grid, values))
    return tuple(targets)


TEXTUAL_TARGETS = [
    ScalarTarget(
        "fig9_half_photosynthesis_new_steady_O2_D17O",
        -0.539,
        "per mil",
        "Young text around Fig. 8/Fig. 9",
        0.03,
        "Halving rp and therefore O2 in the NPP experiment shifts steady-state O2 D17O from -0.410 to -0.539.",
    ),
    ScalarTarget(
        "fig7_100ppm_or_70pct_shift",
        -0.050,
        "per mil change",
        "Young text around Fig. 7",
        0.02,
        "Approximate -50 per meg shift can result from +100 ppmv CO2 or GPP decrease to 70%.",
    ),
    ScalarTarget(
        "fig7_70pct_gpp_494ppm_shift",
        -0.080,
        "per mil change",
        "Young text around Fig. 7",
        0.03,
        "70% GPP with CO2 = 494 ppmv corresponds to an -80 per meg shift.",
    ),
]


FIG8_PCO2_POINTS_SPARSE = (294.4, 1000.0, 5000.0, 10000.0, 20000.0, 30000.0)
FIG8_PCO2_POINTS_DENSE = tuple(
    294.4 + (30000.0 - 294.4) * index / 12.0
    for index in range(13)
)


def fig8_pco2_points(*, dense: bool = True) -> tuple[float, ...]:
    """pCO2 sample points for Fig. 8 validation against digitized trends.

    The digitized Young Fig. 8 curves are dense traces. These points define the
    evaluation grid for model residuals. The dense grid is linearly and evenly
    distributed over the full modern-to-high-pCO2 range shown in the figure so
    tuning is not driven mainly by a few endpoints.
    """

    return FIG8_PCO2_POINTS_DENSE if dense else FIG8_PCO2_POINTS_SPARSE
