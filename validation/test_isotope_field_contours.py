"""Stress tests for dynamic public isotope-field contours."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from isotope_field_contours import isotope_field_area_weights
from public_model_service import isotope_field


PO2_VALUES = (0.1, 1.0, 2.0)
PCO2_WINDOWS = (
    (50.0, 100.0),
    (50.0, 300.0),
    (50.0, 1000.0),
    (100.0, 3000.0),
    (300.0, 10000.0),
    (1000.0, 60000.0),
    (30000.0, 60000.0),
    (50.0, 60000.0),
)
GPP_WINDOWS = (
    (18.256264, 145.0),
    (18.256264, 435.0),
    (145.0, 435.0),
    (261.0, 319.0),
    (435.0, 850.0),
)


def _case_id(case: tuple[float, tuple[float, float], tuple[float, float]]) -> str:
    po2, pco2, gpp = case
    return f"{po2:g}PAL_{pco2[0]:g}-{pco2[1]:g}ppm_{gpp[0]:g}-{gpp[1]:g}gpp"


CASES = tuple(itertools.product(PO2_VALUES, PCO2_WINDOWS, GPP_WINDOWS))


def _field(case: tuple[float, tuple[float, float], tuple[float, float]]) -> dict:
    po2, pco2, gpp = case
    return isotope_field(
        p_o2_pal=po2,
        pco2_bounds_ppm=pco2,
        gpp_bounds_pgC_per_year=gpp,
        pco2_grid_size=37,
        gpp_grid_size=35,
    )["result"]


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_contours_remain_ordered_readable_and_dense_across_domain(case) -> None:
    result = _field(case)
    levels = np.asarray(result["contour_levels_permil"], dtype=float)
    minimum = float(result["minimum_cap_delta17_permil"])
    maximum = float(result["maximum_cap_delta17_permil"])
    span = maximum - minimum

    assert levels.ndim == 1
    assert len(levels) == len(np.unique(levels))
    assert np.all(np.diff(levels) > 0.0)
    assert np.all((levels > minimum) & (levels < maximum))
    assert len(levels) <= 10
    if span >= 0.009:
        assert len(levels) >= 8

    reference_step = float(result["contour_reference_step_permil"])
    for level in levels:
        if abs(level) >= 10.0 and reference_step >= 1.0:
            assert level == pytest.approx(round(level), abs=1.0e-12)
        elif abs(level) >= 5.0 and reference_step >= 0.5:
            assert 2.0 * level == pytest.approx(round(2.0 * level), abs=1.0e-12)


@pytest.mark.parametrize(
    "pco2,gpp",
    (
        ((250.0, 350.0), (261.0, 319.0)),
        ((50.0, 1000.0), (18.256264, 435.0)),
        ((100.0, 3000.0), (145.0, 362.5)),
        ((50.0, 60000.0), (18.256264, 435.0)),
    ),
)
def test_representative_contours_partition_the_displayed_plot(pco2, gpp) -> None:
    result = isotope_field(
        p_o2_pal=1.0,
        pco2_bounds_ppm=pco2,
        gpp_bounds_pgC_per_year=gpp,
        pco2_grid_size=81,
        gpp_grid_size=71,
    )["result"]
    values = np.asarray(result["central_cap_delta17_permil"], dtype=float).reshape(
        result["field_shape"]
    )
    levels = np.asarray(result["contour_levels_permil"], dtype=float)
    weights = isotope_field_area_weights(
        np.asarray(result["axes"]["pCO2"], dtype=float),
        np.asarray(result["axes"]["GPP"], dtype=float),
    )
    boundaries = np.concatenate(([-np.inf], levels, [np.inf]))
    band_masses = np.asarray(
        [
            np.sum(weights[(values >= lower) & (values < upper)])
            for lower, upper in zip(boundaries[:-1], boundaries[1:])
        ]
    )

    assert np.sum(band_masses) == pytest.approx(1.0, abs=1.0e-12)
    assert np.count_nonzero(band_masses >= 0.02) >= 6
    assert float(np.max(band_masses)) <= 0.36


def test_reported_medium_window_does_not_collapse_to_integer_contours() -> None:
    result = isotope_field(
        p_o2_pal=1.0,
        pco2_bounds_ppm=(50.0, 1000.0),
        gpp_bounds_pgC_per_year=(18.256264, 435.0),
        pco2_grid_size=241,
        gpp_grid_size=201,
    )["result"]
    levels = result["contour_levels_permil"]

    assert len(levels) >= 8
    assert any(abs(level - round(level)) > 1.0e-9 for level in levels)
    assert sum(level > -2.0 for level in levels) >= 4
