"""Tests for updated photosynthesis-at-fixed-respiration feedback."""

from __future__ import annotations

import numpy as np
import pytest

from updated_molecular_forward_model import UpdatedForwardInput
from updated_photosynthesis_transient import (
    UpdatedPhotosynthesisTransientInput,
    run_updated_photosynthesis_transient,
)
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


@pytest.fixture(scope="module")
def half_photosynthesis_result():
    return run_updated_photosynthesis_transient(
        UpdatedPhotosynthesisTransientInput(
            initial=UpdatedForwardInput(1.0, 294.4, 290.0),
            photosynthesis_fraction=0.5,
            duration_years=12000.0,
            sample_count=121,
            equilibrium_search_max_years=50000.0,
        )
    )


def test_half_photosynthesis_has_no_instantaneous_isotope_jump(
    half_photosynthesis_result,
) -> None:
    first = half_photosynthesis_result.states[0]
    assert first.cap_delta17_prime_permil == pytest.approx(-0.42655, abs=0.002)
    assert first.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL == pytest.approx(1.0)


def test_half_photosynthesis_produces_fig9_type_overshoot(
    half_photosynthesis_result,
) -> None:
    values = np.asarray(
        [state.cap_delta17_prime_permil for state in half_photosynthesis_result.states]
    )
    minimum_index = int(np.argmin(values))
    long_run = half_photosynthesis_result.long_run_state.cap_delta17_prime_permil
    assert 1000.0 < half_photosynthesis_result.time_years[minimum_index] < 6000.0
    assert values[minimum_index] < long_run - 0.10
    assert half_photosynthesis_result.operational_equilibrium_time_years is not None


def test_half_photosynthesis_evolves_o2_and_co2_inventories(
    half_photosynthesis_result,
) -> None:
    rows = half_photosynthesis_result.rows()
    assert min(row["pO2_pal"] for row in rows) < 0.9
    assert max(row["pCO2_ppm"] for row in rows) > 500.0
    assert half_photosynthesis_result.solver["carbon_success"] is True
    assert half_photosynthesis_result.request.carbon_sink_multiplier == 1.0
    budget = half_photosynthesis_result.solver["carbon_budget"]
    assert budget["sink_multiplier"] == 1.0
    assert budget["weathering_per_year"] == .000208
    assert budget["ocean_uptake_per_year"] == .1172
    assert budget["constant_balancing_flux_mol_per_year"] == pytest.approx(
        (.000208+.1172)*budget['balancing_reference_CO2_mol']-5.884e15
        +budget['initial_biological_o2_tendency_mol_per_year'])
    assert np.isfinite(
        half_photosynthesis_result.solver[
            "carbon_minimum_unclipped_inventory_mol"
        ]
    )


def test_half_photosynthesis_reproduces_all_fig9_panel_directions(
    half_photosynthesis_result,
) -> None:
    rows = half_photosynthesis_result.rows()
    initial = rows[0]

    assert min(row["pO2_pal"] for row in rows) < initial["pO2_pal"]
    assert max(row["pCO2_ppm"] for row in rows) > initial["pCO2_ppm"]
    assert (
        max(row["O2_delta18_prime_permil"] for row in rows)
        > initial["O2_delta18_prime_permil"]
    )
    assert (
        min(row["O2_cap_delta17_prime_permil"] for row in rows)
        < half_photosynthesis_result.long_run_state.cap_delta17_prime_permil
    )


def test_carbon_and_oxygen_share_one_inventory(half_photosynthesis_result) -> None:
    molecular_po2 = np.asarray(
        [row["pO2_pal"] for row in half_photosynthesis_result.rows()]
    )
    carbon_po2 = np.asarray(half_photosynthesis_result.carbon_driver_po2_pal)

    np.testing.assert_array_equal(molecular_po2, carbon_po2)
