"""Regression tests for updated-model slow atmospheric O2 feedback."""

from __future__ import annotations

import pytest

from updated_molecular_forward_model import UpdatedForwardInput
from updated_molecular_transient import (
    UpdatedTransientInput,
    run_updated_transient,
)


INITIAL = UpdatedForwardInput(1.0, 294.0, 290.0)


@pytest.mark.parametrize(
    "final",
    (
        UpdatedForwardInput(1.0, 1000.0, 290.0),
        UpdatedForwardInput(1.0, 294.0, 145.0),
    ),
)
def test_boundary_steps_do_not_instantaneously_change_o2_isotopes(final) -> None:
    result = run_updated_transient(
        UpdatedTransientInput(INITIAL, final, duration_years=100.0, sample_count=5)
    )
    first = result.states[0]
    assert first.delta18_prime_permil == pytest.approx(
        result.initial_steady_state["delta18_prime_permil"], abs=1.0e-12
    )
    assert first.cap_delta17_prime_permil == pytest.approx(
        result.initial_steady_state["cap_delta17_prime_permil"], abs=1.0e-12
    )
    assert result.instantaneous_po2_inventory_step is False


def test_po2_step_preserves_initial_ratios_before_relaxation() -> None:
    result = run_updated_transient(
        UpdatedTransientInput(
            INITIAL,
            UpdatedForwardInput(0.5, 294.0, 290.0),
            duration_years=100.0,
            sample_count=5,
        )
    )
    first = result.states[0]
    assert first.delta18_prime_permil == pytest.approx(
        result.initial_steady_state["delta18_prime_permil"], abs=1.0e-12
    )
    assert first.cap_delta17_prime_permil == pytest.approx(
        result.initial_steady_state["cap_delta17_prime_permil"], abs=1.0e-12
    )
    assert result.instantaneous_po2_inventory_step is True


@pytest.mark.parametrize(
    "final",
    (
        UpdatedForwardInput(1.0, 1000.0, 290.0),
        UpdatedForwardInput(1.0, 294.0, 145.0),
        UpdatedForwardInput(0.5, 294.0, 290.0),
    ),
)
def test_long_transient_approaches_updated_forward_steady_state(final) -> None:
    result = run_updated_transient(
        UpdatedTransientInput(INITIAL, final, duration_years=50000.0, sample_count=41)
    )
    last = result.states[-1]
    assert last.delta18_prime_permil == pytest.approx(
        result.final_steady_state["delta18_prime_permil"], abs=2.0e-4
    )
    assert last.cap_delta17_prime_permil == pytest.approx(
        result.final_steady_state["cap_delta17_prime_permil"], abs=8.0e-4
    )
    assert result.solver["success"] is True


def test_operational_equilibrium_time_is_reported_for_long_run() -> None:
    result = run_updated_transient(
        UpdatedTransientInput(
            INITIAL,
            UpdatedForwardInput(1.0, 1000.0, 290.0),
            duration_years=50000.0,
            sample_count=201,
        )
    )
    equilibrium_time = result.operational_equilibrium_time_years()
    assert equilibrium_time is not None
    assert 0.0 < equilibrium_time < result.time_years[-1]
    assert result.as_dict()["operational_equilibrium"]["time_years"] == pytest.approx(
        equilibrium_time
    )


def test_short_display_run_still_reports_independent_equilibrium_time() -> None:
    result = run_updated_transient(
        UpdatedTransientInput(
            INITIAL,
            UpdatedForwardInput(1.0, 1000.0, 290.0),
            duration_years=100.0,
            sample_count=5,
        )
    )
    equilibrium_time = result.operational_equilibrium_time_years()
    assert equilibrium_time is not None
    assert equilibrium_time > result.request.duration_years
    assert result.time_years[-1] == pytest.approx(100.0)
    assert result.solver["equilibrium_search_horizon_years"] == pytest.approx(
        100000.0
    )


def test_gpp_step_accepts_nondefault_display_duration() -> None:
    result = run_updated_transient(
        UpdatedTransientInput(
            INITIAL,
            UpdatedForwardInput(1.0, 294.0, 145.0),
            duration_years=5000.0,
            sample_count=81,
        )
    )
    assert result.time_years[-1] == pytest.approx(5000.0)
    assert result.operational_equilibrium_time_years() is not None
    assert result.operational_equilibrium_time_years() > 5000.0


def test_fig10_type_pco2_step_has_young_like_monotonic_shape() -> None:
    result = run_updated_transient(
        UpdatedTransientInput(
            INITIAL,
            UpdatedForwardInput(1.0, 400.0, 290.0),
            duration_years=5000.0,
            sample_count=1001,
        )
    )
    shifts = [
        state.cap_delta17_prime_permil
        - result.states[0].cap_delta17_prime_permil
        for state in result.states
    ]

    assert all(later <= earlier + 1.0e-10 for earlier, later in zip(shifts, shifts[1:]))
    assert shifts[30] == pytest.approx(-0.00433192, abs=2.0e-5)
    assert shifts[-1] == pytest.approx(-0.04940769, abs=2.0e-5)
