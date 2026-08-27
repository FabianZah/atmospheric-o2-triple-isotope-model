"""Tests for prescribed gradual pCO2 forcing."""

from __future__ import annotations

import pytest

from updated_molecular_forward_model import UpdatedForwardInput
from updated_pco2_trajectory_transient import (
    UpdatedPCO2TrajectoryInput,
    run_updated_pco2_trajectory,
    trajectory_pco2_ppm,
)


INITIAL = UpdatedForwardInput(1.0, 285.5, 290.0)


def test_trajectory_interpolation_endpoints_and_shape() -> None:
    common = {
        "start_pco2_ppm": 285.5,
        "final_pco2_ppm": 422.8,
        "transition_duration_years": 174.0,
    }
    assert trajectory_pco2_ppm(-10.0, interpolation="linear", **common) == pytest.approx(285.5)
    assert trajectory_pco2_ppm(174.0, interpolation="linear", **common) == pytest.approx(422.8)
    linear_quarter = trajectory_pco2_ppm(43.5, interpolation="linear", **common)
    smooth_quarter = trajectory_pco2_ppm(43.5, interpolation="smoothstep", **common)
    assert smooth_quarter < linear_quarter


def test_gradual_trajectory_preserves_initial_isotope_state() -> None:
    result = run_updated_pco2_trajectory(
        UpdatedPCO2TrajectoryInput(
            initial=INITIAL,
            final_pco2_ppm=422.8,
            transition_duration_years=174.0,
            duration_years=200.0,
            sample_count=9,
            equilibrium_search_max_years=200.0,
        )
    )
    assert result.pco2_ppm[0] == pytest.approx(285.5)
    assert result.pco2_ppm[-1] == pytest.approx(422.8)
    assert result.states[0].cap_delta17_prime_permil == pytest.approx(
        result.initial_steady_state["cap_delta17_prime_permil"], abs=1.0e-12
    )
    assert result.transition_end_state["time_years"] == pytest.approx(174.0)
    assert result.transition_end_state["pco2_ppm"] == pytest.approx(422.8)


def test_transition_endpoint_is_evaluated_beyond_a_short_display_window() -> None:
    result = run_updated_pco2_trajectory(
        UpdatedPCO2TrajectoryInput(
            initial=INITIAL,
            final_pco2_ppm=422.8,
            transition_duration_years=174.0,
            duration_years=50.0,
            sample_count=9,
            equilibrium_search_max_years=200.0,
        )
    )
    assert result.time_years[-1] == pytest.approx(50.0)
    assert result.transition_end_state["time_years"] == pytest.approx(174.0)
    assert result.transition_end_state["cap_delta17_prime_permil"] != pytest.approx(
        result.states[-1].cap_delta17_prime_permil,
        abs=1.0e-8,
    )


def test_long_hold_converges_to_final_steady_state() -> None:
    result = run_updated_pco2_trajectory(
        UpdatedPCO2TrajectoryInput(
            initial=INITIAL,
            final_pco2_ppm=422.8,
            transition_duration_years=174.0,
            duration_years=50000.0,
            sample_count=101,
        )
    )
    final = result.states[-1]
    assert final.cap_delta17_prime_permil == pytest.approx(
        result.final_steady_state["cap_delta17_prime_permil"], abs=8.0e-4
    )
    assert final.delta18_prime_permil == pytest.approx(
        result.final_steady_state["delta18_prime_permil"], abs=2.0e-4
    )
    assert result.equilibrium_time_years is not None
    assert result.equilibrium_time_years > result.request.transition_duration_years
