"""Numerical-resolution gates for public constrained-coordinate inference."""

from __future__ import annotations

import numpy as np
import pytest

from updated_constrained_pco2_posterior import (
    ConstrainedCoordinateInput,
    ConstrainedCoordinateResult,
    CoordinateConstraint,
    constrained_coordinate_posterior,
)
from updated_output_surface import load_updated_output_surface


STATES = (
    ("modern", 294.0, 290.0, 1.0),
    ("low_co2", 100.0, 290.0, 1.0),
    ("low_gpp_low_o2", 10_000.0, 72.5, 0.5),
    ("high_co2", 30_000.0, 145.0, 1.0),
    ("high_o2", 1_000.0, 435.0, 1.8),
)


def _target(pco2: float, gpp: float, po2: float) -> float:
    surface = load_updated_output_surface()
    return float(
        surface.evaluate_central_cap_delta17_grid(
            p_o2_pal=po2,
            p_co2_ppm=pco2,
            gpp_pgC_per_year=gpp,
        )
    )


def _solve(
    solve_for: str,
    pco2: float,
    gpp: float,
    po2: float,
    *,
    sigma: float = 0.003,
    pco2_grid_size: int = 181,
) -> ConstrainedCoordinateResult:
    values = {"pCO2": pco2, "GPP": gpp, "pO2": po2}
    constraints = {
        coordinate: CoordinateConstraint("fixed", center=value)
        for coordinate, value in values.items()
        if coordinate != solve_for
    }
    return constrained_coordinate_posterior(
        ConstrainedCoordinateInput(
            solve_for=solve_for,
            target_air_cap_delta17_permil=_target(pco2, gpp, po2),
            measurement_sigma_permil=sigma,
            constraints=constraints,
            pco2_grid_size=pco2_grid_size,
            gpp_grid_size=81,
            po2_grid_size=17,
        )
    )


@pytest.mark.parametrize(
    "name,pco2,gpp,po2", STATES, ids=[state[0] for state in STATES]
)
@pytest.mark.parametrize("solve_for", ("pCO2", "GPP", "pO2"))
def test_synthetic_state_is_recovered_across_the_operational_domain(
    name: str, pco2: float, gpp: float, po2: float, solve_for: str
) -> None:
    del name
    truth = {"pCO2": pco2, "GPP": gpp, "pO2": po2}[solve_for]
    result = _solve(solve_for, pco2, gpp, po2)
    lower, upper = result.equal_tailed_credible_interval

    assert np.sum(result.solve_marginal_probability_mass) == pytest.approx(1.0)
    assert lower <= truth <= upper
    assert lower <= result.posterior_median <= upper
    assert abs(result.posterior_median - truth) / truth < 0.08
    assert result.final_solve_axis_size == result.initial_solve_axis_size


def test_high_co2_refinement_agrees_with_dense_full_domain_quadrature() -> None:
    refined = _solve("pCO2", 30_000.0, 145.0, 1.0)
    dense = _solve(
        "pCO2",
        30_000.0,
        145.0,
        1.0,
        pco2_grid_size=20_001,
    )

    assert refined.numerical_refinement_applied is True
    assert refined.initial_solve_bounds == (50.0, 60_000.0)
    assert refined.final_solve_bounds[0] > refined.initial_solve_bounds[0]
    assert refined.final_solve_bounds[1] < refined.initial_solve_bounds[1]
    assert refined.posterior_median == pytest.approx(
        dense.posterior_median, abs=10.0
    )
    assert refined.equal_tailed_credible_interval == pytest.approx(
        dense.equal_tailed_credible_interval, abs=12.0
    )


def test_true_domain_edge_remains_boundary_sensitive_and_is_not_refined() -> None:
    result = _solve("pCO2", 60_000.0, 145.0, 1.0, sigma=0.015)

    assert result.status == "solve_boundary_sensitive"
    assert result.solve_boundary_sensitive is True
    assert result.numerical_refinement_applied is False
    assert result.initial_solve_bounds == result.final_solve_bounds == (
        50.0,
        60_000.0,
    )
