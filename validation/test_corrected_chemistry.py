"""Reaction-accounting and solver safeguards independent of fitted outputs."""
import numpy as np
import pytest

from r1_r7_grid_chemistry import (
    modern_r1_r7_rate_fields, r5_o1d_n2_cm3_per_molecule_s,
    r5_o1d_o2_cm3_per_molecule_s, r5_o1d_co2_cm3_per_molecule_s,
)
from self_consistent_isotope_fixed_point import solve_mechanistic_fixed_point_with_hybr_fallback


@pytest.mark.parametrize("co2", [50e-6, 294e-6, .01, .06])
def test_co2_removal_occurs_once(co2):
    temperature = np.array([195., 220., 260., 300.])
    density = np.array([1e16, 1e17, 1e18, 1e19])
    fields = modern_r1_r7_rate_fields(temperature, density,
        o2_mixing_ratio=.212, n2_mixing_ratio=.7808, co2_mixing_ratio=co2)
    background = density*(.212*r5_o1d_o2_cm3_per_molecule_s(temperature)
        +.7808*r5_o1d_n2_cm3_per_molecule_s(temperature))
    co2_loss = density*co2*r5_o1d_co2_cm3_per_molecule_s(temperature)
    np.testing.assert_array_equal(fields["R5"].coefficient, background)
    np.testing.assert_allclose(fields["R5"].coefficient + density*co2*fields["R7"].coefficient,
        background+co2_loss, rtol=1e-15)


def test_inadmissible_newton_trial_restarts_without_changing_equations():
    def target(state):
        if np.any(state > 2):
            raise ValueError("fixed photochemical flux overwhelms an O2 source term")
        return state + 1-state**2
    result = solve_mechanistic_fixed_point_with_hybr_fallback(target, np.array([.01, .01]),
        finite_difference_step=1e-5, tolerance=1e-8, maximum_iterations=36)
    assert result.converged
    assert result.solver_method == "newton_inadmissible_trial_then_scipy_hybr"
    np.testing.assert_allclose(result.state, [1., 1.], atol=1e-8, rtol=0.)


def test_unrelated_validation_errors_are_not_swallowed():
    def invalid(state):
        raise ValueError("Invalid scientific input")
    with pytest.raises(ValueError, match="Invalid scientific input"):
        solve_mechanistic_fixed_point_with_hybr_fallback(invalid, np.array([1., 1.]))
