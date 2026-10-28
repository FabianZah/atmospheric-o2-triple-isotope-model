"""Analytic tests distinguish nodal quadrature weights from interval masses."""
import numpy as np
import pytest
from scipy.special import ndtr, ndtri
from updated_output_surface_joint_posterior import _quantile, _trapezoid_weights


@pytest.mark.parametrize("axis", [np.linspace(50.,60000.,9), np.geomspace(50.,60000.,9)])
@pytest.mark.parametrize("q", [0.,.025,.25,.5,.975,1.])
def test_uniform_density_has_exact_quantiles_on_any_spacing(axis, q):
    mass = _trapezoid_weights(axis)/(axis[-1]-axis[0])
    assert _quantile(axis,mass,q) == pytest.approx(axis[0]+q*(axis[-1]-axis[0]), abs=2e-10)


@pytest.mark.parametrize("increasing", [True, False])
@pytest.mark.parametrize("q", [.025,.5,.975])
def test_triangular_density_integrates_exactly_between_nodes(increasing, q):
    axis = np.array([0.,.01,.13,.7,.99,1.])
    density = 2*axis if increasing else 2*(1-axis)
    expected = np.sqrt(q) if increasing else 1-np.sqrt(1-q)
    assert _quantile(axis,density*_trapezoid_weights(axis),q) == pytest.approx(expected,abs=2e-14)


def test_zero_density_gap_and_total_mass_scaling():
    axis = np.array([0.,1.,2.,3.,4.,5.])
    density = np.array([1.,1.,0.,0.,1.,1.])
    mass = density*_trapezoid_weights(axis)
    assert _quantile(axis,mass,.5) == pytest.approx(2.)
    for q in [.025,.4,.6,.975]:
        assert _quantile(axis,mass,q) == pytest.approx(_quantile(axis,7.*mass,q))


@pytest.mark.parametrize("q", [.025,.5,.975])
def test_truncated_gaussian_converges_without_half_cell_shift(q):
    low,high,mean,sigma = .1,2.,.7,.25
    exact = mean + sigma*ndtri(ndtr((low-mean)/sigma)+q*(ndtr((high-mean)/sigma)-ndtr((low-mean)/sigma)))
    errors = []
    for count in [81,161,321]:
        axis = np.geomspace(low,high,count)
        density = np.exp(-.5*((axis-mean)/sigma)**2)
        errors.append(abs(_quantile(axis,density*_trapezoid_weights(axis),q)-exact))
    assert errors[-1] < 2e-4
    assert errors[-1] < errors[0]/8


@pytest.mark.parametrize("mass", [[0.,0.],[-1.,2.],[np.nan,1.]])
def test_invalid_density_rejected(mass):
    with pytest.raises(ValueError):
        _quantile(np.array([0.,1.]),np.array(mass),.5)


def test_sulfate_summary_grid_convergence_after_quantile_correction():
    from dataclasses import replace
    from test_sulfate_uncertainty import normal_nonair_browser_observation
    from updated_constrained_pco2_posterior import ConstrainedCoordinateInput, CoordinateConstraint as C, constrained_coordinate_posterior
    request = ConstrainedCoordinateInput(solve_for="pCO2",target_air_cap_delta17_permil=None,
        measurement_sigma_permil=None,sulfate=normal_nonair_browser_observation(),
        constraints={"GPP":C("fixed",center=290.),"pO2":C("fixed",center=1.)})
    summaries = []
    for count in [181,361,721,1441]:
        result = constrained_coordinate_posterior(replace(request,pco2_grid_size=count))
        summaries.append(np.array([result.equal_tailed_credible_interval[0], result.posterior_median,
                                   result.equal_tailed_credible_interval[1]]))
    changes = [np.abs(b-a) for a,b in zip(summaries[:-1],summaries[1:])]
    # Piecewise-linear integration converges at second order instead of a half-cell offset.
    assert np.all(changes[1] < .35*changes[0])
    assert np.all(changes[2] < .35*changes[1])
    assert abs(summaries[0][1]-summaries[-1][1]) < 1.
    assert np.max(np.abs(summaries[0]-summaries[-1])) < 5.
