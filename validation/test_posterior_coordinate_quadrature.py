"""Convergence of the physical-coordinate integral, independent of rendering."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from posterior_coordinate_quadrature import (
    affine_gaussian_log_integral, integrated_coordinate_log_likelihood,
)
from updated_output_surface import load_updated_output_surface


@pytest.mark.parametrize("prior_sigma", (None, .17))
@pytest.mark.parametrize("width", (.003, .015, .2))
def test_affine_integral_resolves_a_peak_between_nodes(prior_sigma, width):
    lo, hi, target = .3, 1.2, .71234
    def f(x):
        return np.exp(-.5*((x-target)/width)**2
                      - (0 if prior_sigma is None else .5*((x-.8)/prior_sigma)**2))
    expected = quad(f, lo, hi, points=[target], epsabs=1e-14)[0]
    actual = affine_gaussian_log_integral(
        np.array([lo]), np.array([hi]), np.array([[(lo-target)/width]]),
        np.array([[(hi-target)/width]]),
        prior_mean=None if prior_sigma is None else .8, prior_sigma=prior_sigma,
    )
    assert np.exp(actual[0]) == pytest.approx(expected, rel=1e-10)


def test_multiple_gaussian_observations_and_flat_response():
    left, right = np.array([0., 0.]), np.array([1., 1.])
    fleft, fright = np.array([[-3., 2.], [.3, .4]]), np.array([[4., -1.], [.3, .4]])
    actual = np.exp(affine_gaussian_log_integral(left, right, fleft, fright))
    for i in range(2):
        expected = quad(lambda x: np.exp(-.5*np.sum((fleft[i]+x*(fright[i]-fleft[i]))**2)), 0, 1)[0]
        assert actual[i] == pytest.approx(expected, rel=1e-12)


def _request(target=-10., sigma=.015, prior="normal"):
    return SimpleNamespace(
        po2_prior=prior, po2_prior_mean=1., po2_prior_sigma=.1,
        measurement_sigma_permil=sigma, model_discrepancy_sigma_permil=0.,
        target_air_cap_delta17_permil=target,
        target_air_delta18_conventional_permil=23.9,
        delta18_measurement_sigma_permil=.3,
    )


@pytest.mark.parametrize("prior", ("normal", "uniform"))
def test_high_co2_slice_matches_dense_independent_trapezoidal_integral(prior):
    surface = load_updated_output_surface()
    request = _request(prior=prior)
    x = np.linspace(29000., 37000., 61)
    z = np.linspace(.6, 1.4, 17)
    logs, diagnostics = integrated_coordinate_log_likelihood(
        surface, request, {"pO2":z[None,:], "pCO2":x[:,None], "GPP":290.}, z, "pO2")
    weights = np.r_[np.diff(z)[0]/2, (z[2:]-z[:-2])/2, np.diff(z)[-1]/2]
    actual = np.exp(logs) @ weights
    dense_z = np.linspace(.6,1.4,2049)
    fields = dict(p_o2_pal=dense_z[None,:],p_co2_ppm=x[:,None],gpp_pgC_per_year=290.)
    d17 = surface.evaluate_central_cap_delta17_grid(**fields)
    d18 = 1000*np.expm1(surface.evaluate_central_delta18_prime_grid(**fields)/1000)
    likelihood = np.exp(-.5*((d17+10)/.015)**2-.5*((d18-23.9)/.3)**2
                        - (0 if prior == "uniform" else .5*((dense_z[None,:]-1)/.1)**2))
    expected = np.trapz(likelihood, dense_z, axis=1)
    assert np.sum(np.abs(actual-expected))/np.sum(expected) < .0005
    assert diagnostics["status"] == "converged"
    assert diagnostics["maximum_final_response_error_sigma"] <= .01


def test_separate_nonmonotone_branches_are_both_integrated():
    class Surface:
        def evaluate_central_cap_delta17_grid(self, **kwargs):
            z = kwargs["p_o2_pal"]
            return (z-.8)*(z-1.2)

    request = _request(target=0.,sigma=.0002,prior="uniform")
    request.target_air_delta18_conventional_permil = None
    z = np.linspace(.6,1.4,17)
    logs, _ = integrated_coordinate_log_likelihood(
        Surface(),request,{"pO2":z,"pCO2":300.,"GPP":290.},z,"pO2")
    weights = np.r_[.025,np.full(15,.05),.025]
    mass = np.exp(logs)*weights
    mass /= mass.sum()
    assert mass[z<1].sum() == pytest.approx(.5, abs=1e-6)
    assert mass[z>1].sum() == pytest.approx(.5, abs=1e-6)
    assert mass[(z>.9)&(z<1.1)].sum() < 1e-6


def test_extreme_tail_integral_does_not_underflow_in_log_space():
    logs = affine_gaussian_log_integral(np.array([0.]),np.array([1.]),
                                        np.array([[90.]]),np.array([[91.]]))
    assert np.isfinite(logs[0])
    assert logs[0] < -4000


def test_modern_air_and_gpp_nuisance_range_match_dense_reference():
    surface = load_updated_output_surface()
    request = _request(target=-.432,sigma=.015)
    request.gpp_prior = "uniform"
    x = np.linspace(180.,500.,41)
    gpp = np.linspace(232.,348.,17)
    logs, diagnostics = integrated_coordinate_log_likelihood(
        surface,request,{"pO2":1.,"pCO2":x[:,None],"GPP":gpp[None,:]},gpp,"GPP")
    actual = np.trapz(np.exp(logs),gpp,axis=1)
    dense = np.linspace(232.,348.,1025)
    fields = dict(p_o2_pal=1.,p_co2_ppm=x[:,None],gpp_pgC_per_year=dense[None,:])
    d17 = surface.evaluate_central_cap_delta17_grid(**fields)
    d18 = 1000*np.expm1(surface.evaluate_central_delta18_prime_grid(**fields)/1000)
    expected = np.trapz(np.exp(-.5*((d17+.432)/.015)**2-.5*((d18-23.9)/.3)**2),dense,axis=1)
    assert np.sum(abs(actual-expected))/np.sum(expected) < .0005
    assert diagnostics["coordinate"] == "GPP"
