"""Independent quadrature and covariance tests for exact sulfate uncertainty."""

from dataclasses import replace
from math import exp

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, truncnorm

from sulfate_to_air import (
    IncorporationConstraint, SulfateProcessAssumptions, exact_sulfate_from_air,
    exact_sulfate_grid,
)
from sulfate_uncertainty import (
    BackgroundConstraint, SulfateLikelihoodInput, SulfateIntegrationSettings,
    exact_sulfate_likelihood,
)


def request(**changes):
    base = SulfateLikelihoodInput(
        measured_cap_delta17_permil=-2.0, measured_delta18_permil=15.0,
        cap_delta17_sigma_permil=0.03, delta18_sigma_permil=0.0,
        isotope_error_correlation=0.0, incorporation=IncorporationConstraint("fixed",center=0.2),
        background=BackgroundConstraint("fixed",center=-0.02),
        alpha18_air_to_sulfate=1.0, theta_air_to_sulfate=0.528,
        assumption_note="Synthetic uncertainty test; primary preservation, specified B, no air-path fractionation.",
    )
    return replace(base, **changes)


def scalar(air, air18, sulfate18, fraction, background, r):
    return exact_sulfate_from_air(
        air_cap_delta17_permil=air, air_delta18_permil=air18,
        sulfate_delta18_permil=sulfate18, fraction=fraction,
        process=SulfateProcessAssumptions(background,r.alpha18_air_to_sulfate,
                                          r.theta_air_to_sulfate,r.assumption_note),
    )


def test_batched_roots_match_independent_scalar_solver():
    rng = np.random.default_rng(7161)
    air = rng.uniform(-45.0,0.2,125)
    air18 = rng.uniform(20.0,27.0,125)
    sulfate18 = rng.uniform(-15.0,25.0,125)
    fraction = rng.uniform(0.01,0.8,125)
    background = rng.uniform(-0.15,0.05,125)
    r = request(alpha18_air_to_sulfate=0.985,theta_air_to_sulfate=0.517)
    actual, valid = exact_sulfate_grid(
        air_cap_delta17_permil=air, air_delta18_permil=air18, sulfate_delta18_permil=sulfate18,
        fraction=fraction, background_cap_delta17_permil=background,
        alpha18_air_to_sulfate=r.alpha18_air_to_sulfate,theta_air_to_sulfate=r.theta_air_to_sulfate)
    expected = [scalar(a,d,s,f,b,r) for a,d,s,f,b in zip(air,air18,sulfate18,fraction,background)]
    assert np.all(valid)
    assert actual == pytest.approx(expected,abs=2e-8)


def test_fixed_process_likelihood_is_exact_forward_gaussian():
    r = request()
    air = np.array([-11.0,-10.0,-9.0])
    result = exact_sulfate_likelihood(air,23.9,r)
    predicted = [scalar(a,23.9,15.0,0.2,-0.02,r) for a in air]
    assert result.log_likelihood == pytest.approx(norm.logpdf(-2.0,loc=predicted,scale=0.03),abs=1e-8)
    assert result.diagnostics["air_prior"] is None


@pytest.mark.parametrize("constraint", [IncorporationConstraint("range",lower=0.1,upper=0.25),
                                       IncorporationConstraint("normal",center=0.2,sigma=0.03)])
@pytest.mark.parametrize("air", [-15.0,-10.0,-8.5])
def test_narrow_measurement_fraction_integral_matches_adaptive_scalar_quadrature(constraint,air):
    r = request(incorporation=constraint,cap_delta17_sigma_permil=0.003)
    if constraint.kind == "range":
        low, high = constraint.lower,constraint.upper
        def prior(f):
            return 1.0/(high-low)
    else:
        low, high = 0.0,1.0
        def prior(f):
            return truncnorm.pdf(f,-constraint.center/constraint.sigma,(1.0-constraint.center)/constraint.sigma,
                                 loc=constraint.center,scale=constraint.sigma)
    # Divide the integration interval into many independent subintervals so a
    # narrow likelihood cannot be missed by the reference adaptive integrator.
    grid = np.linspace(low,min(high,0.95),101)
    expected = sum(quad(lambda f: norm.pdf(-2.0,scalar(air,23.9,15.0,f,-0.02,r),0.003)*prior(f),
                        a,b,epsabs=1e-12,epsrel=1e-9)[0] for a,b in zip(grid[:-1],grid[1:]))
    actual = exact_sulfate_likelihood(air,23.9,r)
    assert exp(float(actual.log_likelihood)) == pytest.approx(expected,rel=7e-4,abs=1e-9)


@pytest.mark.parametrize("background",[BackgroundConstraint("range",lower=-0.1,upper=0.1),
                                       BackgroundConstraint("normal",center=-0.02,sigma=0.04)])
def test_background_constraint_is_integrated_in_measurement_space(background):
    r = request(background=background,cap_delta17_sigma_permil=0.005)
    if background.kind == "range":
        low,high = background.lower,background.upper
        def prior(b):
            return 1.0/(high-low)
    else:
        low,high = background.center-8*background.sigma,background.center+8*background.sigma
        def prior(b):
            return norm.pdf(b,background.center,background.sigma)
    expected = quad(lambda b: norm.pdf(-2.0,scalar(-10.0,23.9,15.0,0.2,b,r),0.005)*prior(b),
                    low,high,epsabs=1e-12,epsrel=1e-9)[0]
    actual = exact_sulfate_likelihood(-10.0,23.9,r)
    assert exp(float(actual.log_likelihood)) == pytest.approx(expected,rel=5e-4)


@pytest.mark.parametrize("correlation",[0.0,0.7,-0.7])
def test_delta18_measurement_and_covariance_match_independent_integral(correlation):
    r = request(delta18_sigma_permil=0.5,isotope_error_correlation=correlation)
    conditional_sigma = 0.03*np.sqrt(1.0-correlation**2)
    expected = quad(lambda z: norm.pdf(z)*norm.pdf(-2.0+correlation*0.03*z,
                    scalar(-10.0,23.9,15.0+0.5*z,0.2,-0.02,r),conditional_sigma),
                    -8.0,8.0,epsabs=1e-12,epsrel=1e-10)[0]
    actual = exact_sulfate_likelihood(-10.0,23.9,r)
    assert exp(float(actual.log_likelihood)) == pytest.approx(expected,rel=3e-4)


def test_zero_air_inheritance_contains_no_air_information_even_with_covariance():
    r = request(incorporation=IncorporationConstraint("fixed",center=0.0),
                measured_cap_delta17_permil=-0.05,delta18_sigma_permil=0.5,isotope_error_correlation=0.5,
                background=BackgroundConstraint("normal",center=-0.02,sigma=0.02))
    actual = exact_sulfate_likelihood(np.array([-40.0,-10.0,0.0]),np.array([22.0,24.0,27.0]),r)
    expected = norm.logpdf(-0.05,-0.02,np.hypot(0.03,0.02))
    assert actual.log_likelihood == pytest.approx(np.full(3,expected),abs=1e-5)


def test_refinement_limit_is_an_error_not_an_accepted_result():
    with pytest.raises(RuntimeError,match="did not converge"):
        exact_sulfate_likelihood(-10.0,23.9,request(),settings=SulfateIntegrationSettings(max_level=2))


@pytest.mark.parametrize("changes",[{"cap_delta17_sigma_permil":0.0},{"delta18_sigma_permil":-1.0},
                                    {"isotope_error_correlation":1.0},{"assumption_note":""},
                                    {"incorporation":IncorporationConstraint("fixed",center=1.0)}])
def test_invalid_request_is_rejected(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_combined_uncertainties_match_dense_independent_tensor_quadrature():
    from scipy.special import roots_legendre, roots_hermitenorm
    r = request(incorporation=IncorporationConstraint("range",lower=0.1,upper=0.25),
                background=BackgroundConstraint("normal",center=-0.02,sigma=0.03),delta18_sigma_permil=0.5)
    f, fw = roots_legendre(401)
    fractions, fw = 0.1+0.15*(f+1.0)/2.0, fw/2.0
    b,bw = roots_hermitenorm(21)
    z,zw = roots_hermitenorm(11)
    bw,zw = bw/np.sqrt(2.0*np.pi),zw/np.sqrt(2.0*np.pi)
    expected = 0.0
    for iz,iw in zip(z,zw):
        values, physical = exact_sulfate_grid(
            air_cap_delta17_permil=-10.0,air_delta18_permil=23.9,
            sulfate_delta18_permil=15.0+0.5*iz, fraction=fractions[:,None],
            background_cap_delta17_permil=-0.02+0.03*b[None,:],
            alpha18_air_to_sulfate=1.0,theta_air_to_sulfate=0.528)
        assert np.all(physical)
        expected += iw*float(np.sum(norm.pdf(-2.0,values,0.03)*fw[:,None]*bw[None,:]))
    actual = exact_sulfate_likelihood(-10.0,23.9,r)
    assert exp(float(actual.log_likelihood)) == pytest.approx(expected,rel=3e-4)


def test_no_support_is_zero_without_renormalizing_away_process_conflict():
    r = request(measured_delta18_permil=-900.0,
                incorporation=IncorporationConstraint("range",lower=0.5,upper=0.8))
    out = exact_sulfate_likelihood(np.array([-40.0,-10.0,-0.432]),23.9,r)
    assert np.all(np.isneginf(out.log_likelihood))


def test_zero_endpoint_is_integrable_and_not_clipped_to_a_positive_prior_floor():
    r = request(incorporation=IncorporationConstraint("range",lower=0.0,upper=0.3),
                measured_cap_delta17_permil=-0.02)
    out = exact_sulfate_likelihood(np.array([-10.0,-1.0,0.0]),23.9,r)
    assert np.all(np.isfinite(out.log_likelihood))
    assert out.diagnostics["observation"]["incorporation"]["lower"] == 0.0


def test_chunk_size_does_not_change_probability():
    r = request(incorporation=IncorporationConstraint("range",lower=0.1,upper=0.25),
                background=BackgroundConstraint("normal",center=-0.02,sigma=0.03))
    air = np.linspace(-23,-7,9)
    a = exact_sulfate_likelihood(air,23.9,r,settings=SulfateIntegrationSettings(chunk_size=1))
    b = exact_sulfate_likelihood(air,23.9,r,settings=SulfateIntegrationSettings(chunk_size=8))
    assert a.log_likelihood == pytest.approx(b.log_likelihood,abs=2e-8)


def test_browser_batch_size_preserves_exact_likelihood_and_convergence():
    r = request(measured_cap_delta17_permil=-0.2, measured_delta18_permil=15.0,
                cap_delta17_sigma_permil=0.03, delta18_sigma_permil=0.5,
                incorporation=IncorporationConstraint("range", lower=0.18, upper=0.29),
                background=BackgroundConstraint("fixed", center=0.0),
                alpha18_air_to_sulfate=1.0, theta_air_to_sulfate=0.528)
    air = np.linspace(-2.0, 0.1, 1100)
    d18 = np.linspace(23.4, 24.0, air.size)
    old = exact_sulfate_likelihood(air, d18, r,
                                 settings=SulfateIntegrationSettings(chunk_size=128))
    current = exact_sulfate_likelihood(air, d18, r)
    np.testing.assert_allclose(current.log_likelihood, old.log_likelihood, rtol=0, atol=1e-12)
    assert current.diagnostics["status"] == old.diagnostics["status"] == "converged"


def normal_nonair_browser_observation():
    return request(
        measured_cap_delta17_permil=-0.2, delta18_sigma_permil=0.5,
        incorporation=IncorporationConstraint("range", lower=0.2, upper=0.29),
        background=BackgroundConstraint("normal", center=0.1, sigma=0.05),
        alpha17_air_to_sulfate=1.0, theta_air_to_sulfate=None,
        fractionation_treatment="none",
    )


def test_reversed_integrals_match_independent_dense_measurement_quadrature():
    from scipy.special import roots_legendre, roots_hermitenorm
    r = normal_nonair_browser_observation()
    # Include the failed shoulder, likelihood tails, and both sides of the
    # spread-based integration-order switch (near air anomaly -1.353).
    air = np.array([-2.74, -2.07, -1.74, -1.57, -1.36, -1.35, -1.0, -0.43])

    def reference(order):
        f, fw = roots_legendre(order)
        b, bw = roots_hermitenorm(97)
        z, zw = roots_hermitenorm(15)
        weights = fw[:, None]/2.0*bw[None, :]/np.sqrt(2.0*np.pi)
        expected = np.zeros_like(air)
        for iz, iw in zip(z, zw/np.sqrt(2.0*np.pi)):
            values, physical = exact_sulfate_grid(
                air_cap_delta17_permil=air[:, None, None], air_delta18_permil=23.9,
                sulfate_delta18_permil=15.0+0.5*iz,
                fraction=(0.2+0.09*(f+1.0)/2.0)[None, :, None],
                background_cap_delta17_permil=(0.1+0.05*b)[None, None, :],
                alpha18_air_to_sulfate=1.0, theta_air_to_sulfate=None,
                alpha17_air_to_sulfate=1.0)
            assert np.all(physical)
            expected += iw*np.sum(norm.pdf(-0.2, values, 0.03)*weights, axis=(1, 2))
        return expected

    expected = reference(201)
    np.testing.assert_allclose(expected, reference(301), rtol=1e-7, atol=1e-11)
    out = exact_sulfate_likelihood(air, 23.9, r)
    assert out.diagnostics["status"] == "converged"
    assert out.diagnostics["maximum_level"] <= 5
    np.testing.assert_allclose(np.exp(out.log_likelihood), expected, rtol=2e-4, atol=2e-9)
    chunked = exact_sulfate_likelihood(air, 23.9, r,
                                     settings=SulfateIntegrationSettings(chunk_size=1))
    np.testing.assert_array_equal(chunked.log_likelihood, out.log_likelihood)


def test_alternate_order_still_requires_two_fresh_checks():
    from sulfate_uncertainty import SulfateIntegrationError
    with pytest.raises(SulfateIntegrationError, match="did not converge"):
        exact_sulfate_likelihood(-1.74, 23.9, normal_nonair_browser_observation(),
                                settings=SulfateIntegrationSettings(max_level=2))


def test_alternate_order_shares_the_original_work_budget(monkeypatch):
    import sulfate_uncertainty as module
    r = normal_nonair_browser_observation()
    air, d18 = np.array([-1.74]), np.array([23.9])
    original = module._evaluate_level
    def unfinished_first_order(a, d, request, level, inner, **kwargs):
        value, error, roots = original(a, d, request, level, inner, **kwargs)
        return value, np.full_like(error, 1.) if not inner else error, roots
    monkeypatch.setattr(module, "_evaluate_level", unfinished_first_order)
    first_order_roots = sum(original(air, d18, r, level, False)[2]
                            for level in range(1, 6))
    with pytest.raises(module.SulfateComputeLimitError):
        with module.sulfate_computation_budget(max_roots=first_order_roots):
            exact_sulfate_likelihood(air, d18, r)
    assert module._COMPUTE_BUDGET.get() is None


def test_partial_physical_support_retains_original_fraction_prior_mass():
    r = request(measured_delta18_permil=-500.0,cap_delta17_sigma_permil=3.0,
                incorporation=IncorporationConstraint("range",lower=0.0,upper=0.5))
    observed = scalar(-10.0,23.9,-500.0,0.2,-0.02,r)
    r = replace(r,measured_cap_delta17_permil=observed)
    # The weighted cell-error bound is stricter than the old response check
    # for this deliberately extreme -500 per mil delta18O support test.
    settings = SulfateIntegrationSettings(max_level=8)
    a = exact_sulfate_likelihood(-10.0,23.9,r,settings=settings)
    b = exact_sulfate_likelihood(-10.0,23.9,replace(r,
        incorporation=IncorporationConstraint("range",lower=0.0,upper=0.8)),settings=settings)
    # Both intervals extend beyond the positive-atom support near f=0.489.
    # Additional impossible prior mass must reduce evidence, not disappear.
    assert exp(float(b.log_likelihood-a.log_likelihood)) == pytest.approx(0.5/0.8,rel=1e-5)


def test_extreme_curvature_and_narrow_error_are_not_accepted_without_convergence():
    r = request(measured_delta18_permil=-500.0,
                incorporation=IncorporationConstraint("range",lower=0.0,upper=0.5))
    r = replace(r,measured_cap_delta17_permil=scalar(-10.0,23.9,-500.0,0.2,-0.02,r))
    with pytest.raises(RuntimeError,match="did not converge"):
        exact_sulfate_likelihood(-10.0,23.9,r,settings=SulfateIntegrationSettings(max_level=7))


def test_tighter_integration_tolerances_preserve_combined_likelihood():
    r = request(incorporation=IncorporationConstraint("range",lower=0.1,upper=0.25),
                background=BackgroundConstraint("normal",center=-0.02,sigma=0.03),
                delta18_sigma_permil=0.5,isotope_error_correlation=0.2,
                alpha18_air_to_sulfate=0.985,theta_air_to_sulfate=0.517)
    air = np.array([-13.0,-10.0,-9.0])
    a = exact_sulfate_likelihood(air,23.9,r)
    b = exact_sulfate_likelihood(air,23.9,r,settings=SulfateIntegrationSettings(
        relative_tolerance=5e-5,response_tolerance_sigma=5e-4,max_level=6))
    assert np.exp(a.log_likelihood) == pytest.approx(np.exp(b.log_likelihood),rel=5e-4,abs=1e-9)


def synthetic_observation():
    from public_model_service import forward
    from updated_molecular_forward_model import UpdatedForwardInput
    from isotopes import conventional_delta_from_prime
    model = forward(UpdatedForwardInput(p_o2_pal=0.5,p_co2_ppm=10000.0,gpp_pgC_per_year=72.5))["result"]
    air, d18 = model["central_cap_delta17_prime_permil"],conventional_delta_from_prime(model["central_delta18_prime_permil"])
    r = request(delta18_sigma_permil=0.5)
    return replace(r,measured_cap_delta17_permil=scalar(air,d18,15.0,0.2,-0.02,r))


@pytest.mark.parametrize("coordinate,truth,bounds", [("pCO2",10000.0,(5000.0,15000.0)),
                                                   ("GPP",72.5,(40.0,110.0)),("pO2",0.5,(0.25,0.75))])
def test_sulfate_likelihood_drives_each_solved_coordinate(coordinate,truth,bounds):
    from updated_output_surface_joint_posterior import UpdatedJointPosteriorInput, joint_updated_posterior
    bounds_key = {"pCO2":"pco2_bounds_ppm","GPP":"gpp_bounds_pgC_per_year","pO2":"po2_bounds_pal"}[coordinate]
    result = joint_updated_posterior(UpdatedJointPosteriorInput(
        sulfate=synthetic_observation(),free_coordinates=(coordinate,),p_o2_pal=0.5,
        p_co2_ppm=10000.0,gpp_pgC_per_year=72.5,pco2_prior="uniform",gpp_prior="uniform",
        pco2_grid_size=81,gpp_grid_size=81,po2_grid_size=81,**{bounds_key:bounds}))
    low,high = result.equal_tailed_credible_intervals[coordinate]
    assert low < truth < high
    assert result.effective_likelihood_sigma_permil is None
    assert result.sulfate_likelihood_diagnostics["status"] == "converged"
    assert result.inputs.target_air_cap_delta17_permil is None
    assert result.posterior_integral == pytest.approx(1.0)


def test_incorporation_uncertainty_broadens_synthetic_pco2_posterior():
    from updated_output_surface_joint_posterior import UpdatedJointPosteriorInput, joint_updated_posterior
    obs = synthetic_observation()
    base = UpdatedJointPosteriorInput(sulfate=obs,free_coordinates=("pCO2",),p_o2_pal=0.5,
                                     gpp_pgC_per_year=72.5,pco2_bounds_ppm=(3000.,20000.),
                                     pco2_prior="uniform",pco2_grid_size=121)
    fixed = joint_updated_posterior(base)
    uncertain = joint_updated_posterior(replace(base,sulfate=replace(obs,
        incorporation=IncorporationConstraint("range",lower=0.15,upper=0.25))))
    f = fixed.equal_tailed_credible_intervals["pCO2"]
    u = uncertain.equal_tailed_credible_intervals["pCO2"]
    assert u[1]-u[0] > 2.0*(f[1]-f[0])
    assert u[0] < 10000.0 < u[1]


def test_solver_rejects_fabricated_air_target_or_air_error_added_to_sulfate():
    from updated_output_surface_joint_posterior import UpdatedJointPosteriorInput, joint_updated_posterior
    r = UpdatedJointPosteriorInput(sulfate=synthetic_observation())
    with pytest.raises(ValueError,match="mutually exclusive"):
        joint_updated_posterior(replace(r,target_air_cap_delta17_permil=-10.0,measurement_sigma_permil=0.1))
    with pytest.raises(ValueError,match="silently added"):
        joint_updated_posterior(replace(r,model_discrepancy_sigma_permil=0.03,model_discrepancy_source="test"))


def test_constrained_solver_retains_sulfate_likelihood_during_refinement():
    from updated_constrained_pco2_posterior import (
        ConstrainedCoordinateInput,CoordinateConstraint,constrained_coordinate_posterior)
    result = constrained_coordinate_posterior(ConstrainedCoordinateInput(
        solve_for="pCO2",target_air_cap_delta17_permil=None,measurement_sigma_permil=None,
        constraints={"GPP":CoordinateConstraint("fixed",center=72.5),"pO2":CoordinateConstraint("fixed",center=0.5)},
        sulfate=synthetic_observation(),pco2_grid_size=81,enforce_public_resolution=False))
    assert result.equal_tailed_credible_interval[0] < 10000.0 < result.equal_tailed_credible_interval[1]
    assert result.sulfate_likelihood_diagnostics["status"] == "converged"
    assert "exact sulfate transfer" in result.probability_scope


def test_sulfate_infer_console_request_round_trip(tmp_path,capsys):
    from dataclasses import asdict
    import json
    from public_cli import build_parser,_run
    path = tmp_path / "request.json"
    payload = {"sulfate":asdict(synthetic_observation()),
               "atmosphere":{"free_coordinates":["pCO2"],"p_o2_pal":0.5,"gpp_pgC_per_year":72.5,
                             "pco2_bounds_ppm":[5000.,15000.],"pco2_grid_size":81}}
    path.write_text(json.dumps(payload),encoding="utf-8")
    _run(build_parser().parse_args(["sulfate-infer","--request",str(path)]))
    result = json.loads(capsys.readouterr().out)
    assert result["inputs"]["pco2_prior"] == "uniform"
    assert result["sulfate_likelihood_diagnostics"]["status"] == "converged"
