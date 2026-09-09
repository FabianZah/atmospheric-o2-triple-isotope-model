"""Accuracy, support and request isolation for isotope-space likelihood reuse."""

from dataclasses import replace
import numpy as np
import pytest

from sulfate_likelihood_table import scalable_sulfate_likelihood, _unique_nodes
from sulfate_uncertainty import (
    SulfateIntegrationSettings,
    SulfateComputeLimitError,
    _COMPUTE_BUDGET,
    exact_sulfate_likelihood,
    sulfate_computation_budget,
)
from test_sulfate_uncertainty import normal_nonair_browser_observation


@pytest.mark.parametrize(
    "center,sigma", [(0.0, 0.15), (0.25, 0.15), (1.0, 0.4), (0.5, 0.005)]
)
def test_physical_fraction_quadrature_preserves_truncated_normal_mass(center, sigma):
    from sulfate_uncertainty import (
        IncorporationConstraint,
        _nodes,
        _outer_fraction_rule,
    )
    from scipy.stats import truncnorm

    constraint = IncorporationConstraint("normal", center=center, sigma=sigma)
    nodes, weights = _nodes(constraint, 65, fraction=True)
    mean = truncnorm.mean(
        -center / sigma, (1 - center) / sigma, loc=center, scale=sigma
    )
    assert weights.sum() == pytest.approx(1.0, abs=2e-12)
    assert np.sum(weights * nodes) == pytest.approx(mean, abs=2e-12)
    r = replace(normal_nonair_browser_observation(), incorporation=constraint)
    nodes, weights = _outer_fraction_rule(np.array([-30.0, -2.0, -0.3]), r, 129)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        np.sum(weights * nodes, axis=1), mean, atol=1e-10, rtol=0
    )


def test_machine_resolved_atom_brackets_match_the_scalar_solver():
    from sulfate_uncertainty import _physical_fraction_limit
    from test_sulfate_uncertainty import scalar
    from sulfate_to_air import exact_sulfate_grid

    r = normal_nonair_browser_observation()
    a, d, s = -0.3, 26.0, 15.0
    limit = _physical_fraction_limit(a, d, s, r)
    f = np.array([limit - 1e-8, limit - 1e-10, limit - 1e-12])
    actual, physical = exact_sulfate_grid(
        air_cap_delta17_permil=a,
        air_delta18_permil=d,
        sulfate_delta18_permil=s,
        fraction=f,
        background_cap_delta17_permil=0.1,
        alpha18_air_to_sulfate=1.0,
        theta_air_to_sulfate=None,
        alpha17_air_to_sulfate=1.0,
    )
    assert np.all(physical)
    expected = [
        scalar(
            a,
            d,
            s,
            item,
            0.1,
            replace(r, theta_air_to_sulfate=0.528, alpha17_air_to_sulfate=None),
        )
        for item in f
    ]
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)


def test_roundoff_equivalent_seeds_cannot_create_spline_slivers():
    nodes = _unique_nodes([-4.0, -1.5, np.nextafter(-1.5, -2.0), -0.3, -0.3])
    np.testing.assert_array_equal(nodes, [-4.0, -1.5 - 2.220446049250313e-16, -0.3])
    assert np.all(np.diff(nodes) > 1e-10)


def test_fraction_panels_keep_original_prior_mass_at_physical_boundary():
    from scipy.special import ndtr
    from sulfate_uncertainty import IncorporationConstraint, _outer_fraction_rule

    r = replace(
        normal_nonair_browser_observation(),
        incorporation=IncorporationConstraint("normal", center=0.8, sigma=0.2),
    )
    upper = np.array([0.3, 0.7, 0.99])
    x, w = _outer_fraction_rule(np.array([-20.0, -3.0, -0.3]), r, 65, upper)
    mass = (ndtr((upper - 0.8) / 0.2) - ndtr(-4.0)) / (ndtr(1.0) - ndtr(-4.0))
    np.testing.assert_allclose(w.sum(axis=1), mass, atol=2e-12, rtol=0)
    assert np.all(x[w > 0] < np.broadcast_to(upper[:, None], x.shape)[w > 0])


def test_large_delta18_error_and_near_pure_air_tail_match_full_order_refinement():
    from sulfate_uncertainty import IncorporationConstraint, BackgroundConstraint
    r = replace(normal_nonair_browser_observation(), measured_cap_delta17_permil=-2.,
                cap_delta17_sigma_permil=.15, delta18_sigma_permil=10.,
                incorporation=IncorporationConstraint("normal", center=.2, sigma=.15),
                background=BackgroundConstraint("range", lower=-.25, upper=.25))
    a, d = np.array([-1.4, -.8, -.64695818]), np.array([20.08847845, 23.4535375, 23.4535375])
    settings = SulfateIntegrationSettings(relative_tolerance=5e-5, absolute_tolerance=2.5e-11,
                                          max_level=7, focused_fraction_quadrature=True)
    full = exact_sulfate_likelihood(a, d, r, settings=settings)
    adaptive = exact_sulfate_likelihood(a, d, r, settings=replace(settings, adaptive_delta18=True))
    scale = r.cap_delta17_sigma_permil*np.sqrt(2*np.pi)
    expected, actual = np.exp(full.log_likelihood)*scale, np.exp(adaptive.log_likelihood)*scale
    assert np.all(np.abs(actual-expected) <= 1e-10+2e-4*expected)
    assert adaptive.diagnostics["exact_forward_root_evaluations"] < full.diagnostics["exact_forward_root_evaluations"]


@pytest.mark.parametrize("case", ["reported", "high_co2", "low_oxygen", "precise_isotopes"])
def test_full_joint_posterior_agrees_with_direct_likelihood(monkeypatch, record_property, case):
    import updated_output_surface_joint_posterior as joint
    from sulfate_uncertainty import BackgroundConstraint

    request = joint.UpdatedJointPosteriorInput(
        target_air_cap_delta17_permil=None,
        measurement_sigma_permil=None,
        sulfate=normal_nonair_browser_observation(),
        free_coordinates=("pCO2", "GPP", "pO2"),
        pco2_bounds_ppm=(500.0, 4500.0),
        gpp_bounds_pgC_per_year=(200.0, 400.0),
        po2_bounds_pal=(0.6, 1.4),
        pco2_grid_size=17,
        gpp_grid_size=17,
        po2_grid_size=17,
        pco2_prior="uniform",
        gpp_prior="normal",
        gpp_prior_mean=290.0,
        gpp_prior_sigma=29.0,
        po2_prior="normal",
        po2_prior_mean=1.0,
        po2_prior_sigma=0.2,
    )
    if case in {"high_co2", "low_oxygen"}:
        request = replace(
            request,
            sulfate=replace(
                request.sulfate,
                measured_cap_delta17_permil=-2.0,
                background=BackgroundConstraint("normal", center=0.0, sigma=0.05),
            ),
            pco2_bounds_ppm=(20000.0, 60000.0),
        )
    if case == "low_oxygen":
        request = replace(
            request,
            pco2_bounds_ppm=(50.0, 20000.0),
            gpp_bounds_pgC_per_year=(18.3, 116.0),
            po2_bounds_pal=(0.1, 0.6),
            gpp_prior_mean=58.0,
            gpp_prior_sigma=29.0,
            po2_prior_mean=0.2,
            po2_prior_sigma=0.2,
        )
    if case == "precise_isotopes":
        request = replace(
            request,
            sulfate=replace(request.sulfate, cap_delta17_sigma_permil=0.003),
        )
    # Identical atmospheric grids isolate likelihood-acceleration error from
    # atmospheric quadrature resolution, which has separate refinement tests.
    with sulfate_computation_budget(max_seconds=60):
        actual = joint.joint_updated_posterior(request)

    def direct(a, d, r):
        return exact_sulfate_likelihood(
            a,
            d,
            r,
            settings=SulfateIntegrationSettings(
                relative_tolerance=2.5e-5,
                absolute_tolerance=1e-11,
                max_level=8,
                adaptive_delta18=True,
                focused_fraction_quadrature=True,
            ),
        )

    monkeypatch.setattr(joint, "exact_sulfate_likelihood", direct)
    # Reference accuracy is the gate; allow slower CI hosts more integration time.
    with sulfate_computation_budget(max_seconds=180):
        expected = joint.joint_updated_posterior(request)
    a, b = np.asarray(actual.posterior_probability_mass), np.asarray(
        expected.posterior_probability_mass
    )
    total_variation = float(0.5 * np.sum(np.abs(a - b)))
    record_property("posterior_total_variation", total_variation)
    assert total_variation < 1e-4
    assert actual.posterior_integral == pytest.approx(1.0)
    for coordinate in request.free_coordinates:
        lo, hi = expected.equal_tailed_credible_intervals[coordinate]
        record_property(
            coordinate + "_median_absolute_difference",
            abs(actual.posterior_median[coordinate] - expected.posterior_median[coordinate]),
        )
        record_property(
            coordinate + "_interval_maximum_absolute_difference",
            float(np.max(np.abs(np.asarray(actual.equal_tailed_credible_intervals[coordinate]) - (lo, hi)))),
        )
        assert actual.posterior_median[coordinate] == pytest.approx(
            expected.posterior_median[coordinate], abs=1e-4 * (hi - lo)
        )
        np.testing.assert_allclose(
            actual.equal_tailed_credible_intervals[coordinate],
            (lo, hi),
            atol=1e-4 * (hi - lo),
            rtol=0,
        )


def test_checked_table_and_request_local_reuse_match_direct_integrals():
    rng = np.random.default_rng(96512)
    a = rng.uniform(-4.0, -0.3, 2400)
    d = rng.uniform(20.0, 26.0, a.size)
    r = normal_nonair_browser_observation()
    with sulfate_computation_budget():
        result = scalable_sulfate_likelihood(a, d, r)
        detail = result.diagnostics["likelihood_interpolation"]
        assert detail["maximum_check_error_over_tolerance"] <= 1.0
        before = _COMPUTE_BUDGET.get().remaining_roots
        repeated = scalable_sulfate_likelihood(a, d, r)
        assert repeated.diagnostics["likelihood_interpolation"]["reused_within_request"]
        assert _COMPUTE_BUDGET.get().remaining_roots == before
        np.testing.assert_array_equal(repeated.log_likelihood, result.log_likelihood)
        assert len(_COMPUTE_BUDGET.get().likelihood_tables) == 1
    assert _COMPUTE_BUDGET.get() is None
    index = np.arange(0, a.size, 31)
    direct = exact_sulfate_likelihood(
        a[index],
        d[index],
        r,
        settings=SulfateIntegrationSettings(
            relative_tolerance=2.5e-5, absolute_tolerance=1e-11, max_level=7
        ),
    )
    scale = r.cap_delta17_sigma_permil * np.sqrt(2 * np.pi)
    expected = np.exp(direct.log_likelihood) * scale
    actual = np.exp(result.log_likelihood[index]) * scale
    assert np.max(np.abs(expected - actual) / (1e-10 + 2e-4 * expected)) < 1.0
    with sulfate_computation_budget():
        assert not _COMPUTE_BUDGET.get().likelihood_tables


def test_table_cannot_bypass_the_exact_integration_budget():
    with pytest.raises(SulfateComputeLimitError):
        with sulfate_computation_budget(max_roots=10):
            scalable_sulfate_likelihood(
                np.linspace(-4, -0.3, 2400),
                np.linspace(20, 26, 2400),
                normal_nonair_browser_observation(),
            )


@pytest.mark.parametrize("correlation", [0.0, 0.7, -0.7])
@pytest.mark.parametrize("sigma18", [0.5, 5.0])
def test_independently_refined_delta18_quadrature_matches_full_rules(
    correlation, sigma18
):
    r = replace(
        normal_nonair_browser_observation(),
        isotope_error_correlation=correlation,
        delta18_sigma_permil=sigma18,
    )
    a = np.array([-2.7, -2.1, -1.75, -1.3, -0.6])
    base = SulfateIntegrationSettings(max_level=7, relative_tolerance=5e-5)
    direct = exact_sulfate_likelihood(a, 23.9, r, settings=base)
    adaptive = exact_sulfate_likelihood(
        a, 23.9, r, settings=replace(base, adaptive_delta18=True)
    )
    np.testing.assert_allclose(
        np.exp(adaptive.log_likelihood),
        np.exp(direct.log_likelihood),
        rtol=2e-4,
        atol=2e-9,
    )
    assert adaptive.diagnostics["delta18_quadrature_check_states"] >= a.size
