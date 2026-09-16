"""Reference-coordinate and likelihood tests, independent of atmospheric fitting."""

from dataclasses import replace
from math import exp, log
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, truncnorm

from sulfate_to_air import (
    AnomalyCoordinate,
    ConditionalSulfateTransfer,
    IncorporationConstraint,
    sulfate_log_likelihood,
    SulfateProcessAssumptions,
    exact_air_from_sulfate,
    exact_sulfate_from_air,
)
from isotopes import R17_VSMOW, R18_VSMOW


def scenario(coordinate=AnomalyCoordinate("logarithmic", 0.528)):
    return ConditionalSulfateTransfer(
        coordinate=coordinate, background_permil=0.0, air_path_shift_permil=0.0,
        reported_bias_permil=0.0, air_scope="full_air",
        assumption_note="Synthetic test: zero background and air-path shift; preserved primary signal.",
    )


@pytest.mark.parametrize("form,slope", [("linear", 0.52), ("logarithmic", 0.52),
                                       ("logarithmic", 0.528), ("logarithmic", 0.5305)])
@pytest.mark.parametrize("air", [-0.432, -10.0, -40.0])
def test_coordinate_roundtrip_and_transfer_keeps_native_law(form, slope, air):
    coordinate = AnomalyCoordinate(form, slope)
    native = coordinate.from_oxytib(air, 23.9)
    assert coordinate.to_oxytib(native, 23.9) == pytest.approx(air, abs=2e-12)
    transfer = replace(scenario(coordinate), background_permil=-0.02, air_path_shift_permil=0.03,
                       reported_bias_permil=-0.05)
    sulfate = coordinate.to_oxytib(transfer.sulfate_native(native, 0.17), 15.0)
    result = transfer.infer_air(sulfate_cap_delta17_permil=sulfate, sulfate_delta18_permil=15.0,
                               fraction=0.17, air_delta18_permil=23.9)
    assert result["air_cap_delta17_permil"] == pytest.approx(air, abs=2e-12)
    assert result["transfer"]["coordinate"] == {"form": form, "slope": slope}


def test_historical_arithmetic_retains_scope_and_bias():
    bao = replace(scenario(AnomalyCoordinate("logarithmic", 0.52)),
                  air_scope="historical_photochemical_component", reported_bias_permil=-0.05)
    assert bao.air_native(-0.29, 0.1) == pytest.approx(-2.4)
    assert bao.air_native(-0.70, 0.1) == pytest.approx(-6.5)
    with pytest.raises(ValueError, match="historical photochemical component"):
        bao.infer_air(sulfate_cap_delta17_permil=-0.29, sulfate_delta18_permil=15.0,
                      fraction=0.1, air_delta18_permil=23.9)
    cao = scenario(AnomalyCoordinate("linear", 0.52))
    assert cao.air_native(-4.2, 0.1) == pytest.approx(-42.0)
    assert cao.air_native(-4.2, 0.25) == pytest.approx(-16.8)


def test_converting_only_sulfate_then_dividing_is_not_reference_consistent():
    transfer = scenario(AnomalyCoordinate("logarithmic", 0.52))
    air, fraction = -10.0, 0.1
    native_air = transfer.coordinate.from_oxytib(air, 23.9)
    sulfate = transfer.coordinate.to_oxytib(transfer.sulfate_native(native_air, fraction), 15.0)
    assert abs(sulfate / fraction - air) > 0.9
    result = transfer.infer_air(sulfate_cap_delta17_permil=sulfate, sulfate_delta18_permil=15.0,
                               fraction=fraction, air_delta18_permil=23.9)
    assert result["air_cap_delta17_permil"] == pytest.approx(air)


def test_required_assumptions_and_coefficients_have_no_implicit_defaults():
    with pytest.raises(TypeError):
        ConditionalSulfateTransfer(coordinate=AnomalyCoordinate("logarithmic", 0.528))
    with pytest.raises(ValueError, match="document"):
        replace(scenario(), assumption_note="")
    with pytest.raises(ValueError, match="finite"):
        replace(scenario(), background_permil=float("nan"))
    with pytest.raises(ValueError, match="scope"):
        replace(scenario(), air_scope="unknown")


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_fractions(value):
    with pytest.raises(ValueError):
        scenario().air_native(-1.0, value)
    with pytest.raises(ValueError):
        IncorporationConstraint("fixed", center=value)


def test_zero_inheritance_forward_has_no_air_information_and_inverse_is_rejected():
    transfer = replace(scenario(), background_permil=-0.09)
    assert transfer.sulfate_native(-10.0, 0.0) == pytest.approx(-0.09)
    with pytest.raises(ValueError, match="no finite air"):
        transfer.air_native(-0.09, 0.0)
    values = sulfate_log_likelihood(np.array([-40.0, -10.0, 0.0]), measured_sulfate_permil=-0.09,
                                   analytical_sigma_permil=0.03,
                                   incorporation=IncorporationConstraint("fixed", center=0.0),
                                   transfer=transfer)
    assert np.ptp(values) == 0.0


@pytest.mark.parametrize("constraint", [
    {"kind": "range", "lower": 0.25, "upper": 0.1},
    {"kind": "normal", "center": 0.2, "sigma": 0.0},
    {"kind": "normal", "center": 0.2, "sigma": float("nan")},
    {"kind": "fixed", "center": 0.2, "sigma": 0.1},
    {"kind": "range", "lower": 0.1, "upper": 0.2, "center": 0.15},
])
def test_ambiguous_or_invalid_constraints_fail(constraint):
    with pytest.raises(ValueError):
        IncorporationConstraint(**constraint)


@pytest.mark.parametrize("constraint", [
    IncorporationConstraint("range", lower=0.1, upper=0.25),
    IncorporationConstraint("range", lower=0.0, upper=1.0),
    IncorporationConstraint("normal", center=0.2, sigma=0.03),
    IncorporationConstraint("normal", center=0.05, sigma=0.2),
    IncorporationConstraint("normal", center=1.0, sigma=0.1),
])
@pytest.mark.parametrize("air,observation,sigma", [(-0.432, -0.08, 0.04), (-10.0, -2.0, 0.06),
                                                (1.0, 0.2, 0.05), (0.0, -0.04, 0.03)])
def test_integrated_likelihood_matches_independent_quadrature(constraint, air, observation, sigma):
    transfer = replace(scenario(), background_permil=-0.02, air_path_shift_permil=0.01,
                       reported_bias_permil=-0.005)
    if constraint.kind == "range":
        lower, upper = constraint.lower, constraint.upper
        def prior(f):
            return 1.0 / (upper - lower)
    else:
        lower, upper = 0.0, 1.0
        mu, sd = constraint.center, constraint.sigma
        def prior(f):
            return truncnorm.pdf(f, -mu/sd, (1.0-mu)/sd, loc=mu, scale=sd)
    expected = quad(lambda f: norm.pdf(observation, loc=transfer.sulfate_native(air, f), scale=sigma)
                    * prior(f), lower, upper, epsabs=1e-13, epsrel=1e-10)[0]
    actual = sulfate_log_likelihood(air, measured_sulfate_permil=observation,
                                   analytical_sigma_permil=sigma, incorporation=constraint, transfer=transfer)
    assert exp(float(actual)) == pytest.approx(expected, rel=2e-8, abs=1e-13)


@pytest.mark.parametrize("constraint", [IncorporationConstraint("range", lower=0.1, upper=0.25),
                                       IncorporationConstraint("normal", center=0.2, sigma=0.03)])
def test_no_contrast_limit_and_tiny_contrast_are_stable(constraint):
    transfer = replace(scenario(), background_permil=-0.04, air_path_shift_permil=0.02)
    values = sulfate_log_likelihood(np.array([-0.06, -0.06+1e-14, -0.06-1e-14]),
                                   measured_sulfate_permil=-0.05, analytical_sigma_permil=0.01,
                                   incorporation=constraint, transfer=transfer)
    assert values == pytest.approx(np.full(3, norm.logpdf(-0.05, -0.04, 0.01)), abs=1e-10)


def test_likelihood_is_density_in_measured_sulfate_not_divided_air_density():
    fraction = 0.1
    result = sulfate_log_likelihood(-10.0, measured_sulfate_permil=-1.0, analytical_sigma_permil=0.03,
                                   incorporation=IncorporationConstraint("fixed", center=fraction),
                                   transfer=scenario())
    inverse_space = norm.logpdf(-10.0, loc=-1.0 / fraction, scale=0.03 / fraction)
    assert float(result) == pytest.approx(norm.logpdf(-1.0, -1.0, 0.03))
    assert float(result) - inverse_space == pytest.approx(-log(fraction))


def test_far_tail_likelihood_does_not_underflow_to_negative_infinity():
    result = sulfate_log_likelihood(10.0, measured_sulfate_permil=-1.0, analytical_sigma_permil=0.01,
                                   incorporation=IncorporationConstraint("range", lower=0.1, upper=0.3),
                                   transfer=scenario())
    assert np.isfinite(result) and result < -10000.0


def test_likelihood_rejects_mixed_coordinates_components_and_zero_sigma():
    kwargs = dict(measured_sulfate_permil=-0.5, analytical_sigma_permil=0.03,
                  incorporation=IncorporationConstraint("fixed", center=0.25))
    for transfer in (scenario(AnomalyCoordinate("linear", 0.52)),
                     replace(scenario(), air_scope="historical_photochemical_component")):
        with pytest.raises(ValueError, match="full-air log-0.528"):
            sulfate_log_likelihood(-2.0, transfer=transfer, **kwargs)
    kwargs["analytical_sigma_permil"] = 0.0
    with pytest.raises(ValueError, match="positive analytical sigma"):
        sulfate_log_likelihood(-2.0, transfer=scenario(), **kwargs)


@pytest.mark.parametrize("form", ["logarithmic", "linear"])
def test_invalid_isotope_coordinates(form):
    coordinate = AnomalyCoordinate(form, 0.52)
    with pytest.raises(ValueError):
        coordinate.to_oxytib(-1.0, -1000.0)
    with pytest.raises(ValueError):
        coordinate.from_oxytib(float("nan"), 20.0)
    with pytest.raises(ValueError):
        coordinate.to_oxytib(0.0, float("inf"))
    if form == "linear":
        with pytest.raises(ValueError):
            coordinate.to_oxytib(-1020.0, 20.0)


def exact_process(alpha18=1.0, theta=0.528, background=-0.03):
    return SulfateProcessAssumptions(
        background_cap_delta17_permil=background, alpha18_air_to_sulfate=alpha18,
        theta_air_to_sulfate=theta,
        assumption_note="Synthetic prescribed isotope-atom mixture; no archive calibration or resetting.",
    )


def independent_mixture(air_d17, air_d18, background_d18, fraction, process):
    """Generate both sulfate isotopes from independently specified full sources."""
    def atoms(anomaly, delta18, a17=1.0, a18=1.0):
        r18 = R18_VSMOW * (1.0+delta18/1000.0) * a18
        r17 = R17_VSMOW * exp(anomaly/1000.0) * (1.0+delta18/1000.0)**0.528 * a17
        array = np.array([1.0, r17, r18])
        return array / np.sum(array)
    a18 = process.alpha18_air_to_sulfate
    mixture = (fraction * atoms(air_d17, air_d18, a18**process.theta_air_to_sulfate, a18)
               + (1.0-fraction) * atoms(process.background_cap_delta17_permil, background_d18))
    d18 = 1000.0 * (mixture[2]/mixture[0]/R18_VSMOW-1.0)
    d17 = 1000.0 * log(mixture[1]/mixture[0]/R17_VSMOW) - 0.528*1000.0*log(1.0+d18/1000.0)
    return d17, d18


@pytest.mark.parametrize("air", [-0.432, -10.0, -40.0])
@pytest.mark.parametrize("fraction", [0.05, 0.1, 0.25, 0.8])
@pytest.mark.parametrize("background_d18", [-20.0, 10.0, 30.0])
@pytest.mark.parametrize("alpha,theta", [(1.0, 0.528), (0.985, 0.515), (1.02, 0.530)])
def test_exact_closure_without_inputting_background_delta18(air, fraction, background_d18, alpha, theta):
    process = exact_process(alpha18=alpha, theta=theta)
    s17, s18 = independent_mixture(air, 23.9, background_d18, fraction, process)
    inferred = exact_air_from_sulfate(sulfate_cap_delta17_permil=s17, sulfate_delta18_permil=s18,
                                      fraction=fraction, air_delta18_permil=23.9, process=process)
    assert inferred["air"]["cap_delta17_permil"] == pytest.approx(air, abs=5e-9)
    assert inferred["implied_nonair_oxygen"]["delta18_permil"] == pytest.approx(background_d18, abs=2e-9)
    assert inferred["implied_nonair_oxygen"]["cap_delta17_permil"] == pytest.approx(-0.03, abs=2e-9)
    assert inferred["maximum_atom_closure_residual"] < 5e-15
    predicted = exact_sulfate_from_air(air_cap_delta17_permil=air, air_delta18_permil=23.9,
                                       sulfate_delta18_permil=s18, fraction=fraction, process=process)
    assert predicted == pytest.approx(s17, abs=2e-9)


def test_exact_pure_air_inheritance_and_incompatible_delta18():
    process = exact_process(alpha18=0.99, theta=0.515)
    s17, s18 = independent_mixture(-10.0, 23.9, 10.0, 1.0, process)
    output = exact_air_from_sulfate(sulfate_cap_delta17_permil=s17, sulfate_delta18_permil=s18,
                                  fraction=1.0, air_delta18_permil=23.9, process=process)
    assert output["air"]["cap_delta17_permil"] == pytest.approx(-10.0)
    assert output["implied_nonair_oxygen"] is None
    assert exact_sulfate_from_air(air_cap_delta17_permil=-10.0, air_delta18_permil=23.9,
                                 sulfate_delta18_permil=s18, fraction=1.0, process=process) == pytest.approx(s17)
    with pytest.raises(ValueError, match="matching"):
        exact_air_from_sulfate(sulfate_cap_delta17_permil=s17, sulfate_delta18_permil=15.0,
                              fraction=1.0, air_delta18_permil=23.9, process=process)


def test_exact_zero_inheritance_and_nonphysical_mixture():
    process = exact_process()
    assert exact_sulfate_from_air(air_cap_delta17_permil=-10.0, air_delta18_permil=23.9,
                                 sulfate_delta18_permil=10.0, fraction=0.0, process=process) == -0.03
    with pytest.raises(ValueError, match="no finite"):
        exact_air_from_sulfate(sulfate_cap_delta17_permil=-0.03, sulfate_delta18_permil=10.0,
                              fraction=0.0, air_delta18_permil=23.9, process=process)
    with pytest.raises(ValueError, match="no physical"):
        exact_sulfate_from_air(air_cap_delta17_permil=-10.0, air_delta18_permil=23.9,
                              sulfate_delta18_permil=-900.0, fraction=0.5, process=process)


def test_exact_requires_explicit_process_coefficients_and_finite_inputs():
    with pytest.raises(TypeError):
        SulfateProcessAssumptions(background_cap_delta17_permil=0.0)
    for values in ({"alpha18_air_to_sulfate": 0.0}, {"theta_air_to_sulfate": float("nan")},
                   {"background_cap_delta17_permil": float("inf")}, {"assumption_note": ""}):
        with pytest.raises(ValueError):
            replace(exact_process(), **values)
    with pytest.raises(ValueError, match="finite"):
        exact_air_from_sulfate(sulfate_cap_delta17_permil=-1.0, sulfate_delta18_permil=15.0,
                              fraction=0.1, air_delta18_permil=float("nan"), process=exact_process())


@pytest.mark.parametrize("po2,pco2,gpp", [(1.0, 294.0, 290.0), (0.5, 10000.0, 72.5)])
def test_atmospheric_model_pair_to_sulfate_and_back_without_modern_air_delta18_override(po2, pco2, gpp):
    from public_model_service import forward
    from updated_molecular_forward_model import UpdatedForwardInput
    from isotopes import conventional_delta_from_prime

    model = forward(UpdatedForwardInput(p_o2_pal=po2, p_co2_ppm=pco2, gpp_pgC_per_year=gpp))["result"]
    air = model["central_cap_delta17_prime_permil"]
    delta18 = conventional_delta_from_prime(model["central_delta18_prime_permil"])
    process = exact_process()
    sulfate = exact_sulfate_from_air(air_cap_delta17_permil=air, air_delta18_permil=delta18,
                                     sulfate_delta18_permil=10.0, fraction=0.25, process=process)
    recovered = exact_air_from_sulfate(sulfate_cap_delta17_permil=sulfate, sulfate_delta18_permil=10.0,
                                     air_delta18_permil=delta18, fraction=0.25, process=process)
    assert recovered["air"]["cap_delta17_permil"] == pytest.approx(air, abs=2e-9)
    assert recovered["air"]["delta18_permil"] == pytest.approx(delta18, abs=2e-9)


def test_console_sulfate_calculation_preserves_process_metadata(capsys):
    from public_cli import build_parser, _run

    args = build_parser().parse_args([
        "sulfate", "--d17o", "-1", "--d18o", "15", "--fraction", "0.1",
        "--air-d18o", "23.9", "--background-d17o", "0", "--alpha18", "1",
        "--theta", "0.528", "--assumptions", "Synthetic preserved-primary scenario",
    ])
    _run(args)
    result = json.loads(capsys.readouterr().out)
    assert result["coordinate"]["slope"] == 0.528
    assert result["sulfate"]["delta18_permil"] == pytest.approx(15.0)
    assert result["process"]["assumption_note"] == "Synthetic preserved-primary scenario"
    assert result["atmospheric_incorporation_fraction"] == 0.1
    assert "implied_nonair_oxygen" in result


def test_synthetic_audit_distinguishes_exact_and_first_order_results():
    from audit_sulfate_transfer import implementation_checks

    result = implementation_checks()["finite_anomaly_approximation"]
    assert result["maximum_absolute_air_error_permil_by_true_air"]["-10.0"] > 0.1
    assert result["exact_inverse_maximum_absolute_air_error_permil"] < 1e-8
@pytest.mark.parametrize("air_anomaly", [-.432, -10., -25.])
@pytest.mark.parametrize("alpha18,alpha17", [(1., np.exp(-.02376/1000)),
                                           (np.exp(-.0216), np.exp(-.0216*.5145))])
def test_independent_factors_match_atom_mixture_scalar_grid_and_inverse(air_anomaly, alpha18, alpha17):
    from sulfate_to_air import (
        _composition, _isotopes, _atoms, exact_sulfate_grid,
        exact_air_from_sulfate, exact_sulfate_from_air, SulfateProcessAssumptions,
    )
    air = _composition(air_anomaly, 23.9)
    transferred = _atoms(air[1]/air[0]*alpha17, air[2]/air[0]*alpha18)
    background = _composition(.02, -5.)
    mixture = .25*transferred+.75*background
    sulfate = _isotopes(mixture)
    process = SulfateProcessAssumptions(.02, alpha18, None, "Conditional test", alpha17)
    inverse = exact_air_from_sulfate(sulfate_cap_delta17_permil=sulfate["cap_delta17_permil"],
        sulfate_delta18_permil=sulfate["delta18_permil"], fraction=.25, air_delta18_permil=23.9, process=process)
    assert inverse["air"]["cap_delta17_permil"] == pytest.approx(air_anomaly, abs=1e-9)
    forward = exact_sulfate_from_air(air_cap_delta17_permil=air_anomaly, air_delta18_permil=23.9,
        sulfate_delta18_permil=sulfate["delta18_permil"], fraction=.25, process=process)
    grid, valid = exact_sulfate_grid(air_cap_delta17_permil=air_anomaly, air_delta18_permil=23.9,
        sulfate_delta18_permil=sulfate["delta18_permil"], fraction=.25, background_cap_delta17_permil=.02,
        alpha18_air_to_sulfate=alpha18, theta_air_to_sulfate=None, alpha17_air_to_sulfate=alpha17)
    assert valid
    assert forward == pytest.approx(sulfate["cap_delta17_permil"], abs=1e-9)
    assert float(grid) == pytest.approx(forward, abs=1e-9)


def test_independent_factors_pure_endpoint_and_ambiguous_input():
    from sulfate_to_air import air_fractionation_factors, exact_sulfate_grid
    with pytest.raises(ValueError, match="not both"):
        air_fractionation_factors(1., .528, .999)
    with pytest.raises(ValueError, match="positive"):
        air_fractionation_factors(1., None, 0.)
    with pytest.raises(ValueError):
        air_fractionation_factors(1., None, float("nan"))
    grid, valid = exact_sulfate_grid(air_cap_delta17_permil=-10., air_delta18_permil=23.9,
        sulfate_delta18_permil=23.9, fraction=1., background_cap_delta17_permil=.02,
        alpha18_air_to_sulfate=1., theta_air_to_sulfate=None, alpha17_air_to_sulfate=np.exp(-.02376/1000))
    assert valid
    assert float(grid) == pytest.approx(-10.02376, abs=1e-9)
