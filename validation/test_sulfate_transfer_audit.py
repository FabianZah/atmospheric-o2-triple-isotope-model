"""Published arithmetic, input-domain, and reference-convention checks."""

from math import expm1, log1p

import pytest

from audit_sulfate_transfer import (
    air_interval, build_report, convert_anomaly, equilibrium_sulfate, first_order_air_anomaly,
)


@pytest.mark.parametrize("sulfate,expected", [(-0.29, -2.4), (-0.70, -6.5)])
def test_bao_published_air_requires_source_specific_analytical_correction(sulfate, expected):
    corrected = first_order_air_anomaly(sulfate, 0.10, nonair_endmember_permil=0.0, reported_bias_permil=-0.05)
    uncorrected = first_order_air_anomaly(sulfate, 0.10, nonair_endmember_permil=0.0, reported_bias_permil=0.0)
    assert corrected == pytest.approx(expected, abs=1e-12)
    assert corrected - uncorrected == pytest.approx(0.5)


def test_svalbard_endmember_reproduces_rounded_published_interval():
    bounds = air_interval((-4.2, -4.2), (0.10, 0.25), nonair_bounds=(0.0, 0.0), reported_bias_permil=0.0)
    assert bounds == pytest.approx((-42.0, -16.8))
    assert max(abs(a-b) for a, b in zip(bounds, (-42.0, -17.0))) <= 0.201


def test_interval_remains_ordered_when_target_and_background_overlap():
    bounds = air_interval((-0.1, 0.1), (0.1, 0.2), nonair_bounds=(-0.2, 0.2), reported_bias_permil=0.0)
    assert bounds == pytest.approx((-2.8, 2.8))


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_fraction_cannot_be_zero_or_silently_clipped(fraction):
    with pytest.raises(ValueError):
        first_order_air_anomaly(-1.0, fraction, nonair_endmember_permil=0.0, reported_bias_permil=0.0)


def test_range_reaching_zero_is_unbounded_and_rejected():
    with pytest.raises(ValueError, match="fraction"):
        air_interval((-1.0, -1.0), (0.0, 0.2), nonair_bounds=(0.0, 0.0), reported_bias_permil=0.0)


def test_water_equilibrium_matches_waldeck_rounded_endmember():
    d18, anomaly = equilibrium_sulfate(25.0, slope=0.5305)
    assert d18 == pytest.approx(23.0, abs=0.1)
    assert anomaly == pytest.approx(-0.148, abs=0.001)
    converted = convert_anomaly(anomaly, d18, source_slope=0.5305, source_form="logarithmic")
    assert converted == pytest.approx(equilibrium_sulfate(25.0, slope=0.528)[1])


def test_large_linear_anomaly_uses_full_logarithmic_conversion():
    d18, d17 = 24.0, -29.52  # Linear Delta_0.52 = -42 per mil.
    actual = convert_anomaly(-42.0, d18, source_slope=0.52, source_form="linear")
    expected = 1000 * (log1p(d17/1000) - 0.528 * log1p(d18/1000))
    wrong = convert_anomaly(-42.0, d18, source_slope=0.52, source_form="logarithmic")
    assert actual == pytest.approx(expected)
    assert abs(actual - wrong) > 0.1


def test_reference_change_preserves_original_isotope_ratio():
    original, d18 = -0.65, 17.0
    converted = convert_anomaly(original, d18, source_slope=0.52, source_form="logarithmic")
    prime18 = 1000 * log1p(d18/1000)
    assert expm1((original + 0.52 * prime18)/1000) == pytest.approx(
        expm1((converted + 0.528 * prime18)/1000), abs=1e-14)


def test_audit_keeps_unknown_geological_constraints_unassigned():
    report = build_report()
    decision = report["decision"]
    assert decision["universal_incorporation_fraction"] is None
    assert decision["universal_background_range"] is None
    assert decision["unknown_preservation_distribution"] is None
    assert report["bao_2008_co2_from_published_table_only"]["calculated_pco2_ppm"] == pytest.approx(
        {"Early Cambrian": 4263.0, "Marinoan barite": 12137.666666666666})
