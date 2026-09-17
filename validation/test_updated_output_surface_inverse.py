from __future__ import annotations

from itertools import product

import pytest

from forward_isotope_constraints import ForwardIsotopeConstraints, predict_isotopes
from updated_constrained_pco2_posterior import CoordinateConstraint

from updated_output_surface import (
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)
from updated_output_surface_inverse import (
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)


@pytest.mark.parametrize(
    "solve_for,point",
    (
        ("pCO2", (0.5, 4200.0, 182.56264)),
        ("GPP", (1.0, 10000.0, 290.0)),
        ("pO2", (0.5, 10000.0, 182.56264)),
    ),
)
def test_each_coordinate_recovers_surface_generated_target(solve_for, point) -> None:
    surface = load_updated_output_surface()
    target = surface.evaluate(UpdatedOutputSurfaceInput(*point)).central_cap_delta17_prime_permil
    result = invert_updated_output_surface(
        UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=target,
            solve_for=solve_for,
            p_o2_pal=point[0],
            p_co2_ppm=point[1],
            gpp_pgC_per_year=point[2],
        ),
        verify_live_root=False,
    )
    expected = {"pO2": point[0], "pCO2": point[1], "GPP": point[2]}[solve_for]
    assert result.central_root == pytest.approx(expected, rel=1.0e-9)
    assert result.admissible_interval is not None
    assert result.admissible_interval[0] <= expected <= result.admissible_interval[1]
    assert result.live_root_verified is False


def test_live_root_verification_reports_small_residual() -> None:
    result = invert_updated_output_surface(
        UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=-0.432,
            measurement_uncertainty_permil=0.015,
            solve_for="pCO2",
            p_o2_pal=1.0,
            gpp_pgC_per_year=290.0,
        )
    )
    assert result.live_root_verified is True
    assert result.central_root is not None
    assert result.live_root_residual_permil == pytest.approx(0.0, abs=0.005)
    assert result.admissible_interval is not None


@pytest.mark.parametrize(
    "solve_for,target,p_o2,p_co2,gpp",
    (
        ("pCO2", -10.0, 1.0, 294.0, 91.28132),
        ("GPP", -10.0, 1.0, 10000.0, 290.0),
        ("pO2", -10.0, 1.0, 10000.0, 91.28132),
    ),
)
def test_live_residual_at_accelerated_root_is_within_surface_guardrail(
    solve_for, target, p_o2, p_co2, gpp
) -> None:
    result = invert_updated_output_surface(
        UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=target,
            measurement_uncertainty_permil=0.05,
            solve_for=solve_for,
            p_o2_pal=p_o2,
            p_co2_ppm=p_co2,
            gpp_pgC_per_year=gpp,
        )
    )
    assert result.live_root_verified is True
    assert result.live_root_residual_permil is not None
    assert abs(result.live_root_residual_permil) <= result.interpolation_guardrail_permil


def test_inverse_rejects_bounds_outside_surface() -> None:
    with pytest.raises(ValueError, match="exceed pCO2 surface domain"):
        invert_updated_output_surface(
            UpdatedSurfaceInverseInput(
                target_air_cap_delta17_permil=-1.0,
                solve_bounds=(49.0, 60000.0),
            ),
            verify_live_root=False,
        )


def _release_roundtrip_points():
    domain = load_updated_output_surface().domain
    corners = list(product(domain['po2_pal'], domain['pco2_ppm'], domain['gpp_pgC_per_year']))
    return corners + [
        (1.0, 294.0, 290.0), (1.0, 150.0, 290.0),
        (0.2, 8000.0, 58.0), (0.5, 30000.0, 72.5),
        (1.7, 52000.0, 675.0), (0.35, 17250.0, 113.0),
    ]


@pytest.mark.parametrize('point', _release_roundtrip_points())
@pytest.mark.parametrize('solve_for', ['pCO2', 'GPP', 'pO2'])
def test_public_forward_roundtrips_interior_and_domain_corners(point, solve_for):
    """Check numerical self-consistency, independently of observational validity."""
    po2, pco2, gpp = point
    prediction = predict_isotopes(ForwardIsotopeConstraints(
        pco2_constraint=CoordinateConstraint('fixed', center=pco2),
        gpp_constraint=CoordinateConstraint('fixed', center=gpp),
        po2_constraint=CoordinateConstraint('fixed', center=po2),
    ))
    target = prediction['isotopes']['cap_delta17_prime_permil']['median']
    recovered = invert_updated_output_surface(UpdatedSurfaceInverseInput(
        target_air_cap_delta17_permil=target, solve_for=solve_for,
        p_o2_pal=po2, p_co2_ppm=pco2, gpp_pgC_per_year=gpp,
    ), verify_live_root=False)
    expected = {'pO2':po2, 'pCO2':pco2, 'GPP':gpp}[solve_for]
    assert recovered.central_root == pytest.approx(expected, rel=1e-9)
    assert recovered.accelerated_root_residual_permil == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize('solve_for', ['pCO2', 'GPP', 'pO2'])
def test_measurement_overlap_expands_with_measurement_uncertainty(solve_for):
    point = (0.5, 8000.0, 72.5)
    target = load_updated_output_surface().evaluate(
        UpdatedOutputSurfaceInput(*point)).central_cap_delta17_prime_permil
    intervals = []
    for sigma in (0.0, 0.003, 0.06):
        result = invert_updated_output_surface(UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=target, solve_for=solve_for,
            measurement_uncertainty_permil=sigma,
            p_o2_pal=point[0], p_co2_ppm=point[1], gpp_pgC_per_year=point[2],
        ), verify_live_root=False)
        assert result.admissible_interval is not None
        intervals.append(result.admissible_interval)
    for inner, outer in zip(intervals, intervals[1:]):
        assert outer[0] <= inner[0]
        assert outer[1] >= inner[1]
