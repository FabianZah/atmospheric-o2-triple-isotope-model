"""Tests for joint inference on the updated molecular output surface."""

from __future__ import annotations

import numpy as np
import pytest

from updated_output_surface import UpdatedOutputSurfaceInput, load_updated_output_surface
from updated_output_surface_joint_posterior import (
    UpdatedJointPosteriorInput,
    joint_updated_posterior,
)


def _target(point: UpdatedOutputSurfaceInput) -> float:
    return load_updated_output_surface().evaluate(point).central_cap_delta17_prime_permil


def test_vectorized_central_grid_matches_scalar_evaluations() -> None:
    surface = load_updated_output_surface()
    pco2 = np.asarray([[200.0, 294.0], [1000.0, 10000.0]])
    gpp = np.asarray([[120.0, 290.0], [450.0, 700.0]])
    vectorized = surface.evaluate_central_cap_delta17_grid(
        p_o2_pal=1.0,
        p_co2_ppm=pco2,
        gpp_pgC_per_year=gpp,
    )
    scalar = np.asarray(
        [
            surface.evaluate(UpdatedOutputSurfaceInput(1.0, co2, production)).central_cap_delta17_prime_permil
            for co2, production in zip(pco2.reshape(-1), gpp.reshape(-1), strict=True)
        ]
    ).reshape(pco2.shape)
    assert vectorized == pytest.approx(scalar, abs=1.0e-12)


def test_two_coordinate_posterior_normalizes_and_contains_generated_point() -> None:
    point = UpdatedOutputSurfaceInput(1.0, 4200.0, 290.0)
    result = joint_updated_posterior(
        UpdatedJointPosteriorInput(
            target_air_cap_delta17_permil=_target(point),
            measurement_sigma_permil=0.02,
            free_coordinates=("pCO2", "GPP"),
            p_o2_pal=point.p_o2_pal,
            pco2_bounds_ppm=(500.0, 20000.0),
            gpp_bounds_pgC_per_year=(120.0, 700.0),
            pco2_grid_size=61,
            gpp_grid_size=61,
        )
    )

    assert result.posterior_integral == pytest.approx(1.0, abs=1.0e-12)
    assert sum(result.posterior_probability_mass) == pytest.approx(1.0, abs=1.0e-12)
    assert result.hpd_probability_mass >= 0.95
    assert result.posterior_shape == (61, 61)
    for coordinate, expected in (("pCO2", 4200.0), ("GPP", 290.0)):
        lower, upper = result.equal_tailed_credible_intervals[coordinate]
        assert lower < expected < upper


def test_three_coordinate_posterior_preserves_all_marginals() -> None:
    point = UpdatedOutputSurfaceInput(0.5, 3000.0, 250.0)
    result = joint_updated_posterior(
        UpdatedJointPosteriorInput(
            target_air_cap_delta17_permil=_target(point),
            measurement_sigma_permil=0.03,
            free_coordinates=("pCO2", "GPP", "pO2"),
            pco2_bounds_ppm=(500.0, 10000.0),
            gpp_bounds_pgC_per_year=(120.0, 700.0),
            po2_bounds_pal=(0.2, 1.2),
            pco2_grid_size=31,
            gpp_grid_size=31,
            po2_grid_size=31,
        )
    )

    assert result.posterior_shape == (31, 31, 31)
    assert set(result.marginal_density) == {"pCO2", "GPP", "pO2"}
    for coordinate in result.free_coordinates:
        assert sum(result.marginal_probability_mass[coordinate]) == pytest.approx(
            1.0, abs=1.0e-12
        )


def test_explicit_model_discrepancy_changes_joint_posterior_and_is_reported() -> None:
    point = UpdatedOutputSurfaceInput(1.0, 5000.0, 290.0)
    common = dict(
        target_air_cap_delta17_permil=_target(point),
        measurement_sigma_permil=0.01,
        free_coordinates=("pCO2", "GPP"),
        p_o2_pal=1.0,
        pco2_bounds_ppm=(1000.0, 20000.0),
        gpp_bounds_pgC_per_year=(200.0, 500.0),
        pco2_grid_size=61,
        gpp_grid_size=61,
    )
    analytical_only = joint_updated_posterior(UpdatedJointPosteriorInput(**common))
    with_discrepancy = joint_updated_posterior(
        UpdatedJointPosteriorInput(
            **common,
            model_discrepancy_sigma_permil=0.02,
            model_discrepancy_source="synthetic validation test",
        )
    )

    assert with_discrepancy.posterior_probability_mass != pytest.approx(
        analytical_only.posterior_probability_mass, abs=1.0e-12
    )
    assert with_discrepancy.posterior_integral == pytest.approx(1.0, abs=1.0e-12)
    assert with_discrepancy.probabilistic_model_discrepancy_included
    assert with_discrepancy.effective_likelihood_sigma_permil == pytest.approx(
        np.hypot(0.01, 0.02)
    )
    assert "synthetic validation test" in with_discrepancy.probability_scope


def test_model_discrepancy_requires_traceable_source() -> None:
    with pytest.raises(ValueError, match="traceable source"):
        joint_updated_posterior(
            UpdatedJointPosteriorInput(
                target_air_cap_delta17_permil=-0.432,
                measurement_sigma_permil=0.01,
                model_discrepancy_sigma_permil=0.02,
                pco2_grid_size=17,
                gpp_grid_size=17,
            )
        )
