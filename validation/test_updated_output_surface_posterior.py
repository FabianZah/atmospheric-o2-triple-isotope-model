"""Tests for conditional inference on the updated molecular surface."""

from __future__ import annotations

import numpy as np
import pytest

from updated_output_surface import UpdatedOutputSurfaceInput, load_updated_output_surface
from updated_output_surface_posterior import (
    UpdatedConditionalPosteriorInput,
    _integral,
    conditional_updated_posterior,
)


def _target(point: UpdatedOutputSurfaceInput) -> float:
    return load_updated_output_surface().evaluate(point).central_cap_delta17_prime_permil


def test_integral_supports_numpy_without_legacy_trapz(monkeypatch) -> None:
    if not hasattr(np, "trapezoid"):
        monkeypatch.setattr(np, "trapezoid", np.trapz, raising=False)
    if hasattr(np, "trapz"):
        monkeypatch.delattr(np, "trapz")
    assert _integral(np.array([0.0, 1.0]), np.array([0.0, 2.0])) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    ("solve_for", "point", "expected"),
    (
        ("pCO2", UpdatedOutputSurfaceInput(0.5, 4200.0, 182.56264), 4200.0),
        ("GPP", UpdatedOutputSurfaceInput(1.0, 10000.0, 290.0), 290.0),
        ("pO2", UpdatedOutputSurfaceInput(0.5, 10000.0, 182.56264), 0.5),
    ),
)
def test_uniform_conditional_posterior_recovers_generated_target(
    solve_for, point, expected
) -> None:
    result = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(
            target_air_cap_delta17_permil=_target(point),
            measurement_sigma_permil=0.01,
            solve_for=solve_for,
            prior="uniform",
            p_o2_pal=point.p_o2_pal,
            p_co2_ppm=point.p_co2_ppm,
            gpp_pgC_per_year=point.gpp_pgC_per_year,
            grid_size=2049,
        ),
        verify_live_mode=False,
    )

    assert result.posterior_integral == pytest.approx(1.0, abs=2.0e-10)
    assert result.equal_tailed_credible_interval[0] < expected
    assert result.equal_tailed_credible_interval[1] > expected
    assert result.central_likelihood_root == pytest.approx(expected, rel=1.0e-9)


def test_smaller_measurement_sigma_narrows_posterior() -> None:
    common = dict(
        target_air_cap_delta17_permil=-3.0,
        solve_for="pCO2",
        prior="log_uniform",
        p_o2_pal=1.0,
        gpp_pgC_per_year=290.0,
        grid_size=2049,
    )
    broad = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(measurement_sigma_permil=0.05, **common),
        verify_live_mode=False,
    )
    narrow = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(measurement_sigma_permil=0.01, **common),
        verify_live_mode=False,
    )

    broad_width = np.diff(broad.equal_tailed_credible_interval)[0]
    narrow_width = np.diff(narrow.equal_tailed_credible_interval)[0]
    assert narrow_width < broad_width


def test_prior_choice_is_explicit_and_changes_conditional_summary() -> None:
    common = dict(
        target_air_cap_delta17_permil=-0.432,
        measurement_sigma_permil=0.015,
        solve_for="pCO2",
        p_o2_pal=1.0,
        gpp_pgC_per_year=290.0,
        solve_bounds=(294.0, 1000.0),
        grid_size=2049,
    )
    uniform = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(prior="uniform", **common),
        verify_live_mode=False,
    )
    log_uniform = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(prior="log_uniform", **common),
        verify_live_mode=False,
    )

    assert log_uniform.posterior_mean < uniform.posterior_mean
    assert log_uniform.inputs.prior == "log_uniform"
    assert "analytical measurement uncertainty only" in log_uniform.probability_scope


def test_out_of_range_target_is_reported_as_boundary_sensitive() -> None:
    result = conditional_updated_posterior(
        UpdatedConditionalPosteriorInput(
            target_air_cap_delta17_permil=-0.1,
            measurement_sigma_permil=0.01,
            solve_for="pCO2",
            prior="log_uniform",
            grid_size=1025,
        ),
        verify_live_mode=False,
    )

    assert result.status == "boundary_sensitive"
    assert result.boundary_sensitive is True
    assert result.central_likelihood_root is None
    assert result.lower_edge_probability > 0.05


@pytest.mark.parametrize(
    "posterior_input",
    (
        UpdatedConditionalPosteriorInput(-1.0, 0.0),
        UpdatedConditionalPosteriorInput(-1.0, -0.01),
        UpdatedConditionalPosteriorInput(-1.0, 0.01, credible_mass=1.0),
        UpdatedConditionalPosteriorInput(-1.0, 0.01, grid_size=256),
    ),
)
def test_invalid_probability_inputs_are_rejected(posterior_input) -> None:
    with pytest.raises(ValueError):
        conditional_updated_posterior(posterior_input, verify_live_mode=False)
