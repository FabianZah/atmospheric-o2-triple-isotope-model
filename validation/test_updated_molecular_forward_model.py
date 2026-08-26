"""Regression tests for the candidate updated molecular forward kernel."""

from __future__ import annotations

from pathlib import Path

import pytest

from isotopes import conventional_delta_from_prime

from updated_molecular_forward_model import (
    UpdatedForwardInput,
    run_updated_central_state,
    run_updated_forward,
)


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
BUNDLE = ROOT / "model_data" / "updated_r7_response_surface_v1.json"


def test_central_only_solver_matches_full_forward_central_state() -> None:
    request = UpdatedForwardInput(
        p_o2_pal=0.5,
        p_co2_ppm=4200.0,
        gpp_pgC_per_year=182.56264,
    )
    central = run_updated_central_state(request, bundle_path=BUNDLE)
    full = run_updated_forward(request, bundle_path=BUNDLE)
    assert central.numerically_converged is True
    assert central.maximum_fixed_point_residual_permil <= 1.0e-8
    assert central.delta18_prime_permil == pytest.approx(
        full.central_delta18_prime_permil, abs=1.0e-12
    )
    assert central.cap_delta17_prime_permil == pytest.approx(
        full.central_cap_delta17_prime_permil, abs=1.0e-12
    )


def test_modern_central_gpp_overlaps_pack_without_output_offset() -> None:
    result = run_updated_forward(
        UpdatedForwardInput(
            p_o2_pal=1.0,
            p_co2_ppm=294.0,
            gpp_pgC_per_year=290.0,
        ),
        bundle_path=BUNDLE,
    )

    assert result.numerically_converged is True
    assert result.numerical_solver_methods_used == ("undamped_newton",)
    assert result.central_cap_delta17_prime_permil == pytest.approx(
        -0.42635037513013607, abs=2.0e-9
    )
    assert result.central_residual_to_pack_permil == pytest.approx(
        0.005649624869863923, abs=2.0e-9
    )
    assert result.model_guardrail_overlaps_pack_observation is True
    assert result.pack_observed_uncertainty_permil == 0.015
    assert result.uncertainty_policy["output_offset_applied"] is False


def test_modern_conventional_delta18_overlaps_pack_measurement() -> None:
    result = run_updated_forward(
        UpdatedForwardInput(p_o2_pal=1.0, p_co2_ppm=294.0, gpp_pgC_per_year=290.0),
        bundle_path=BUNDLE,
    )
    conventional_delta18 = conventional_delta_from_prime(
        result.central_delta18_prime_permil
    )

    assert conventional_delta18 == pytest.approx(23.680359312, abs=2.0e-9)
    assert abs(conventional_delta18 - 23.9) <= 0.3
    assert result.biological_pathway_uncertainty_included is True
    assert result.biological_full_ensemble_member_count == 54
    assert len(result.biological_compact_envelope_member_keys) == 7
    assert (
        result.biological_process_interval_cap_delta17_permil[0]
        < result.pack_observed_cap_delta17_permil
        < result.biological_process_interval_cap_delta17_permil[1]
    )
    assert result.uncertainty_policy["output_offset_applied"] is False


def test_adnew_uncertainty_changes_forcing_monotonically() -> None:
    result = run_updated_forward(
        UpdatedForwardInput(gpp_pgC_per_year=290.0), bundle_path=BUNDLE
    )
    samples = result.sample_states
    assert (
        samples["plus_1sigma"]["cap_delta17_prime_permil"]
        < samples["mean"]["cap_delta17_prime_permil"]
        < samples["minus_1sigma"]["cap_delta17_prime_permil"]
    )
    scales = result.forcing_scale_samples
    assert scales["mean"] == pytest.approx(
        2.0 * 1.2539582757549723 * 1.306298723719676
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"p_o2_pal": 0.09}, "pO2"),
        ({"p_o2_pal": 2.01}, "pO2"),
        ({"p_co2_ppm": 49.0}, "pCO2"),
        ({"p_co2_ppm": 60001.0}, "pCO2"),
        ({"gpp_pgC_per_year": 0.0}, "finite and positive"),
        ({"gpp_pgC_per_year": 851.0}, "GPP"),
    ),
)
def test_candidate_engine_rejects_extrapolation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_updated_forward(UpdatedForwardInput(**kwargs), bundle_path=BUNDLE)


def test_crossed_holdout_surface_value_is_preserved_without_normalization() -> None:
    """The stored surface itself retains the independently validated kernel."""

    import json
    import numpy as np

    from global_o2_isotope_reservoir import frozen_photochemical_steady_state
    from local_r7_response_operator import LocalR7ResponseSurface
    from modern_isotope_column import modern_reference_isotope_compositions
    from self_consistent_isotope_fixed_point import solve_mechanistic_fixed_point
    from young_global_o2_budget import (
        GLOBAL_MAJOR_O2_MOLES_1PAL,
        fixed_po2_young_biological_budget,
    )

    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    surface = LocalR7ResponseSurface.from_dict(bundle["surface"])
    reference, _co2 = modern_reference_isotope_compositions()

    def target(state):
        photo = surface.evaluate_prime_tendency_at(
            state,
            po2_pal=0.5,
            pco2_ppm=5000.0,
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        biology = fixed_po2_young_biological_budget(
            po2_pal=0.5, gpp_percent=10.0, photochemical=photo
        )
        equilibrium = frozen_photochemical_steady_state(
            biology, photo, source="stored-surface holdout test"
        )
        return np.asarray(
            [equilibrium.delta18_prime_permil, equilibrium.cap_delta17_prime_permil]
        )

    solved = solve_mechanistic_fixed_point(
        target,
        np.asarray(
            [reference.delta18_prime_permil, reference.cap_delta17_prime_permil]
        ),
        finite_difference_step=0.5,
        tolerance=1.0e-8,
        maximum_iterations=10,
    )
    assert solved.converged
    assert solved.state[1] == pytest.approx(-5.935216904289571, abs=2.0e-9)


def test_slow_newton_node_converges_without_relaxation() -> None:
    result = run_updated_forward(
        UpdatedForwardInput(
            p_o2_pal=0.5,
            p_co2_ppm=24727.711798851356,
            gpp_pgC_per_year=182.56264,
        ),
        bundle_path=BUNDLE,
    )
    assert result.numerically_converged is True
    assert result.maximum_fixed_point_residual_permil <= 1.0e-8
    assert result.central_cap_delta17_prime_permil == pytest.approx(
        -12.45578252, abs=1.0e-8
    )


def test_low_po2_low_gpp_envelope_slow_node_converges() -> None:
    result = run_updated_forward(
        UpdatedForwardInput(
            p_o2_pal=0.2,
            p_co2_ppm=24727.711798851356,
            gpp_pgC_per_year=57.731375805336214,
        ),
        bundle_path=BUNDLE,
    )
    assert result.numerically_converged is True
    assert result.maximum_fixed_point_residual_permil <= 1.0e-8
    assert result.central_cap_delta17_prime_permil == pytest.approx(
        -20.99777789, abs=1.0e-8
    )


def test_hybr_fallback_solves_same_mechanistic_fixed_point() -> None:
    result = run_updated_forward(
        UpdatedForwardInput(
            p_o2_pal=0.31622776601683794,
            p_co2_ppm=15874.507866387554,
            gpp_pgC_per_year=18.256264,
        ),
        bundle_path=BUNDLE,
    )
    assert result.numerically_converged is True
    assert result.maximum_fixed_point_residual_permil <= 1.0e-8
    assert "undamped_newton_then_scipy_hybr" in result.numerical_solver_methods_used
    assert result.central_cap_delta17_prime_permil == pytest.approx(
        -23.93423794016048, abs=1.0e-8
    )
