"""Configuration isolation and physical contracts shared by both biospheres."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from biosphere_configurations import configuration, marine_factors, output_surface_path
from forward_isotope_constraints import ForwardIsotopeConstraints, predict_isotopes
from updated_constrained_pco2_posterior import CoordinateConstraint as C
from updated_molecular_forward_model import (
    UpdatedForwardInput, build_updated_central_forcing,
    run_updated_central_state, run_updated_forward,
)
from updated_molecular_transient import UpdatedTransientInput, run_updated_transient
from updated_output_surface import UpdatedOutputSurfaceInput, run_updated_accelerated_forward
from updated_output_surface_inverse import UpdatedSurfaceInverseInput, invert_updated_output_surface
from updated_output_surface_joint_posterior import UpdatedJointPosteriorInput, joint_updated_posterior
from updated_pco2_trajectory_transient import UpdatedPCO2TrajectoryInput, run_updated_pco2_trajectory
from updated_photosynthesis_transient import UpdatedPhotosynthesisTransientInput, run_updated_photosynthesis_transient
from global_o2_isotope_reservoir import GlobalO2Reservoir, sum_tendencies
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


CONFIGURATIONS = ("modern_reference", "marine_endmember")


def test_marine_fractionation_mixes_ratios_and_equilibrates_once():
    factors = marine_factors()
    equilibrium = 1 + (.8746 - .0083*20)/1000
    expected18 = equilibrium*(.95*.978 + .05*.988)
    expected17 = equilibrium**.5284*(.95*.978**.514 + .05*.988**.516)
    assert factors["alpha18"] == pytest.approx(expected18, abs=1e-15)
    assert factors["alpha17"] == pytest.approx(expected17, abs=1e-15)
    assert factors["log_exponent"] == pytest.approx(np.log(expected17)/np.log(expected18))


@pytest.mark.parametrize("name", CONFIGURATIONS)
@pytest.mark.parametrize("point", [(1., 294., 290.), (.3, 20000., 80.), (1.5, 50000., 600.)])
def test_each_configuration_closes_live_isotopologue_budget(name, point):
    request = UpdatedForwardInput(*point, biosphere_configuration=name)
    state = run_updated_central_state(request)
    forcing = build_updated_central_forcing(request)
    photo = forcing.photochemical_tendency(np.array([state.delta18_prime_permil, state.cap_delta17_prime_permil]))
    reservoir = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=request.p_o2_pal*GLOBAL_MAJOR_O2_MOLES_1PAL,
        delta18_prime_permil=state.delta18_prime_permil,
        cap_delta17_prime_permil=state.cap_delta17_prime_permil, source="closure test")
    biology = forcing.biological_budget(photo)
    residual = sum_tendencies((biology.tendency(reservoir), photo), source="closure test").values
    production = request.gpp_pgC_per_year*1e15/12.0107
    # Normalize each isotopologue by its own production scale.
    ratios = reservoir.isotopologue_moles/reservoir.o16o16
    assert np.max(np.abs(residual/(production*ratios))) < 1e-9
    assert state.numerically_converged
    assert state.maximum_fixed_point_residual_permil < 1e-8


@pytest.mark.parametrize("name", CONFIGURATIONS)
@pytest.mark.parametrize("coordinate", ["pCO2", "GPP", "pO2"])
def test_inverse_roundtrip_uses_selected_biology(name, coordinate):
    point = dict(p_o2_pal=.7, p_co2_ppm=5000., gpp_pgC_per_year=180., biosphere_configuration=name)
    target = run_updated_accelerated_forward(UpdatedOutputSurfaceInput(**point))
    result = invert_updated_output_surface(UpdatedSurfaceInverseInput(
        target.central_cap_delta17_prime_permil, solve_for=coordinate, **point))
    expected = {"pCO2":5000., "GPP":180., "pO2":.7}[coordinate]
    assert result.central_root == pytest.approx(expected, rel=1e-8)
    assert result.live_root_verified
    assert result.surface_data_id == target.surface_data_id
    assert abs(result.live_root_residual_permil) <= result.interpolation_guardrail_permil


def test_marine_fixed_biology_does_not_inherit_modern_ensemble():
    result = run_updated_forward(UpdatedForwardInput(biosphere_configuration="marine_endmember"))
    assert not result.biological_pathway_uncertainty_included
    assert result.biological_full_ensemble_member_count == 0
    assert result.biological_process_interval_cap_delta17_permil == (
        result.central_cap_delta17_prime_permil, result.central_cap_delta17_prime_permil)


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_joint_probability_is_normalized_with_selected_surface(name):
    point = UpdatedOutputSurfaceInput(.7, 5000., 180., name)
    target = run_updated_accelerated_forward(point)
    request = UpdatedJointPosteriorInput(
        target_air_cap_delta17_permil=target.central_cap_delta17_prime_permil,
        measurement_sigma_permil=.06, p_o2_pal=.7,
        pco2_bounds_ppm=(3000.,8000.), gpp_bounds_pgC_per_year=(130.,240.),
        pco2_prior="uniform", gpp_prior="uniform",
        pco2_grid_size=41, gpp_grid_size=41, biosphere_configuration=name)
    result = joint_updated_posterior(request)
    assert result.posterior_integral == pytest.approx(1., abs=1e-12)
    assert sum(result.posterior_probability_mass) == pytest.approx(1., abs=1e-12)
    assert result.surface_data_id == target.surface_data_id
    assert result.inputs.biosphere_configuration == name


def test_explicit_foreign_cache_rejected_before_probability_calculation():
    request = UpdatedJointPosteriorInput(target_air_cap_delta17_permil=-.4, measurement_sigma_permil=.06)
    with pytest.raises(ValueError, match="biology differs"):
        joint_updated_posterior(request, surface_path=output_surface_path("marine_endmember"))


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_all_three_forward_uncertainties_use_selected_biology(name):
    request = ForwardIsotopeConstraints(C("range", lower=280, upper=320),
        C("normal", center=290, sigma=10), C("range", lower=.9, upper=1.1), name)
    result = predict_isotopes(request)
    central = run_updated_accelerated_forward(UpdatedOutputSurfaceInput(biosphere_configuration=name))
    interval = result["isotopes"]["cap_delta17_prime_permil"]["interval95"]
    assert interval[0] < central.central_cap_delta17_prime_permil < interval[1]


@pytest.mark.parametrize("name", CONFIGURATIONS)
@pytest.mark.parametrize("change", [dict(p_co2_ppm=1000.), dict(gpp_pgC_per_year=145.), dict(p_o2_pal=.5), {}])
def test_steps_preserve_initial_ratios_and_converge_to_same_config_equilibrium(name, change):
    initial = UpdatedForwardInput(biosphere_configuration=name)
    final = replace(initial, **change)
    result = run_updated_transient(UpdatedTransientInput(initial, final,
        duration_years=50000., sample_count=21, equilibrium_search_max_years=50000.))
    start, end = result.states[0], result.states[-1]
    assert start.cap_delta17_prime_permil == pytest.approx(result.initial_steady_state["cap_delta17_prime_permil"], abs=1e-12)
    assert end.cap_delta17_prime_permil == pytest.approx(result.final_steady_state["cap_delta17_prime_permil"], abs=8e-4)
    assert end.delta18_prime_permil == pytest.approx(result.final_steady_state["delta18_prime_permil"], abs=2e-4)
    if not change:
        assert max(abs(s.cap_delta17_prime_permil-start.cap_delta17_prime_permil) for s in result.states) < 1e-7


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_gradual_co2_reaches_configuration_specific_equilibrium(name):
    result = run_updated_pco2_trajectory(UpdatedPCO2TrajectoryInput(
        initial=UpdatedForwardInput(biosphere_configuration=name), final_pco2_ppm=422.8,
        transition_duration_years=174., duration_years=50000., sample_count=21,
        equilibrium_search_max_years=50000.))
    assert result.states[-1].cap_delta17_prime_permil == pytest.approx(result.final_steady_state["cap_delta17_prime_permil"], abs=8e-4)
    assert result.states[-1].delta18_prime_permil == pytest.approx(result.final_steady_state["delta18_prime_permil"], abs=2e-4)


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_photosynthesis_step_is_continuous_and_has_positive_inventories(name):
    initial = UpdatedForwardInput(biosphere_configuration=name)
    result = run_updated_photosynthesis_transient(UpdatedPhotosynthesisTransientInput(
        initial=initial, photosynthesis_fraction=.5, duration_years=22000., sample_count=101,
        equilibrium_search_max_years=50000.))
    live = run_updated_central_state(initial)
    assert result.states[0].cap_delta17_prime_permil == pytest.approx(live.cap_delta17_prime_permil, abs=1e-12)
    assert all(np.all(s.isotopologue_moles > 0) for s in result.states)
    rows = result.rows()
    assert min(r["pO2_pal"] for r in rows) < .9
    assert max(r["pCO2_ppm"] for r in rows) > 500
    assert max(abs(r["pO2_pal"]-p) for r,p in zip(rows,result.carbon_driver_po2_pal)) < .01


def test_changing_biosphere_within_state_step_requires_separate_experiment():
    with pytest.raises(ValueError, match="biosphere"):
        run_updated_transient(UpdatedTransientInput(UpdatedForwardInput(),
            UpdatedForwardInput(biosphere_configuration="marine_endmember")))


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_service_provenance_names_actual_surface_and_budget(name):
    from public_model_service import forward
    result = forward(UpdatedForwardInput(biosphere_configuration=name))
    assert result["provenance"]["accelerated_surface_data_id"] == result["result"]["surface_data_id"]
    assert result["provenance"]["biosphere_configuration"] == name
    assert result["provenance"]["biosphere_version"] == configuration(name)["version"]


@pytest.mark.parametrize("name", CONFIGURATIONS)
@pytest.mark.parametrize("coordinate,truth,bounds", [
    ("pCO2",10000.,(5000.,15000.)), ("GPP",72.5,(40.,110.)), ("pO2",.5,(.25,.75))])
def test_sulfate_posterior_recovers_each_coordinate_for_selected_biology(name, coordinate, truth, bounds):
    from sulfate_to_air import IncorporationConstraint, SulfateProcessAssumptions, exact_sulfate_from_air
    from sulfate_uncertainty import BackgroundConstraint, SulfateLikelihoodInput
    state = run_updated_accelerated_forward(UpdatedOutputSurfaceInput(.5,10000.,72.5,name))
    observed = exact_sulfate_from_air(
        air_cap_delta17_permil=state.central_cap_delta17_prime_permil,
        air_delta18_permil=1000*np.expm1(state.central_delta18_prime_permil/1000),
        sulfate_delta18_permil=15., fraction=.2,
        process=SulfateProcessAssumptions(-.02,1.,.528,"synthetic recovery"))
    sulfate = SulfateLikelihoodInput(
        measured_cap_delta17_permil=observed, measured_delta18_permil=15.,
        cap_delta17_sigma_permil=.03, delta18_sigma_permil=.5, isotope_error_correlation=0.,
        incorporation=IncorporationConstraint("fixed",center=.2),
        background=BackgroundConstraint("fixed",center=-.02),
        alpha18_air_to_sulfate=1., theta_air_to_sulfate=.528, assumption_note="synthetic recovery")
    bounds_key = {"pCO2":"pco2_bounds_ppm", "GPP":"gpp_bounds_pgC_per_year", "pO2":"po2_bounds_pal"}[coordinate]
    result = joint_updated_posterior(UpdatedJointPosteriorInput(
        sulfate=sulfate, free_coordinates=(coordinate,), p_o2_pal=.5, p_co2_ppm=10000.,
        gpp_pgC_per_year=72.5, pco2_prior="uniform",gpp_prior="uniform",po2_prior="uniform",
        pco2_grid_size=81,gpp_grid_size=81,po2_grid_size=81,biosphere_configuration=name,
        **{bounds_key:bounds}))
    low, high = result.equal_tailed_credible_intervals[coordinate]
    assert low < truth < high
    assert result.sulfate_likelihood_diagnostics["status"] == "converged"
    assert result.posterior_integral == pytest.approx(1.)


@pytest.mark.parametrize("name", CONFIGURATIONS)
@pytest.mark.parametrize("coordinate,truth", [("pCO2",5000.),("GPP",180.),("pO2",.7)])
def test_constrained_solver_keeps_configuration_through_refinement(name, coordinate, truth):
    from updated_constrained_pco2_posterior import ConstrainedCoordinateInput, constrained_coordinate_posterior
    point = UpdatedOutputSurfaceInput(.7,5000.,180.,name)
    target = run_updated_accelerated_forward(point).central_cap_delta17_prime_permil
    constraints = {"pCO2":C("range",lower=4000.,upper=6000.),
        "GPP":C("normal",center=180.,sigma=5.),"pO2":C("range",lower=.6,upper=.8)}
    del constraints[coordinate]
    result = constrained_coordinate_posterior(ConstrainedCoordinateInput(
        solve_for=coordinate, target_air_cap_delta17_permil=target, measurement_sigma_permil=.06,
        constraints=constraints, biosphere_configuration=name))
    assert result.equal_tailed_credible_interval[0] < truth < result.equal_tailed_credible_interval[1]
    assert result.inputs.biosphere_configuration == name
    assert sum(result.solve_marginal_probability_mass) == pytest.approx(1.,abs=1e-10)
