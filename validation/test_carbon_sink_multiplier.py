"""Carbon sink scaling is explicit and separate from reference closure."""
from unittest.mock import patch

import numpy as np
import pytest

from calibrated_model import build_reactions
from model_scenarios import CURRENT_PUBLICATION_PRESET, ScenarioInput, config_from_scenario
from updated_molecular_forward_model import UpdatedForwardInput
import updated_photosynthesis_transient as transient
from young_model_inventory import PARAMETERS
from young_reactions import TABLE3_CO2_TROP


@pytest.mark.parametrize("factor", [.94, 1.0])
def test_rates_and_separate_balancing_flux(factor):
    config = config_from_scenario(ScenarioInput(
        preset=CURRENT_PUBLICATION_PRESET, solve_mode="full_atmosphere", co2_sink_factor=factor))
    rates = {r.key:r.rate_constant for r in build_reactions(config) if hasattr(r, "rate_constant")}
    assert config.co2_ocean_infusion_factor == 1.
    assert rates["CO2_weathering"] == pytest.approx(factor*.000208, rel=1e-14)
    assert rates["CO2_ocean_infusion"] == pytest.approx(factor*.1172, rel=1e-14)
    source = PARAMETERS["f_volcanic_CO2_mol_per_year"] + PARAMETERS["f_ocean_CO2_mol_per_year"]
    assert rates["CO2_source"] == source
    expected_balance = factor*(.000208+.1172)*TABLE3_CO2_TROP-source
    assert rates["co2_modern_closure"] == pytest.approx(expected_balance, abs=2.)
    assert abs(source + rates["co2_modern_closure"]
               - (rates["CO2_weathering"]+rates["CO2_ocean_infusion"])*TABLE3_CO2_TROP) <= 2.
    # Isotopologue kinetics receive the same scaling; no isotope-only tuning.
    for heavy in ("CO18O", "CO17O"):
        assert rates[heavy+"_weathering"] == rates["CO2_weathering"]
        assert rates[heavy+"_ocean_infusion"] == rates["CO2_ocean_infusion"]


def test_current_default_is_separate_from_historical_preset():
    request = transient.UpdatedPhotosynthesisTransientInput(UpdatedForwardInput())
    assert request.carbon_sink_multiplier == 1.
    historical = config_from_scenario(ScenarioInput(preset=CURRENT_PUBLICATION_PRESET))
    assert historical.co2_sink_factor == .94


@pytest.mark.parametrize("factor", [0., -1., np.inf, np.nan])
def test_invalid_multiplier_rejected_before_calculation(factor):
    request = transient.UpdatedPhotosynthesisTransientInput(
        UpdatedForwardInput(), carbon_sink_multiplier=factor)
    with patch.object(transient, "run_updated_central_state") as solve:
        with pytest.raises(ValueError, match="carbon sink multiplier"):
            transient.run_updated_photosynthesis_transient(request)
        solve.assert_not_called()


@pytest.mark.parametrize("factor", [.94, 1.])
def test_multiplier_reaches_shared_budget_without_mutating_preset(factor):
    request = transient.UpdatedPhotosynthesisTransientInput(
        UpdatedForwardInput(), carbon_sink_multiplier=factor)
    class Captured(Exception):
        pass
    budgets = []
    original=transient.SharedCarbonBudget
    def capture(*args, **kwargs):
        budgets.append(original(*args, **kwargs))
        raise Captured
    with patch.object(transient, "SharedCarbonBudget", capture):
        with pytest.raises(Captured):
            transient.run_updated_photosynthesis_transient(request)
    assert budgets[0].sink_multiplier == factor
    assert budgets[0].initial_pco2_ppm == 294.
    historical = config_from_scenario(ScenarioInput(preset=CURRENT_PUBLICATION_PRESET))
    assert historical.co2_sink_factor == .94


@pytest.mark.parametrize("factor", [.94,1.])
def test_no_perturbation_is_stationary_and_keeps_the_requested_time_grid(factor):
    from integrate_fig9_fig10_transients import integrate_photosynthesis_carbon_driver
    config = config_from_scenario(ScenarioInput(solve_mode="full_atmosphere",co2_sink_factor=factor))
    result = integrate_photosynthesis_carbon_driver(config,
        photosynthesis_fraction=1.,duration_years=100000.,sample_count=41)
    assert result.solver.success and result.solver.nfev == 0
    assert len(result.time_years) == 41 and result.time_years[-1] == 100000.
    assert np.all(result.pco2_ppm == config.p_co2_ppm)
    assert np.all(result.po2_pal == config.p_o2_pal)
