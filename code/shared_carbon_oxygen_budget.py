"""Carbon exchange coupled to the same biological flux used by global O2.

Young et al. (2014), Table 2 and Eq. 27 supply the carbon coefficients.
The initial-state balance and major-isotopologue-equivalent convention are
explicit in docs/photosynthesis_carbon_budget.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from young_model_inventory import PARAMETERS
from young_reactions import TABLE3_CO2_TROP

SHARED_BUDGET_ID = "shared_carbon_oxygen_v1"
REFERENCE_PCO2_PPM = 294.4
DEFAULT_CARBON_SINK_MULTIPLIER = 1.0


@dataclass(frozen=True)
class SharedCarbonBudget:
    initial_pco2_ppm: float
    initial_biological_o2_tendency: float
    sink_multiplier: float = DEFAULT_CARBON_SINK_MULTIPLIER

    def __post_init__(self):
        if (not np.isfinite(self.initial_pco2_ppm) or self.initial_pco2_ppm <= 0
                or not np.isfinite(self.sink_multiplier) or self.sink_multiplier <= 0
                or not np.isfinite(self.initial_biological_o2_tendency)):
            raise ValueError("Carbon budget requires positive pCO2/multiplier and finite initial flux")

    @property
    def initial_inventory(self):
        trop = TABLE3_CO2_TROP * self.initial_pco2_ppm / REFERENCE_PCO2_PPM
        return np.array([trop, trop * PARAMETERS['k_TS_per_year'] / PARAMETERS['k_ST_per_year']])

    @property
    def uptake_rate(self):
        return self.sink_multiplier * (PARAMETERS['k_CO2_weathering_per_year']
                                      + PARAMETERS['k_ocean_CO2_infusion_per_year'])

    @property
    def constant_sources(self):
        return PARAMETERS['f_volcanic_CO2_mol_per_year'] + PARAMETERS['f_ocean_CO2_mol_per_year']

    @property
    def balancing_flux(self):
        return (self.uptake_rate * self.initial_inventory[0] - self.constant_sources
                + self.initial_biological_o2_tendency)

    def pco2(self, inventory):
        return float(inventory[0] * REFERENCE_PCO2_PPM / TABLE3_CO2_TROP)

    def tendency(self, inventory, biological_o2_tendency):
        trop, strat = inventory
        exchange = PARAMETERS['k_ST_per_year']*strat - PARAMETERS['k_TS_per_year']*trop
        # Algebraically FV+FO+B-K*Ct-RP+R. Centering avoids cancellation
        # of large reference fluxes without changing the signed balance.
        external = self.initial_biological_o2_tendency - self.uptake_rate*(trop-self.initial_inventory[0])
        return np.array([-biological_o2_tendency + external + exchange, -exchange])

    def metadata(self):
        return dict(
            architecture=SHARED_BUDGET_ID,
            sink_multiplier=self.sink_multiplier,
            weathering_per_year=self.sink_multiplier*PARAMETERS['k_CO2_weathering_per_year'],
            ocean_uptake_per_year=self.sink_multiplier*PARAMETERS['k_ocean_CO2_infusion_per_year'],
            volcanic_plus_ocean_source_mol_per_year=self.constant_sources,
            constant_balancing_flux_mol_per_year=self.balancing_flux,
            balancing_reference_CO2_mol=float(self.initial_inventory[0]),
            initial_stratospheric_CO2_mol=float(self.initial_inventory[1]),
            initial_pco2_ppm=self.initial_pco2_ppm,
            inventory_conversion_reference_ppm=REFERENCE_PCO2_PPM,
            inventory_conversion_reference_trop_mol=TABLE3_CO2_TROP,
            initial_biological_o2_tendency_mol_per_year=self.initial_biological_o2_tendency,
            balancing_policy="signed constant closes the entered initial carbon inventory and shared biological flux; held fixed after the step",
            biological_stoichiometry="one carbon equivalent per major-O2 equivalent, using the same production and respiration fluxes",
            geological_o2_policy="balanced slow background held fixed; conditional biological/isotope relaxation, without dynamic geological feedback",
            source="Young et al. (2014), Tables 2-3, Eq. 27 and Section 3.6; doi:10.1016/j.gca.2014.03.026",
        )
