# Model parameter provenance

OXYTIB parameters are defined by the scientific implementation and versioned
runtime data. The authoritative entry points and checksums are in
[the model contract](../model_data/publication_model_contract_v1.json).

| Component | Definition and source |
|---|---|
| Photochemical response | Altitude-resolved oxygen, ozone and carbon-dioxide chemistry, including oxygen-isotope exchange between CO2 and excited atomic oxygen, O(1D). Response calculations are in `code/local_r7_response_operator.py`, `code/r7_rate_sources.py` and `model_data/updated_r7_response_surface_v1.json`. The R7 label in these identifiers follows Young et al. (2014)'s reaction numbering. |
| Molecular oxygen balance | `code/global_o2_isotope_reservoir.py` and `code/updated_molecular_forward_model.py`; the CO2-to-O2 molecular balance follows Liang et al. (2023). |
| Forcing normalization | ERA5 atmospheric reanalysis and altitude-resolved model geometry, with the Adnew et al. (2025) CO2 isotope-isoflux constraint (CO2 flux weighted by its isotope anomaly), recorded in the response-surface metadata. |
| Biological sources and respiration | `code/biological_o2_ensemble.py`; land-ocean production, respiratory pathways and source-water compositions have explicit literature alternatives. |
| Fixed optional marine biology | `model_data/biosphere_configurations_v1.json` and `code/biosphere_configurations.py`; source-qualified surface/interior respiration, reference seawater and dissolved/air equilibrium at 20 C. Ratios are mixed before calculating the effective exponent. |
| Modern production reference | `code/gpp_normalization.py`; the interface defines 100% modern as 290 Pg C-equivalent per year, following the isotope-budget convention. The dominant-O2 production flux is G*1e15/12.0107 mol/year; external carbon GPP/NPP estimates require explicit source-consistent conversion. |
| Photosynthesis experiment carbon sinks | Young et al. (2014) weathering 0.000208 yr^-1 and ocean uptake 0.1172 yr^-1, applied with multiplier 1.00. The constant reference balancing flux is calculated separately. See [carbon budget](photosynthesis_carbon_budget.md). |
| Modern isotope observations | Pack (2021): atmospheric O2 Δ′¹⁷O = -0.432 ± 0.015 per mil and conventional δ¹⁸O = 23.9 ± 0.3 per mil. These are observational checks, distinct from calculated equilibrium values. |
| Spherule transfer | `code/spherule_to_air_d17o.py`; Zahnow et al. (2025), Eq. 3, with measured spherule δ¹⁸O. |
| Sulfate transfer | `code/sulfate_to_air.py`; measured sulfate isotopes, specified incorporation and non-air constraints, and explicitly selected fractionation treatment. See [sulfate transfer](sulfate_transfer.md). |
| Numerical interpolation | Configuration-specific grid values in `model_data/updated_molecular_output_surface_v1.json` and `model_data/marine_molecular_output_surface_v1.json`, validated against their respective direct physical-model calculations. |
| Uncertainty | [The uncertainty contract](../model_data/uncertainty/updated_o2_uncertainty_layers_v1.json) separates measurement, parameter, numerical and structural terms. |

The default reference calculation is 1 PAL O2, 294 ppm CO2 and 290 Pg C per
year GPP. Its unadjusted outputs and the comparison observations are stored
separately in the model contract. User-defined pO2, pCO2 and GPP are scenario
boundary conditions.

CO2 removal of excited atomic oxygen is counted only in the complete CO2
reaction family, including exchange and nonexchange channels. Non-exchange
N2/O2 quenching is separate. The original 400 ppm isoflux normalization gives
the corrected forcing scale 3.2726126929006267. Original imported dataset
identities and metadata revisions are recorded in
`model_data/model_input_provenance_v1.json`.

The historical Young et al. (2014) calculation is a validation reference.
Its numerical conventions are confined to the corresponding implementation
and tests; the OXYTIB contract defines the model used by the API and console.
