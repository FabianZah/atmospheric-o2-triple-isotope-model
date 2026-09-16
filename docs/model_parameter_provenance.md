# Model parameter provenance

OXYTIB parameters are defined by the scientific implementation and versioned
runtime data. The authoritative entry points and checksums are in
[the model contract](../model_data/publication_model_contract_v1.json).

| Component | Definition and source |
|---|---|
| Photochemical response | R1-R7 column chemistry and its response operators in `code/local_r7_response_operator.py`, `code/r7_rate_sources.py` and `model_data/updated_r7_response_surface_v1.json`. Source conventions are retained with the operators. |
| Molecular oxygen balance | `code/global_o2_isotope_reservoir.py` and `code/updated_molecular_forward_model.py`; the CO2-to-O2 molecular balance follows Liang et al. (2023). |
| Forcing normalization | ERA5/native-column geometry and Adnew et al. (2025) CO2 isotope-isoflux normalization, recorded in the response-surface metadata. |
| Biological sources and respiration | `code/biological_o2_ensemble.py`; land-ocean production, respiratory pathways and source-water compositions have explicit literature alternatives. |
| Modern GPP | `code/gpp_normalization.py`; the interface defines 100% modern as 290 Pg C per year, following Liang et al. (2023). |
| Modern isotope observations | Pack (2021): atmospheric O2 Δ′¹⁷O = -0.432 ± 0.015 per mil and conventional δ¹⁸O = 23.9 ± 0.3 per mil. These are observational checks, distinct from calculated equilibrium values. |
| Spherule transfer | `code/spherule_to_air_d17o.py`; Zahnow et al. (2025), Eq. 3, with measured spherule δ¹⁸O. |
| Sulfate transfer | `code/sulfate_to_air.py`; measured sulfate isotopes, specified incorporation and non-air constraints, and explicitly selected fractionation treatment. See [sulfate transfer](sulfate_transfer.md). |
| Numerical interpolation | Versioned output nodes in `model_data/updated_molecular_output_surface_v1.json`, validated against the physical kernel. |
| Uncertainty | [The uncertainty contract](../model_data/uncertainty/updated_o2_uncertainty_layers_v1.json) separates measurement, parameter, numerical and structural terms. |

The default reference calculation is 1 PAL O2, 294 ppm CO2 and 290 Pg C per
year GPP. Its unadjusted outputs and the comparison observations are stored
separately in the model contract. User-defined pO2, pCO2 and GPP are scenario
boundary conditions.

The historical Young et al. (2014) calculation is a validation reference.
Its numerical conventions are confined to the corresponding implementation
and tests; the OXYTIB contract defines the model used by the API and console.
