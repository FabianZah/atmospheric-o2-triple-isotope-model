# Structural uncertainty

OXYTIB reports measurement/proxy, parameter, numerical and structural
uncertainty as distinct layers. Their machine-readable definitions are in
`model_data/uncertainty/updated_o2_uncertainty_layers_v1.json`.
The model-comparison classification is
`model_data/literature/multimodel_evidence_matrix_20261019.json`.

## Statistical interpretation

Measurement uncertainty and explicitly specified coordinate constraints enter
the likelihood and priors. Alternatives for the CO2 isotope-anomaly flux
(isoflux) and biological processes define parameter sensitivities. Direct
model calculations withheld from the interpolation grid quantify numerical error.

Differences between atmospheric models quantify structural sensitivity.
They are not repeated measurements of model error and do not define a
whole-domain Gaussian standard deviation. The public posterior is conditional
on the declared physical model and entered constraints.

The Yang-Banerjee blocked-validation analysis provides a domain-limited
predictive assessment at low pCO2, 1 PAL and specified GPP assumptions.
Its excess isotope-coordinate scatter is recorded in the current
`yang_lowco2_predictive_error.json` evidence report.
This includes possible GPP/pO2 variability, chronology and reference-coordinate
effects. It is recorded separately from the public measurement likelihood.
Brandon-event residuals similarly describe that event and its assumptions.

## Climate and oxygen-accessibility sensitivity

Archived cloud-free Clima calculations assess thermal structure, ozone
chemistry and pressure conventions using the earlier atmospheric chemistry.
Their offsets remain historical structural experiments and are excluded from
the corrected core's numerical uncertainty layer. A current climate sensitivity
would require consistent regeneration of the atmospheric states and isotope
coefficients. This is deferred; an unquantified contribution is not zero error.

Liu et al. (2021)'s atmosphere-accessible marine oxygen convention is another
structural sensitivity. Its current-core comparison is reported separately in
`model_data/validation_evidence/corrected_core_20261019/marine_o2_accessibility_audit.json`.
Both supported biological configurations retain full atmospheric accessibility.
The fixed marine endmember changes isotope sources and respiratory effects;
it does not introduce an ocean gas-transfer model.

## High-pCO2 and low-GPP interpretation

Young et al. (2014), Liu et al. (2021), Cao and Bao (2013), and OXYTIB
provide complementary constraints on response direction and amplitude.
Their differences involve oxygen turnover, photochemical source strength,
vertical structure, gas exchange, respiratory recycling and O(1D) competition.

The operational domain and physical-domain mask remain explicit.
Structural envelopes qualify predictions in the high-pCO2/low-GPP region;
they are not used as empirical damping factors or converted automatically
to posterior confidence. Nonzero Gaussian discrepancy in the advanced
joint engine requires a stated source and application domain.

The [evidence bundle](validation_evidence_bundle.md) records the numerical
results and provenance supporting these distinctions.
