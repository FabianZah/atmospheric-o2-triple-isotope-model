# Structural uncertainty

OXYTIB reports measurement/proxy, parameter, numerical and structural
uncertainty as distinct layers. Their machine-readable definitions are in
`model_data/uncertainty/updated_o2_uncertainty_layers_v1.json`.
The model-comparison classification is
`model_data/literature/multimodel_evidence_matrix_v1.json`.

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
Its excess isotope-coordinate scatter is approximately 0.01324 per mil.
This includes possible GPP/pO2 variability, chronology and reference-coordinate
effects. It is recorded separately from the public measurement likelihood.
Brandon-event residuals similarly describe that event and its assumptions.

## Climate and oxygen-accessibility sensitivity

Cloud-free Clima calculations assess sensitivity to thermal structure and
ozone chemistry at high pCO2. Additive-CO2 and fixed-total-dry-gas pressure
conventions differ by at most 0.0404 per mil through 30,000 ppm and
0.2270 per mil across the tested domain. The larger fixed-profile versus
Clima difference reaches several per mil.

Exact end-member calculations at 0.1, 1 and 2 PAL demonstrate a material
pO2-climate-ozone interaction. These scenarios retain their own physical
assumptions and are reported as non-probabilistic sensitivity evidence.
The modern ozone-heated radiative-convective equilibrium (RCE) check and the high-pCO2 comparisons are summarized
in [the validation assessment](publication_model_acceptance.md).

Liu et al. (2021)'s atmosphere-accessible marine oxygen convention is another
structural sensitivity. The test improves the Liu response comparison but
weakens Young Fig. 8 and Yang ice-core comparisons. OXYTIB therefore uses
total global GPP in its central biological budget.
See [marine oxygen accessibility](marine_o2_accessibility_audit.md).

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
