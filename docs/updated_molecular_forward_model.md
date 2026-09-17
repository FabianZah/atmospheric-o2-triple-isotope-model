# OXYTIB physical model and numerical evaluation

## Physical calculation

The model couples altitude-resolved oxygen, ozone and carbon-dioxide
photochemistry to a globally mixed atmospheric O2 reservoir with biological
oxygen production and respiration.
The fast photochemical response and slow isotope budget are solved
self-consistently for atmospheric Δ′¹⁷O and δ′¹⁸O.

The forcing normalization combines ERA5 atmospheric reanalysis and the
altitude-resolved model geometry, the Adnew et al. (2025) CO2 isotope-isoflux
observation (CO2 flux weighted by its isotope anomaly), and the molecular two-site oxygen
balance following Liang et al. (2023). Biological production and fractionation
are specified from land-ocean and respiratory-pathway literature constraints.
The raw mechanistic solution is the primary model output.

The implementation is `code/updated_molecular_forward_model.py`. Its entry
points, domain and data identities are fixed by
[the model contract](../model_data/publication_model_contract_v1.json).
[Parameter provenance](model_parameter_provenance.md) identifies the defining
files and literature roles.

## Reference state

At 1 PAL O2, 294 ppm CO2 and 290 Pg C per year GPP, the contract records:

| Quantity | Model | Modern observational reference, Pack (2021) |
|---|---:|---:|
| Atmospheric O2 Δ′¹⁷O, per mil | -0.426353 | -0.432 ± 0.015 |
| Atmospheric O2 δ′¹⁸O, per mil | 23.404329 | Compare after conversion to conventional notation |
| Atmospheric O2 conventional δ¹⁸O, per mil | 23.680359 | 23.9 ± 0.3 |

Scenario values are calculated using their specified boundary conditions.
The modern residual quantifies the comparison with independent observations;
scenario outputs retain the raw model prediction.

## Numerical surface

`model_data/updated_r7_response_surface_v1.json` contains the photochemical
response operators. `model_data/updated_molecular_output_surface_v1.json`
contains direct physical-model solutions used for faster evaluation by interpolation.
The declared domain is 0.1-2 PAL pO2, 50-60,000 ppm pCO2 and
18.256264-850 Pg C per year GPP.

The precomputed model grid (the output surface) uses separate interpolation representations for Δ′¹⁷O,
δ′¹⁸O and the positive distances to uncertainty-envelope bounds. Independent
tests at points withheld from this grid, together with curve-shape tests,
quantify interpolation error, interval ordering and
monotonicity. The contract forbids numerical evaluation outside its domain.

The browser, API and console share this surface and model service. Direct
physical-model calculations are available from the Python implementation;
grid interpolation evaluates the same physical formulation more quickly.

## Inference and uncertainty

[Constrained inference](constrained_coordinate_inference.md) combines measured
isotopes with fixed, Gaussian or bounded-uniform constraints on the other
coordinates. Measurement/proxy, parameter, numerical and structural uncertainty
have separate definitions. Deterministic literature envelopes are reported
as sensitivity bounds; credible intervals depend on explicit likelihoods
and priors. See [structural uncertainty](structural_uncertainty_policy.md).

## Time response

The atmospheric state-step solver evolves scaled O2 isotopologue inventories
using DOP853. pCO2 and GPP changes alter forcing while isotope ratios remain
continuous at the step. A pO2 step rescales isotopologue inventories together.
Final equilibrium is evaluated independently.

The photosynthesis-at-fixed-respiration experiment couples the carbon-box
trajectory to the global molecular O2 budget through operator splitting.
Its scope is a prescribed-forcing sensitivity experiment, rather than a
fully simultaneous carbon-oxygen climate model. Gradual pCO2 trajectories
use user-specified endpoints, transition duration and interpolation.

## Reproduction

Run a forward case and the integrated validation:

```powershell
python run_model.py calculate forward --po2 1 --pco2 294 --gpp 290
python run_model.py validate
```

The direct-model, grid-interpolation, inverse and time-response regression tests are
listed in [SETUP.md](../SETUP.md). Compact scientific evidence and provenance
are stored in `model_data/validation_evidence/`. The
[validation assessment](publication_model_acceptance.md) distinguishes
formal checks, published-model comparisons and scope limits.
