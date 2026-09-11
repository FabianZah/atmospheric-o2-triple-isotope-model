# High-pCO2 model comparisons

OXYTIB and the Young et al. (2014) Fig. 8 response differ in isotope amplitude
at high pCO2, especially at low GPP. This is evaluated as structural
model disagreement alongside Liu et al. (2021) and Cao and Bao (2013).

## Mechanistic interpretation

The photochemical response includes converged parent atmospheres through
60,000 ppm. As CO2 increases, R7 competes with other O(1D) sinks. Its
forcing elasticity decreases smoothly as the available O(1D) is shared
among those sinks. This provides a physical source of curvature.

Atmospheric temperature, vertical transport, pressure convention and
oxygen-accessible biological production can also alter the response.
OXYTIB retains prescribed column structure and a total-global-GPP budget.
The Clima and marine-accessibility calculations quantify alternatives to
those assumptions.

A difference from one published curve does not uniquely identify an error
in either model. Own-reference GPP comparisons and common-absolute-GPP
comparisons answer different questions and are kept separate.

## Evidence

The versioned evidence includes:

- `model_data/validation_evidence/updated_molecular_release_scorecard.json`:
  numerical gates and the Young response comparison.
- `model_data/validation_evidence/liu_2021_low_gpp_multimodel_benchmark.json`:
  low-GPP response topology and amplitude.
- `model_data/validation_evidence/cao_bao_2013_multimodel_benchmark.json`:
  high-pCO2 response direction and oxygen dependence.
- Clima and marine-accessibility records in the same evidence bundle:
  thermal, pressure and biological-boundary sensitivity.

## Model scope

OXYTIB uses its physical response throughout the declared 50-60,000 ppm
domain. Structural sensitivity is retained explicitly rather than fitted
away to reproduce a particular curve. The strongest qualifications apply
to high pCO2 and low GPP, where independent direct observations are sparse.

See [the integrated assessment](publication_model_acceptance.md) and
[structural uncertainty](structural_uncertainty_policy.md) for the statistical
interpretation and the role of each comparison.
