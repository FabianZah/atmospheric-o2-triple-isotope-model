# Joint posterior inference

The updated model supports joint inference over any two or all three of pCO2,
absolute GPP, and pO2. A single atmospheric O2 Delta-prime-17O observation
usually defines a curved solution ridge, not a unique point. The joint engine
therefore retains the multidimensional posterior and reports marginal summaries
without implying that the isotope measurement independently identifies all
three coordinates.

For parameters `theta` and measured isotope value `y`, the implemented
likelihood is

```text
p(y | theta) proportional to
exp[-0.5 * ((model(theta) - y) / sigma_effective)^2]

sigma_effective^2 = sigma_measurement^2 + sigma_model_discrepancy^2
```

The coordinate priors are independent, bounded, and declared explicitly as
uniform or log-uniform. Numerical integration uses trapezoid cell volumes in
the physical coordinates, so posterior probability does not depend on whether
an axis was evaluated on a linear or geometric grid.

## Model-discrepancy policy

Analytical measurement uncertainty is required. A Gaussian model-discrepancy
term is optional, but a positive value is rejected unless its provenance is
provided. The source-isoflux, biological-process, and interpolation intervals
stored in the output surface remain non-probabilistic guardrails. They are not
silently interpreted as one-sigma ranges or assigned arbitrary distributions.

The release scorecard uses a synthetic discrepancy value only to test numerical
propagation and generated-target recovery. It is not an empirical calibration.
A public default requires an independently justified discrepancy model, ideally
estimated from withheld observations while accounting for uncertainty in GPP,
pO2, chronology, and isotope reference frames. Data used to tune that
discrepancy must not also be presented as independent validation.

The frozen eligibility assessment is
`model_data/literature/multimodel_evidence_matrix_v1.json`; its rationale is
documented in `docs/structural_uncertainty_policy.md`. Young, Liu, and Cao-Bao
model differences may define structural sensitivity envelopes but may not be
entered as a Gaussian sigma. The Yang-Banerjee residual is eligible only for a
low-pCO2, 1 PAL, fixed-GPP predictive-error model, and the Brandon residual is
event specific. No whole-domain empirical discrepancy is currently assigned.

## Reported quantities

The engine returns:

- the full two- or three-dimensional posterior density and probability mass;
- the highest-posterior-density region for the requested credible mass;
- marginal densities, means, medians, and equal-tailed credible intervals;
- the joint maximum-a-posteriori coordinates;
- posterior mass near each declared prior boundary;
- the complete probability scope and model-data identifiers.

Boundary-sensitive marginals must be reported as prior-bound dependent. A
maximum-a-posteriori point is a summary of the ridge and must not replace the
joint solution surface.

## Implementation and validation

The implementation is `code/updated_output_surface_joint_posterior.py`. Tests
cover vectorized/scalar surface agreement, two- and three-coordinate
normalization, generated-target containment, discrepancy propagation, and the
provenance requirement. The unified release scorecard records joint recovery
as a formal numerical gate and empirical discrepancy calibration as a separate
remaining limitation.
