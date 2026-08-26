# Updated molecular forward model

Status: **molecular forward engine used by the public UI; physical shape and accelerator release gates passed**.

## Architecture

The updated engine couples a reduced native-Photochem R7 response surface to
one globally mixed O2 reservoir. The promoted response surface spans 0.1-2 PAL
O2 and 50-60,000 ppm CO2. Low-CO2 training operators use native columns at
50, 100, 150, 200, and 250 ppm; 75, 125, 175, 225, and 275 ppm columns remain independent
holdouts. An independently calculated 294 ppm operator anchors the modern
interval. The model solves both delta-prime-18O and
Delta-prime-17O as a self-consistent fast-slow fixed point.

The measured Adnew et al. (2025) molecular CO2 anomaly isoflux is translated
to O2 with an equal-and-opposite material balance. The native-surface forcing
normalization is the product of three named quantities:

1. ERA5 2-D/native-column geometry ratio: `1.253958`.
2. Adnew/ERA5 mean isoflux ratio: `1.306299`.
3. Molecular two-site balance: `2`, following Liang et al. (2023), Eq. 3.

The resulting mean native-surface scale is `3.276088`. This is not an empirical
fit to atmospheric O2. No O2 output offset or damping factor is applied.

## Modern state and uncertainty

At 1 PAL O2, 294 ppm CO2, and 290 PgC yr-1 GPP, the model gives atmospheric O2
Delta-prime-17O = `-0.426350 per mil`. Its residual from Pack (2021),
`-0.432 +/- 0.015 per mil`, is `+0.005650 per mil`, smaller than the stated
observational uncertainty.

The same state gives Delta-prime-18O = `23.404329 per mil`, equivalent to
conventional delta-18O = `23.680359 per mil`. The conventional value differs
from Pack's modern `23.9 +/- 0.3 per mil` summary by `-0.219641 per mil` and is
therefore within the reported interval. Prime and conventional notation are
converted before comparison.

The central biological convention uses Liang et al. (2023)'s central global
land-ocean partition, the Barkan-Luz marine source as interpreted by Young et
al. (2014), proportional allocation of Young's printed COX/AOX/photorespiration
fractions, and the midpoint of Bender et al. (1994)'s terrestrial-source
interval. None of these values is selected by fitting atmospheric O2.

The current model guardrail propagates:

- Adnew's `+/-1.6 per mil PgC yr-1` source-isoflux uncertainty;
- a source-backed biological-process envelope spanning 54 deterministic
  literature corners for land-ocean production, respiration pathways, marine
  source convention, and terrestrial source-water composition;
- the maximum crossed-holdout Delta-prime-17O interpolation residual,
  `0.001667 per mil`.

Seven fixed biological members reproduce the full 54-member envelope over the
current output-grid audit spanning all three Adnew forcing samples, 0.1-2 PAL
O2, 50-60,000 ppm CO2, and 18.256-850 PgC yr-1 GPP. The denser audit identified
two extrema that were absent from the earlier five-member, 81-scenario check.
The envelope is a guardrail over cited alternatives, not a probability
distribution.

Pack's uncertainty is kept separate and is used only to test overlap between
the model and observation intervals. The resulting guardrail is not a formal
posterior. Uncertainty in the numerical reference assigned to 100% modern GPP
remains separate because it applies only when GPP is supplied in
relative-modern units; an absolute GPP input has no normalization ambiguity.

The relative-GPP wrapper defaults to Liang et al. (2023): 100% modern is
`290 +/- 30 PgC yr-1`. It calls the unchanged absolute-GPP kernel at 260, 290,
and 320 PgC yr-1. At 1 PAL and 294 ppm, this expands the combined model
guardrail from `-0.447072` to `-0.403653 per mil` at fixed 290 PgC yr-1 to
`-0.464447` to `-0.390335 per mil` when the GPP-reference interval is included.
Young and Beerling normalizations remain deterministic conventions unless a
custom uncertainty is supplied.

## pCO2 inversion

The companion inverse API solves pCO2 at fixed pO2 and absolute GPP. It reports
two different results:

- a central root where the mean-forcing model equals the measured central
  Delta-prime-17O value;
- an admissible pCO2 interval containing every point where the measurement
  interval overlaps the model guardrail.

For Pack's modern `-0.432 +/- 0.015 per mil` value at 1 PAL and
290 PgC yr-1, the central root is about `305 ppm`, and the admissible range is
`235-384 ppm`. The former 294 ppm lower truncation is removed by the physical
low-CO2 extension. The interval is not a formal confidence or posterior
interval.

If the same case is entered as 100% of Liang et al. (2023)'s modern reference,
its combined reference-uncertainty guardrail is `211-424 ppm`. All three GPP
normalization samples now have interior central roots and admissible intervals.

The inverse is intentionally not pooled with the legacy compact solver. For
example, the following central roots use identical fixed physical inputs:

| Air O2 Delta-prime-17O | pO2 | GPP | Updated molecular model | Current compact model |
|---:|---:|---:|---:|---:|
| -10 per mil | 1 PAL | 25% Young | 9,679 ppm | 8,094 ppm |
| -5 per mil | 0.5 PAL | 50% Young | 6,129 ppm | 5,089 ppm |
| -0.432 per mil | 1 PAL | 100% Young | 385 ppm | 297 ppm |

These differences are architecture sensitivity, not analytical uncertainty.
The modern 290 PgC yr-1 case discussed above is distinct from 100% of Young's
365.126 PgC yr-1 normalization.

## Conditional probability inference

`updated_output_surface_posterior.py` provides a separate probability layer for
one solved coordinate while the other two physical coordinates remain fixed.
It uses a Gaussian analytical-measurement likelihood and requires an explicit
bounded uniform or log-uniform prior. It reports the mode, mean, median,
equal-tailed credible interval, and probability accumulated near either prior
boundary.

The posterior is conditional on the central updated model. Literature process
alternatives and interpolation guardrails are deliberately not converted into
Gaussian errors because they are not probability distributions. The existing
non-probabilistic admissible interval is returned alongside the posterior so
the two uncertainty statements remain distinguishable. A boundary-sensitive
result is flagged rather than interpreted as a well-constrained interior
solution. Joint pCO2-GPP-pO2 probability inference would require explicit
priors for all free coordinates and a justified probabilistic model-discrepancy
term; neither is implied by this conditional API.

## Shape and inversion gates

A 375-scenario high-domain audit covers 0.1, 1, and 2 PAL O2; 5, 25, 50, 100,
and 232% of Young GPP; and 25 log-spaced CO2 values from 294 to 60,000 ppm. A
separate low-CO2 native-column audit covers 150-300 ppm, 0.1-2 PAL, and
5-232% Young GPP. All tested
curves:

- become more negative with increasing pCO2;
- become less negative with increasing GPP;
- become less negative with increasing pO2;
- contain one pCO2 root for every tested interior isotope target;
- converge with fixed-point residual below `1e-8 per mil`.

No hooks or sign reversals occur in the tested domain. The updated curves are
less negative than the current compact public model at high pCO2 by up to about
`1.14 per mil` in the five comparison cases. This difference is retained as a
model-architecture consequence rather than removed by fitting.

## Isotope output accelerator

Repeated heatmap and inversion calculations may use the versioned numerical
cache in `model_data/updated_molecular_output_surface_v1.json`. Its
Delta-prime-17O component contains 5,292 live-kernel solutions on a
9 x 28 x 21 pO2-pCO2-GPP grid and does not alter the physical model. Central
Delta-prime-17O uses a tensor-product cubic
interpolant in pO2, pCO2, and ln(GPP). Each uncertainty bound is represented as
a positive distance from the central value and interpolated in log space;
interval crossing is therefore excluded by construction.

A fresh audit at 432 deterministic noncentral positions, distinct from the
geometric cell centers used during method selection, gives a maximum absolute
residual of `0.004940 per mil` across central Delta-prime-17O and all reported
uncertainty fields. This passes the predeclared `0.015 per mil` central and
`0.020 per mil` all-field limits. The measured repeated-evaluation speedup is
about `468x` relative to the full uncertainty-aware live kernel.

A separate 401,841-point shape audit finds no nonfinite values, interval
crossings, or positive pCO2 steps. Cubic overshoot beyond the training-node
Delta-prime-17O range is `0.000045 per mil`.

Delta-prime-18O has stronger low-GPP and high-pCO2 curvature, so trilinear
interpolation on that grid failed its predeclared `0.050 per mil` target
(maximum residual `0.118604 per mil`). It therefore uses a separate
17 x 49 x 25 grid of 20,825 central live-kernel states and local
tensor-product quadratic interpolation in pO2, ln(pCO2), and ln(GPP). No
physical rate, offset, or model output is fitted in this numerical refinement.

A 432-case method-selection audit gives a maximum absolute delta-prime-18O residual of
`0.028124 per mil`. A stricter audit at one new noncentral point in every one
of the 18,432 refined-grid cells gives mean, 95th-percentile, and maximum
absolute residuals of `0.001194`, `0.005056`, and `0.034922 per mil`,
respectively. Delta-prime-18O is therefore approved for accelerated output
within the declared domain.

## Slow atmospheric response

The updated model propagates the globally mixed atmospheric O2 isotopologue
inventories after a step change in pCO2, absolute GPP, or pO2. The year-zero
state is the initial updated-model equilibrium. A pCO2 or GPP step changes the
photochemical or biological forcing without an instantaneous isotope shift. A
pO2 step changes the major and rare O2 inventories proportionally, preserving
both isotope ratios at year zero. The subsequent state-dependent R7 tendency
and partitioned biological source and respiration budget are the same central
physical terms used by the updated steady solver.

Integration uses DOP853 on scaled O2 isotopologue inventories. The final
equilibrium is calculated independently with the updated forward solver and is
reported with every time series. Regression tests require all three step types
to approach that equilibrium and explicitly reject an instantaneous isotope
response to pCO2 or GPP.

The separate photosynthesis-at-fixed-respiration experiment uses operator
splitting: the detailed atmospheric boxes supply the live pCO2 trajectory to
the updated molecular O2 reservoir. Both integrations independently evolve an
O2 inventory under the same photosynthesis perturbation. At 50% photosynthesis,
their pO2 trajectories differ by at most `0.001394 PAL`, while the molecular
reservoir produces the Fig. 9-type Delta-prime-17O overshoot before relaxation.
This closure supports operator splitting for the present global sensitivity
experiment, but it is not a claim of fully simultaneous carbon-oxygen dynamics.
The BDF carbon solver retains internal finite-difference trial-state warnings
as provenance; accepted trajectories must be finite and are checked for
negative inventories before clipping.

## Unified release and Young-response diagnostics

`validation/audit_updated_molecular_release_scorecard.py` evaluates the exact
molecular engine and accelerator used by the public UI. Formal numerical and
physical gates are kept separate from Young et al. (2014) comparison
diagnostics. The accepted modern-reference difference is reported but is not
applied as a model-output offset.

The current updated model follows digitized Young Fig. 7 closely after that
reference difference is stated, but its high-pCO2 Fig. 8 response is more
negative, especially at 50% GPP. The curve-wide diagnostic in
`validation/audit_updated_fig8_response_shape.py` shows that reproducing Young
inside the unchanged updated model would require a pCO2-dependent effective
GPP increase, reaching about 1.26 times nominal at 50% GPP and 30,000 ppm.
Because the biological law itself has no pCO2 dependence, this pattern points
to the high-pCO2 photochemical response or a missing carbon-cycle feedback.
It is not used as a correction or fitted factor.

The competing-hypotheses assessment in
`docs/high_pco2_model_disagreement.md` further shows that the updated tail is
supported by direct converged 30,000 and 60,000 ppm parent atmospheres and a
smooth O(1D)-competition response. It also records why this does not establish
accuracy: modern temperature/transport structure and a modern transfer
normalization are retained at extreme pCO2. The Young divergence is therefore
treated as structural model disagreement, with imprecision in Young's
high-pCO2 50% GPP limb remaining a plausible explanation.

## Reproduce

```powershell
python validation/export_updated_response_surface_bundle.py
python -m pytest validation/test_updated_molecular_forward_model.py `
  validation/test_updated_molecular_inverse.py `
  validation/test_biological_o2_ensemble.py `
  validation/test_updated_molecular_relative_gpp.py `
  validation/test_updated_response_surface_bundle.py -q -p no:cacheprovider
python validation/audit_updated_biological_envelope.py
python validation/audit_updated_forward_surface.py
python validation/audit_updated_output_surface.py `
  --candidate outputs/updated_molecular_output_surface_full_refined_candidate.json `
  --maximum-holdouts 432
python validation/audit_updated_output_surface_shape.py `
  --surface model_data/updated_molecular_output_surface_v1.json
python validation/audit_updated_delta18_surface.py `
  --surface model_data/updated_molecular_output_surface_v1.json `
  --report outputs/updated_molecular_output_surface_delta18_every_cell_audit.json
python validation/audit_updated_molecular_release_scorecard.py
python validation/audit_updated_fig8_response_shape.py
python validation/audit_high_pco2_competing_hypotheses.py
```

The source-controlled runtime data are in
`model_data/updated_r7_response_surface_v1.json` and
`model_data/updated_molecular_output_surface_v1.json`. Generated audit tables
and figures remain untracked outputs.
