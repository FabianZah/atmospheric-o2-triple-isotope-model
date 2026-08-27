# OXYTIB expert review guide

## Review status

This package is a release candidate for OXYTIB 0.1.0. The deterministic model,
operational domain, and uncertainty contract are fixed for this review. The
review is intended to identify scientific, interpretive, reproducibility, and
interface issues before the archival release and DOI are created.

Please record the exact Git tag or commit shown with the review invitation.
Results from different commits should not be combined as one review.

## Review routes

The hosted interface is available at
[mycompton.de/oxytib](https://mycompton.de/oxytib/). It stores neither submitted
constraints nor calculated results.

For an independent local check, clone the tagged release candidate, follow
[`SETUP.md`](../SETUP.md), and run:

```powershell
python run_model.py validate
```

A complete local validation ends with zero release blockers and the verdict
`accepted_for_steady_forward_inverse_and_declared_time_responses`.

## Suggested review sequence

### 1. Scientific scope and definitions

Read the model summary, notation, FAQ, and references in the browser interface.
Check whether the stated relationship among atmospheric O₂ isotope composition,
pCO₂, pO₂, and GPP is scientifically clear and appropriately qualified.

### 2. Known-answer forward calculation

Run the local console calculation:

```powershell
python run_model.py calculate forward --po2 1 --pco2 294 --gpp 290
```

The raw central result should be approximately:

| Quantity | Expected value |
|---|---:|
| Δ′¹⁷O of atmospheric O₂ | -0.426353 per mil |
| δ′¹⁸O of atmospheric O₂ | 23.404329 per mil |

This is a reproducibility check, not a requirement that the mechanistic result
equal the observational reference exactly.

### 3. Forward-inverse closure

Use the calculated Δ′¹⁷O value as the target, fix pO₂ at 1 PAL and GPP at
290 Pg C yr⁻¹, and solve for pCO₂. The central solution should recover 294 ppm
to numerical precision. Repeat the exercise by solving for GPP or pO₂.

### 4. Constrained inference

In the Model solver, test fixed, 1σ, and range constraints for the two known
coordinates. Check whether the reported solution, probability field, interval,
and boundary behavior remain interpretable. Include at least one target that
does not have an accepted solution inside the operational domain.

### 5. Proxy conversion

Select I-type cosmic spherules, enter Δ′¹⁷O and δ¹⁸O values with analytical
uncertainties, and check the atmospheric conversion and downstream inference.
Confirm that the input reference scales and exported metadata are unambiguous.

### 6. Isotope field

Inspect broad and narrow pCO₂-GPP windows at several pO₂ values. Check contour
ordering, adaptive labels, color interpretation, axis scaling, and the exported
PNG. Report any hook, reversal, discontinuity, or misleading contour placement.

### 7. Time responses

Run a pCO₂ step, GPP step, pO₂ step, photosynthesis step, and gradual pCO₂
trajectory. Check the pre-change state, imposed forcing, isotope response,
steady-state reporting, progress indicator, and XLSX export. The photosynthesis
experiment is expected to take longest.

### 8. Reproducibility and exports

Download a solver XLSX file, a time-response XLSX file, and at least one PNG.
Check that units, constraints, uncertainties, model version, and calculation
metadata are sufficient to interpret the result without the browser session.

## Review priorities

Please distinguish among:

- **Release blocker:** wrong scientific result, non-reproducible calculation,
  broken core workflow, missing provenance, security or privacy problem.
- **Major:** misleading interpretation, important export omission, substantial
  usability problem, or behavior that needs scientific investigation.
- **Minor:** wording, typography, layout, or convenience issue that does not
  change scientific interpretation.
- **Suggestion:** a useful extension that can follow version 0.1.0.

Use [`expert_review_feedback_template.md`](expert_review_feedback_template.md)
for the report. Screenshots and exported workbooks are useful for defects, but
do not include confidential or unpublished proxy data unless intended.

## Declared qualifications

The integrated acceptance report records two qualified comparisons: the
amplitude difference from Young et al. (2014) Fig. 8 and model-family spread in
the extreme high-pCO₂/low-GPP region. Fully coupled carbon-oxygen transients and
a whole-domain probabilistic structural-error model are outside the current
scope. These qualifications should be assessed for clarity and scientific
appropriateness; their presence alone is not a software failure.
