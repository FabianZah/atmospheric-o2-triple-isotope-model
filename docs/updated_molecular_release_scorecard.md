# Updated Molecular Model Release Scorecard

This scorecard evaluates the molecular engine used by the public UI. Young comparisons are diagnostics, not fitted constraints. The accepted modern-reference discrepancy is reported but is not applied inside the model.

| Category | Metric | Value | Status | Criterion |
|---|---|---:|---|---|
| modern | Pack Delta-prime-17O residual | 0.00564962 permil | pass | absolute residual <= 0.015 permil |
| modern | Pack conventional delta-18O residual | -0.219641 permil | pass | absolute residual <= 0.3 permil |
| numerics | Modern fixed-point residual | 5.85976e-13 permil | pass | solver converged at tolerance 1e-8 permil |
| accelerator | Delta-prime-17O maximum holdout residual | 0.00494045 permil | pass | all released fields < 0.020 permil |
| accelerator | delta-prime-18O maximum holdout residual | 0.0349224 permil | pass | maximum residual < 0.050 permil |
| accelerator | Low-pCO2 maximum Delta-prime-17O holdout residual | 0.00152147 permil | pass | all released fields < 0.020 permil |
| accelerator | Low-pCO2 maximum delta-prime-18O holdout residual | 0.00183333 permil | pass | maximum residual < 0.050 permil |
| shape | Dense-domain monotonic and finite gates | 401841 points | pass | all declared shape gates true |
| inversion | Maximum three-coordinate round-trip error | 8.70841e-13 relative | pass | relative error <= 1e-8 |
| inversion | Maximum live-kernel residual at accelerated roots | 4.39709e-05 permil | pass | absolute residual <= 0.015 permil |
| transient | Balanced pCO2-step positivity | True  | pass | all isotopologue inventories > 0 |
| transient | Balanced pCO2-step distance at 12000 years | 0.000318537 permil | pass | both isotope coordinates within 0.001 permil |
| transient | Fig. 10-type monotonic Delta-prime-17O relaxation | True  | pass | no reversal after the maintained 294-to-400 ppm step |
| Young diagnostic | Accepted modern-reference difference | -0.0152665 permil | diagnostic | reported, not corrected |
| Young diagnostic | Fig. 7 aligned mean absolute residual | 0.0128009 permil | diagnostic | comparison only |
| Young diagnostic | Fig. 8 aligned mean absolute residual | 0.403343 permil | diagnostic | comparison only |
| Young diagnostic | Fig. 10 Delta-prime-17O shift residual at 150 years | 0.00166808 permil | diagnostic | comparison only |
| Young diagnostic | Fig. 10 Delta-prime-17O shift residual at 5000 years | -0.00340769 permil | diagnostic | comparison only |
| domain | Physical and accelerated pCO2 domain | 50-60000 ppm | pass | native low-pCO2 training and holdout gates passed |
| Young diagnostic | Fig. 7 contours below physical domain | 0 contours | pass | reported without extrapolation below 50 ppm |
| external application | Banerjee MPT minus pre-MPT GPP | 8.36022 percentage points | diagnostic | paper reports an approximately 10% increase; no fit applied |
| external application | Yang raw age-block CO2 tracking skill | 0.22914 fraction | diagnostic | positive skill relative to an intercept-only holdout model |
| external application | Yang 21-kyr-smoothed CO2 tracking correlation | 0.775993 Pearson r | diagnostic | descriptive comparison; overlapping windows are non-independent |
| external application | Yang 21-kyr cross-validated CO2 inversion RMSE | 12.6913 ppm | diagnostic | compare with independently measured ice-core CO2 |
| external application | CO2-only slope difference from Yang TB model | -0.10071 relative | diagnostic | comparison only |
| external application | Continuous-CO2 emergent response lag | 1250 years | diagnostic | emerges from the global O2 budget; not fitted to observations |
| external application | Yang 21-kyr transient age-block tracking skill | 0.63435 fraction | diagnostic | compare with the identical instantaneous forcing experiment |
| external application | Termination V fixed-GPP mean residual | 12.9362 ppm | diagnostic | a coherent positive residual is required before varying GPP |
| external application | Termination V inferred GPP | 112.317 % pre-industrial | diagnostic | compare with Brandon et al. reported 110-130% range |
| external application | Termination V GPP-pulse SSE reduction | 0.676225 fraction | diagnostic | improvement over the identical fixed-GPP transient |
| uncertainty | Conditional one-coordinate posterior normalization | 1.11022e-16 absolute error | pass | posterior integral error <= 1e-8 |
| uncertainty | Conditional posterior generated-target recovery | True  | pass | central root recovered and contained by the credible interval |
| uncertainty | Joint pCO2-GPP-pO2 posterior generated-target recovery | True  | pass | normalized posterior and all generated coordinates inside marginal credible intervals |
| uncertainty | Explicit model-discrepancy provenance requirement | True  | pass | nonzero probabilistic discrepancy requires a named source |
| uncertainty | Separated uncertainty-layer contract | True  | pass | measurement, parameter, numerical, and structural layers remain separate |
| uncertainty | Yang low-pCO2 excess predictive scale | 0.0132415 permil | diagnostic | late-Quaternary, 1 PAL, fixed-GPP domain only |
| uncertainty | Empirical structural model-discrepancy calibration | False  | limitation | required before a default joint posterior is presented as calibrated |
| transient | Fully simultaneous carbon-oxygen coupling | False  | limitation | required before claiming fully coupled transient dynamics |

`pass` and `fail` are formal release gates. `diagnostic` records Young agreement without retuning. `limitation` identifies remaining publication work.

The companion curve-wide mechanism audit is `validation/audit_updated_fig8_response_shape.py`; it reports the required-GPP equivalent without using it as a correction.

Complete values and provenance are in `outputs/updated_molecular_release_scorecard.json`.
