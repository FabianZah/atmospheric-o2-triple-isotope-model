# Updated Molecular Model Release Scorecard

This scorecard evaluates the molecular engine used by the public UI. Young comparisons are diagnostics, not fitted constraints. The accepted modern-reference discrepancy is reported but is not applied inside the model.

| Category | Metric | Value | Status | Criterion |
|---|---|---:|---|---|
| modern | Pack Delta-prime-17O residual | 0.00581407 permil | pass | absolute residual <= 0.015 permil |
| modern | Pack conventional delta-18O residual | -0.219671 permil | pass | absolute residual <= 0.3 permil |
| numerics | Modern fixed-point residual | 2.23821e-13 permil | pass | solver converged at tolerance 1e-8 permil |
| accelerator | Delta-prime-17O maximum holdout residual | 0.00328227 permil | pass | all released fields < 0.020 permil |
| accelerator | delta-prime-18O maximum holdout residual | 0.0124491 permil | pass | maximum residual < 0.050 permil |
| accelerator | Low-pCO2 maximum Delta-prime-17O holdout residual | 0.000422753 permil | pass | all released fields < 0.020 permil |
| accelerator | Low-pCO2 maximum delta-prime-18O holdout residual | 0.00150657 permil | pass | maximum residual < 0.050 permil |
| shape | Dense-domain monotonic and finite gates | 401841 points | pass | all declared shape gates true |
| inversion | Maximum three-coordinate round-trip error | 8.32675e-13 relative | pass | relative error <= 1e-8 |
| inversion | Maximum live-kernel residual at accelerated roots | 2.75563e-05 permil | pass | absolute residual <= 0.015 permil |
| transient | Balanced pCO2-step positivity | True  | pass | all isotopologue inventories > 0 |
| transient | Balanced pCO2-step distance at 12000 years | 0.000318164 permil | pass | both isotope coordinates within 0.001 permil |
| transient | Fig. 10-type monotonic Delta-prime-17O relaxation | True  | pass | no reversal after the maintained 294-to-400 ppm step |
| Young diagnostic | Accepted modern-reference difference | -0.0151015 permil | diagnostic | reported, not corrected |
| Young diagnostic | Fig. 7 aligned mean absolute residual | 0.0126056 permil | diagnostic | comparison only |
| Young diagnostic | Fig. 8 aligned mean absolute residual | 0.371921 permil | diagnostic | comparison only |
| Young diagnostic | Fig. 10 Delta-prime-17O shift residual at 150 years | 0.00167449 permil | diagnostic | comparison only |
| Young diagnostic | Fig. 10 Delta-prime-17O shift residual at 5000 years | -0.00333502 permil | diagnostic | comparison only |
| domain | Physical and accelerated pCO2 domain | 50-60000 ppm | pass | native low-pCO2 training and holdout gates passed |
| Young diagnostic | Fig. 7 contours below physical domain | 0 contours | pass | reported without extrapolation below 50 ppm |
| external application | Banerjee MPT minus pre-MPT GPP | 8.35066 percentage points | diagnostic | paper reports an approximately 10% increase; no fit applied |
| external application | Yang raw age-block CO2 tracking skill | 0.229363 fraction | diagnostic | positive skill relative to an intercept-only holdout model |
| external application | Yang 21-kyr-smoothed CO2 tracking correlation | 0.77602 Pearson r | diagnostic | descriptive comparison; overlapping windows are non-independent |
| external application | Yang 21-kyr cross-validated CO2 inversion RMSE | 12.6967 ppm | diagnostic | compare with independently measured ice-core CO2 |
| external application | CO2-only slope difference from Yang TB model | -0.101596 relative | diagnostic | comparison only |
| external application | Continuous-CO2 emergent response lag | 1250 years | diagnostic | emerges from the global O2 budget; not fitted to observations |
| external application | Yang 21-kyr transient age-block tracking skill | 0.634771 fraction | diagnostic | compare with the identical instantaneous forcing experiment |
| external application | Termination V fixed-GPP mean residual | 12.903 ppm | diagnostic | a coherent positive residual is required before varying GPP |
| external application | Termination V inferred GPP | 112.299 % pre-industrial | diagnostic | compare with Brandon et al. reported 110-130% range |
| external application | Termination V GPP-pulse SSE reduction | 0.675058 fraction | diagnostic | improvement over the identical fixed-GPP transient |
| uncertainty | Conditional one-coordinate posterior normalization | 2.22045e-16 absolute error | pass | posterior integral error <= 1e-8 |
| uncertainty | Conditional posterior generated-target recovery | True  | pass | central root recovered and contained by the credible interval |
| uncertainty | Joint pCO2-GPP-pO2 posterior generated-target recovery | True  | pass | normalized posterior and all generated coordinates inside marginal credible intervals |
| uncertainty | Explicit model-discrepancy provenance requirement | True  | pass | nonzero probabilistic discrepancy requires a named source |
| uncertainty | Separated uncertainty-layer contract | True  | pass | measurement, parameter, numerical, and structural layers remain separate |
| uncertainty | Yang low-pCO2 excess predictive scale | 0.0132387 permil | diagnostic | late-Quaternary, 1 PAL, fixed-GPP domain only |
| uncertainty | Empirical structural model-discrepancy calibration | False  | limitation | required before a default joint posterior is presented as calibrated |
| Young diagnostic | Fig. 9-type half-photosynthesis minimum | -0.776244 permil | diagnostic | Young visual minimum approximately -0.73 permil |
| Young diagnostic | Fig. 9-type peak delta-prime-18O | 28.4733 permil | diagnostic | Young visual peak approximately 28 permil |
| transient | Half-photosynthesis multi-reservoir response | True  | pass | pO2 decreases, pCO2 overshoots, and isotope extrema precede equilibrium |
| transient | Shared carbon/O2 inventory | 0 PAL | pass | identical O2 inventory in the joint budget |
| transient | Uncontained numerical warning classes | 0  | pass | zero |

`pass` and `fail` are formal release gates. `diagnostic` records Young agreement without retuning. `limitation` identifies remaining publication work.

The companion curve-wide mechanism audit is `validation/audit_updated_fig8_response_shape.py`; it reports the required-GPP equivalent without using it as a correction.

Complete values and provenance are in `outputs/updated_molecular_release_scorecard.json`.
