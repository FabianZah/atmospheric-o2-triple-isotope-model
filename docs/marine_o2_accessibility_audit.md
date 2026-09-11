# Marine O2 accessibility audit

## Question

Liu et al. (2021) distinguish total marine gross primary production from the
fraction of produced O2 that reaches the atmosphere before being recycled in
the ocean. This audit asks whether applying that accessibility law to only the
marine share of OXYTIB's total global GPP improves the single
publication model.

The sensitivity experiment is independently parameterized from the archived Liu `GPPOXY`
fluxes at 0.1, 0.3, and 1 PAL pO2. Terrestrial production remains fully
atmosphere-accessible. No isotope observation or inter-model residual is used
to fit the term, and the major-O2 budget closes exactly.

## Results

| Guardrail | OXYTIB | Marine-access sensitivity | Change |
|---|---:|---:|---:|
| Modern Delta-prime-17O (per mil) | -0.42635 | -0.43379 | -0.00743 |
| Modern conventional delta-18O (per mil) | 23.607 | 23.760 | +0.153 |
| Liu shared-grid response RMSE (per mil) | 6.17696 | 6.05275 | -0.12420 |
| Young Fig. 7 aligned mean absolute residual (per mil) | 0.01280 | 0.02007 | +0.00727 |
| Young Fig. 8 aligned mean absolute residual (per mil) | 0.40334 | 0.54352 | +0.14018 |
| Yang raw blocked-CV tracking skill | 0.22914 | 0.21765 | -0.01149 |
| Yang 21-kyr blocked-CV tracking skill | 0.58746 | 0.56801 | -0.01945 |

The sensitivity case overlaps the Pack (2021) one-sigma intervals for both modern
Delta-prime-17O and delta-18O. It modestly improves the independent Liu
response comparison, especially at high GPP and high pCO2. It does not close
the extreme low-GPP amplitude difference. After diagnostic modern-reference
alignment, it also worsens both Young curve families and slightly weakens the
Yang ice-core CO2 tracking test.

## Decision

Marine accessibility is retained as a structural uncertainty experiment.
OXYTIB uses a total-global-GPP boundary; the tested alternative has mixed
performance across the independent validation families and a narrower
source-constrained pO2 mapping.

Machine-readable results are in
`model_data/validation_evidence/marine_o2_accessibility_audit.json`.
