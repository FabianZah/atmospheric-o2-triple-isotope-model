# Model validation criteria

OXYTIB is assessed against numerical, physical and observational evidence.
The criteria are implemented in
`validation/audit_publication_model_acceptance.py` and summarized in
[the validation assessment](publication_model_acceptance.md).

## Numerical and physical checks

- Runtime data identity and file integrity.
- Finite solutions, isotope and oxygen budget consistency.
- Monotonic response to pCO2, GPP and pO2 within the declared domain.
- Interpolation errors against direct model calculations at points withheld from the grid.
- Forward-to-inverse closure and synthetic-target recovery.
- Normalized posterior mass and converged uncertainty integration.
- Continuous isotope response after prescribed perturbations, with agreement
  between long-time solutions and independently calculated equilibrium.

## Observational and published-model evidence

Modern atmospheric oxygen is compared with Pack (2021), with conventional
and logarithmic isotope notation converted consistently. Ice-core tests use
blocked validation and retain the assumptions about GPP, pO2 and chronology.
Young et al. (2014), Liu et al. (2021), Cao and Bao (2013), and Luz et al.
(1999) provide complementary response and productivity comparisons.

Published-model differences are evaluated by quantity and domain. Agreement
in response direction and disagreement in amplitude are reported separately.
The historical Young response test can be reproduced with
`python validation/audit_young_acceptance_gate.py`.

## Interpretation

Formal numerical gates, descriptive comparison metrics and scientific scope
limits are distinct. A pass applies to the documented tests and operational
domain; high-pCO2 structural sensitivity remains part of the interpretation.

Reproduce the integrated assessment with `python run_model.py validate`.
The immutable input evidence is under `model_data/validation_evidence/`;
generated reports are written to `outputs/`.
