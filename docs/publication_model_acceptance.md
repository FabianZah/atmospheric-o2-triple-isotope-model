# Publication Model Acceptance

**Verdict:** `accepted_for_steady_forward_inverse_and_declared_time_responses`

This audit evaluates the single OXYTIB publication model against its declared evidence and scope.

| Section | Gate | Status | Result |
|---|---|---|---|
| core | Formal numerical and physical release gates | pass | 23 passed; 0 failed |
| modern | Modern atmospheric O2 Delta-prime-17O | pass | residual +0.005814 per mil |
| modern | Modern atmospheric O2 delta-18O | pass | residual -0.219671 per mil |
| surface | Finite monotonic solution surface | pass | 401,841 audited points |
| surface | Fixed marine numerical accelerator | pass | 4640 main-field and 20352 oxygen-18 holdouts |
| scope | Fixed marine biological interpretation | scope_limit | 20 C/reference-seawater endmember; biological parameter uncertainty unassessed |
| surface | Declared operational domain | pass | pO2 0.1-2.0 PAL; pCO2 50-60000 ppm; GPP 18.256-850 PgC/yr |
| inverse | Forward-inverse round trips | pass | relative error 8.327e-13; live residual 2.756e-05 per mil |
| inverse | Joint posterior solution ridge | pass | generated target recovered |
| uncertainty | Separated uncertainty layers | pass | measurement, parameter, numerical, structural |
| validation | Young Fig. 7 historical response anchor | pass | aligned MAE 0.0126 per mil |
| validation | Young Fig. 8 historical response anchor | qualified | aligned MAE 0.3719 per mil |
| validation | Liu low-GPP response topology | pass | rank r=0.9960; RMSE 5.39 per mil |
| validation | Cao-Bao high-pCO2 response direction | pass | all pO2 cases monotonic through 30,000 ppm |
| validation | Luz normalized-productivity inversion | pass | RMSE 1.62 percentage points; r=0.977 |
| validation | Yang ice-core CO2 predictive information | pass | raw blocked-CV skill 0.229 |
| validation | Independent paleo applications | pass | Banerjee change 8.35 points; Brandon GPP 112.3% |
| structural | Extreme high-pCO2/low-GPP amplitude | qualified | model families agree in direction but not amplitude |
| structural | Marine O2 accessibility | excluded | Liu RMSE change -0.178 per mil |
| structural | pCO2-dependent climate profile | excluded | deferred; archived offsets use earlier chemistry |
| scope | Slow geological reservoirs during photosynthesis transients | scope_limit | shared prognostic carbon/O2 budget; balanced slow geological background |
| scope | Whole-domain probabilistic model discrepancy | scope_limit | no defensible global Gaussian sigma |

## Accepted claims

- steady forward calculation within the declared pO2-pCO2-GPP domain
- conditional and joint inversion with independent constraints and explicit priors
- validated pCO2, GPP, and pO2 perturbation steps and prescribed gradual pCO2 trajectories
- separate propagation and reporting of declared uncertainty layers

## Scope boundaries

- inversion combines one isotope observation with independent coordinate constraints
- historical Young calculations provide validation provenance
- carbon forcing is prescribed in the declared atmospheric-O2 transients
- structural evidence is reported by domain and with explicit priors
- precision is qualified in the extreme high-pCO2 and low-GPP corner

## Decision

There are no release-blocking failures. The deterministic core is accepted for steady forward and inverse applications and for the declared time-response experiments. High-pCO2 model-family spread, the climate and marine-access sensitivity cases, and the balanced slow geological background of the photosynthesis experiment remain explicit scope limits.

Central-model change policy: Replace a central equation or parameter only when independently constrained evidence improves multiple validation families while preserving conservation, modern observations, and surface behavior.

Machine-readable details are written to `outputs/publication_model_acceptance.json`.
