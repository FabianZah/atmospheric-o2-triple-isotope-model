# OXYTIB documentation

OXYTIB is the atmospheric oxygen triple-isotope budget and inference model.
The authoritative machine-readable model identity is
`model_data/publication_model_contract_v1.json`.

## Start here

- `publication_model_definition.md`: model architecture, reporting, and scope.
- `publication_model_acceptance.md`: integrated scientific evidence and release
  decision.
- `spherule_inversion_workflow.md`: I-type cosmic-spherule conversion,
  uncertainty propagation, and constrained inference.
- `constrained_coordinate_inference.md`: fixed, normal, and range constraints.
- `sulfate_transfer.md`: sulfate inputs, oxygen incorporation, and uncertainty integration.
- `posterior_numerical_resolution.md`: numerical resolution of probability fields.
- `GPP_NORMALIZATION_POLICY.md`: absolute and relative GPP reporting.
- `server_deployment.md`: independent Linux-server deployment and recovery.
- `validation_evidence_bundle.md`: evidence provenance and integrity policy.

## Model use

OXYTIB predicts atmospheric O₂ Δ′¹⁷O and δ′¹⁸O from pCO₂, pO₂, and GPP. Its
inference workflow combines an isotope observation with independent constraints
on two coordinates to estimate the third.

```powershell
python run_model.py calculate forward --po2 1 --pco2 294 --gpp 290
python run_model.py calculate infer --target-d17o -0.426 --solve-for pCO2 --po2 1 --gpp 290
python run_model.py validate
```

The hosted browser, HTTP API, and console entry point share the same
framework-neutral service functions and versioned model data.

## Operational domain

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| pO₂ | 0.10 PAL | 2.00 PAL |
| pCO₂ | 50 ppm | 60,000 ppm |
| GPP | 18.256 Pg C yr⁻¹ | 850 Pg C yr⁻¹ |

## Terms used in technical reports

- **PAL:** present atmospheric level. The model defines 1 PAL O₂ as 21.2%.
- **GPP:** gross primary production; 100% modern in the interface is 290 Pg C yr⁻¹.
- **Physical kernel:** the direct calculation of the physical model.
- **Output surface or accelerator:** a precomputed model grid used for faster
  evaluation by interpolation. Its error is tested against direct calculations.
- **Holdout:** a calculation or observation withheld from construction or
  calibration and used to test the resulting model or interpolation.
- **Isoflux:** a flux weighted by its isotope composition; here, the CO₂ flux
  weighted by its oxygen-isotope anomaly.
- **R1-R7 and R7:** reaction labels following Young et al. (2014), retained in
  technical identifiers. R7 describes CO₂ oxygen-isotope exchange with excited
  atomic oxygen, O(1D).
- **Young anchor:** a comparison with the published model of Young et al. (2014).
- **RCE:** radiative-convective equilibrium, used in the climate sensitivity tests.

## Evidence structure

Validation records modern observations, analytical benchmarks, conservation,
numerical interpolation, ice-core behavior, published-model comparisons, and
declared structural end members. Measurement/proxy, parameter, numerical, and
structural uncertainty are retained as distinct evidence layers.

The release includes compact evidence under `model_data/validation_evidence/`.
Advanced source-model calculations involving native Photochem, ERA5, or Clima
are provenance workflows and are outside the normal OXYTIB runtime.
