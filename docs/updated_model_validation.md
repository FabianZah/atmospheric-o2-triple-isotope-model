# Updated Model Validation

Young checkpoint: `young_reproduction`
Updated model: `physical_extrapolation`

## Modern Comparison

| quantity | Young checkpoint | updated physical |
|---|---:|---:|
| O2 Delta'17O (per mil) | -0.434496 | -0.430954 |
| O2 delta'18O (per mil) | 23.192088 | 23.180334 |
| CO2 strat Delta'17O (per mil) | 1.607477 | 1.412911 |
| CO2 export Delta'17O (per mil) | 1.607477 | 0.893360 |
| a_MIF | 1.065000000 | 1.065000000 |

## Reference Residuals

| reference | model value | reference | residual | units |
|---|---:|---:|---:|---|
| Pack, 2021 / modern_o2_delta17o_pack_2021 | -0.430954 | -0.432 | +0.00104574 | permil |
| Adnew et al., 2025 / adnew_2025_strat_trop_co2_delta17o_net_isoflux | 54.4735 | 51.3 +/- 1.6 | +3.17352 | permil PgC yr-1 |
| Liang et al., 2023 / liang_2023_global_gpp_o2_co2_delta17o | 365.125 | 290 +/- 30 | +75.1253 | PgC yr-1 |

## Structural Comparison

| metric | Young checkpoint | updated physical |
|---|---:|---:|
| Fig. 8 mean abs residual to Young (per mil) | 0.04640 | 0.16567 |
| Fig. 8 max abs residual to Young (per mil) | 0.10115 | 1.13395 |
| pO2 grid min updated Delta'17O (per mil) |  | -3.07334 |
| pO2 grid max updated Delta'17O (per mil) |  | -0.38388 |

## Pack Anchor Policy Diagnostic

The local-offset preset applies the Pack 2021 modern O2 Delta'17O anchor after solving the validated Young-like ODE surface. It is a scenario-interface calibration layer: raw ODE outputs are retained, and reported O2 isotope values carry explicit offset metadata.

| metric | Young checkpoint | updated_physical | Pack local O2 offset | Pack via beta respiration | Pack via water beta |
|---|---:|---:|---:|---:|---:|
| Modern O2 Delta'17O (per mil) | -0.434496 | -0.430954 | -0.432000 | -0.432388 | -0.432395 |
| Local O2 offset (per mil) | 0.000000 | not used | -0.016861 | not used | not used |
| Fig. 8 mean abs residual to Young (per mil) | 0.04640 | 0.16567 | 0.05325 | 0.05175 | 0.05178 |
| Fig. 8 max abs residual to Young (per mil) | 0.10115 | 1.13395 | 0.17074 | 0.16279 | 0.16242 |

## Outputs

- Plot: `outputs/updated_model_validation.png`
- Modern comparison: `outputs/updated_model_validation_modern.csv`
- Fig. 8 comparison: `outputs/updated_model_validation_fig8.csv`
- pO2 comparison: `outputs/updated_model_validation_po2.csv`
- Modern references: `outputs/updated_model_validation_references.csv`
