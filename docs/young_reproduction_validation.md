# Young-Reproduction Validation

Preset: `young_reproduction`

| metric | value |
|---|---:|
| overall score | 80.1% |
| Fig. 7 mean abs residual | 0.0229‰ |
| Fig. 8 mean abs residual | 0.0464‰ |
| Fig. 8 max abs residual | 0.1012‰ |
| Fig. 9 final Δ′17O | -0.53384‰ |
| Fig. 9 minimum Δ′17O | -0.73120‰ |
| Fig. 10 interpolated shift at 150 yr | -0.00503‰ |
| Fig. 10 final plotted shift | -0.04925‰ |

## Dynamic Scorecard Rows

| constraint | model | Young | residual |
|---|---:|---:|---:|
| fig9_half_rp_final_O2_D17O | -0.53482 | -0.539 | +0.0041787 |
| fig9_min_O2_D17O | -0.7314 | -0.73 | -0.0013974 |
| fig9_peak_CO2_ppm | 1013.7 | 1000 | +13.706 |
| fig9_peak_O2_d18p | 27.896 | 28 | -0.10352 |
| fig10_150yr_shift | -0.0050313 | -0.006 | +0.00096866 |

## Outputs

- Plot: `outputs/young_reproduction_validation.png`
- Scorecard: `outputs/young_reproduction_validation_scorecard.csv`
- Summary: `outputs/young_reproduction_validation_summary.csv`
- Fig. 7 rows: `outputs/young_reproduction_validation_fig7_rows.csv`
- Fig. 8 rows: `outputs/young_reproduction_validation_fig8_rows.csv`
- Transient rows: `outputs/young_reproduction_validation_transients.csv`

Solver failures: `0`
