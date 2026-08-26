# Extrapolation bounds for O2 Delta'17O

For states outside Young's calibration box (high pCO2, low GPP, non-modern pO2),
a single O2 Delta'17O number is not trustworthy. The model can report a small,
physically-interpretable envelope instead. It is opt-in and does not change the
headline value.

## How to get it

```python
run_scenario(ScenarioInput(preset="physical_extrapolation",
                           p_co2_ppm=50000, p_o2_pal=0.5, gpp_scale=0.3,
                           report_extrapolation_bounds=True))
```

or on the CLI:

```
python code/run_scenario.py --preset physical_extrapolation \
  --pCO2-ppm 50000 --pO2-pal 0.5 --gpp-scale 0.3 --report-extrapolation-bounds
```

## What you get (one idea, not many knobs)

The whole pCO2/GPP/pO2 response of the resolved-chemistry branch collapses to

```
Delta_O2 = delta_eq / (1 + GPP / a)
```

- `delta_eq` is the **saturation cap**: the most negative O2 Delta'17O reachable
  at any GPP. It equals minus the O(1D) Delta'17O, so it is set by one literature
  quantity. Young's preferred value gives a cap near **-27.6 permil**; the Barkan
  & Luz (2011) anchor gives **-46.6 permil**.
- `a` is the **pump strength** (the O(1D) -> CO2 transfer), which grows roughly
  linearly with pCO2.

Young et al. (2014, Section 6) identify exactly these as the controlling and most
uncertain inputs ("the isotopic composition of O(1D) is the ultimate driver";
"Delta'17O O2 is most sensitive to the rate of O(1D) quenching").

## Output fields

| field | meaning |
|---|---|
| `O2_trop_D17O_permil` | the estimate (Young's preferred O(1D) Delta'17O = 27 permil). Unchanged. |
| `O2_trop_D17O_bound_high_permil` | less-negative end = the estimate. |
| `O2_trop_D17O_bound_low_permil` | more-negative end: Young's 45.9 permil O(1D) sensitivity case, adjusted to match the Barkan-Luz air-O2 target. |
| `O2_trop_D17O_bound_spread_permil` | width of the band. Small near modern, wide toward extreme states. |
| `O2_trop_D17O_bound_basis` | the literature range driving the band. |
| `O2_trop_D17O_bounds_note` | one-paragraph interpretation, including the saturation caps. |

## How to read it

- The **estimate** is the best single value (Young's preferred O(1D)).
- The **band** is the single dominant, citable uncertainty: the literature range
  of the O(1D) Delta'17O. It is tight near modern (~0.1 permil) and widens toward
  high pCO2 / low GPP / low pO2 - exactly where extrapolation is least certain.
- `bound_low` is a **conservative bound, not a 50/50 alternative**: Young shows
  the 45.9 permil anchor makes the modern CO2 isoflux about 4x the observed
  value, so the true value sits toward the estimate end.
- No state can pass the **saturation cap** (~-28 permil Young / ~-47 permil
  Barkan-Luz). If a reconstruction needs O2 Delta'17O beyond that, the cause is
  the O(1D) Delta'17O, not pCO2/GPP/pO2.

Example (50000 ppm, 0.5 PAL, 30% GPP): estimate -20.5 permil, band
[-34.8, -20.5] permil. Reported as "-20.5 permil (Young O(1D)), down to -34.8
permil at the Barkan-Luz O(1D) bound."
