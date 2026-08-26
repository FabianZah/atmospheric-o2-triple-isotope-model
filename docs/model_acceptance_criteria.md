# Model Acceptance Criteria

This project now has two different targets that should not be mixed up.

## 1. Young-Reproduction Track

Purpose: recover the Young et al. 2014 model closely enough that deviations
identify missing equations or conventions.

Reasonable tolerances:

```text
Printed Table 3 isotope values          <= 0.02 to 0.05 permil
Digitized Fig. 7/Fig. 8 curves          <= 0.1 to 0.2 permil typical residual
Figure-only/extrapolated regions        correct shape and monotonicity first
CO2 anomaly flux / rate sensitivities   within about 10-20%
Large perturbation sensitivities        correct sign and order, then improve
```

For true printed table values, an exact reconstruction should eventually be
closer than 0.01 permil. But the current goal is not arbitrary exact fitting;
it is to find the remaining missing equations. A model that matches one table
to 1 per meg but breaks Fig. 8, Fig. 9, or Section 6 is not better.

## 2. Usable Physical Model Track

Purpose: produce a model suitable for exploring pCO2, GPP, and pO2 in earlier
Earth-history settings, including cases outside Young's plotted range.

Modern atmospheric O2 Delta17O validation interval: use Pack (2021)'s reviewed value of
Delta prime 17 O = -0.432 +/- 0.015 permil for air O2, rather than forcing the
physical model to Young et al.'s internally reported Table 3 value of -0.410
permil. Young's value remains a useful Young-reproduction constraint, but it is
not the preferred modern validation target for the final physical model. Pack's
uncertainty is kept separate from model uncertainty, and no output offset is
applied to force agreement.

Reasonable tolerances:

```text
Modern calibration                  within about 0.05 permil for O2 Delta17O
Published Young trends              visually and mathematically similar
pCO2/GPP inversions                 stable to parameter sweeps, no false hooks
pO2 perturbations                   physically interpretable budgets
Literature-improved terms           explicit and switchable, not hidden tuning
```

Practical pCO2 envelope:

```text
<= 30,000 ppm    Young Fig. 8 validation range
> 30,000 ppm     exploratory extrapolation for very low Delta17O values
```

Realistically, the final model does not need to make extremely high pCO2 a
routine Phanerozoic or Proterozoic scenario. However, the interface should
allow high pCO2 exploration because very low Delta17O values such as -10 per
mil may require the plot to extend beyond Young's 30,000 ppm figure range. The
requirement is therefore: label the region as extrapolative and avoid
unphysical hooks or sign reversals.

For this track, a few tens of per meg in Delta17O are acceptable if the model
has the right sensitivities and the equations are physically defensible. A
few ppm/per meg is smaller than the uncertainty from digitized figures and
from several biological/geochemical parameters. It is not worth sacrificing
physical behavior to hit those values.

## Practical Rule

I will treat deviations like this:

```text
<= 0.02 permil        excellent for printed numerical outputs
0.02-0.05 permil      acceptable for modern calibration
0.05-0.2 permil       acceptable for digitized figures if shape is right
> 0.2 permil          needs explanation or flags a missing term
wrong sign/curvature  not acceptable even if one point matches
```

The scorecard remains useful, but it should not be the sole judge. For the
final model, behavior across multiple independent Young constraints and
physical budget interpretability matter more than exact reproduction of a
single plotted or text value.
