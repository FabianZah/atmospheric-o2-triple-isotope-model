# I-type spherule to atmospheric O2 calibration

The public inversion uses Zahnow et al. (2025), Eq. 3:

```text
Delta'17O_air = Delta'17O_spherule + 0.0285 * delta18O_spherule - 1.005 per mil
```

The plus sign before the `delta18O` term is required by the regression geometry
and reproduces the modern MA-9 result reported by Zahnow et al. (2025).
Fischer et al. (2021), Eq. 7, prints a minus sign, but that sign is inconsistent
with its reported regression (`Delta'17O_spherule = -0.0285 * delta18O + a0`)
and does not reconstruct modern air. The model therefore follows the later
Zahnow equation.

Fischer et al. (2021) report a regression slope of `-0.0285 +/- 0.003` and an
intercept of `0.57 +/- 0.13 per mil` (1 sigma). Their slope-intercept covariance
is not reported. The model consequently keeps two uncertainty products separate:

- analytical measurement uncertainty, propagated through Zahnow Eq. 3;
- an independent parameter-corner sensitivity envelope using the reported
  slope and intercept errors.

The second product is not a statistical confidence interval and is not combined
quadratically with analytical error. A statistically complete calibration
uncertainty requires the original regression covariance or source data.
