# Photosynthesis experiment carbon and oxygen budget

The photosynthesis-at-fixed-respiration experiment integrates five inventories
together: three global O2 isotopologues, tropospheric CO2 (Ct) and stratospheric
CO2 (Cs). Production changes at time zero while the initial respiration rate
coefficients remain fixed. Respiration flux then changes with the O2 inventory.
Both biological configurations use this same coupling.

## Equations and units

Let P and R be the major-O2-equivalent photosynthetic production and respiratory
consumption fluxes, in mol/year. The existing biological formulation supplies
these fluxes and their isotope-specific equivalents. J denotes the atmospheric
photochemical tendency. For the major O2 inventory N:

```text
dN/dt  = P - R + J
dCt/dt = R - P + FV + FO + B - K*Ct + kST*Cs - kTS*Ct
dCs/dt = kTS*Ct - kST*Cs
K = m*(kW + kA)
```

The two rare-O2 inventories evolve with their corresponding biological and
photochemical tendencies. One evaluation of P-R is used with opposite signs
in oxygen and carbon. Consequently, the biological terms cancel exactly in
`d(N+Ct+Cs)/dt`. Atmospheric transport also cancels in the total carbon budget.
Photochemistry and the external carbon sources/sinks remain explicit. This is
an open atmospheric budget using the model's major-isotopologue-equivalent
production convention; it does not track all Earth-system carbon reservoirs.

## Literature coefficients

Young, Yeung and Kohl (2014), Table 2 and Section 3.5, provide the carbon terms:

| Quantity | Value | Role |
|---|---:|---|
| Weathering coefficient, kW | 0.000208 yr^-1 | First-order CO2 removal |
| Ocean uptake coefficient, kA | 0.1172 yr^-1 | First-order CO2 removal |
| Volcanic source, FV | 4.0e12 mol yr^-1 | Constant CO2 source |
| Ocean source, FO | 5.88e15 mol yr^-1 | Constant CO2 source |
| Stratosphere-to-troposphere exchange, kST | 1 yr^-1 | Carbon transport |
| Troposphere-to-stratosphere exchange, kTS | 0.1 yr^-1 | Carbon transport |
| Carbon-sink multiplier, m | 1.00 | Uses both removal coefficients as reported |

Young's footnote i attributes weathering to Kump et al. (2000); footnote j
attributes ocean exchange and volcanic input to Solomon et al. (2007).
These are adopted coefficients from Young's parameterization. Their suitability
for a particular nonmodern carbon cycle is a structural assumption.

Table 3 supplies 5.29e16 mol tropospheric CO2 at 294.4 ppm. Initial inventories
use the requested starting pCO2 directly:

```text
Ct0 = 5.29e16 * pCO2_initial / 294.4
Cs0 = (kTS/kST)*Ct0
pCO2(t) = 294.4 * Ct(t)/5.29e16
```

This fixed reference-atmosphere conversion is consistent with the atmospheric
box convention. Ocean uptake is a first-order exchange parameterization rather
than a dynamically resolved ocean carbon inventory.

## Initial balance and multiplier

The signed constant balancing flux is calculated separately:

```text
B = K*Ct0 - (FV+FO) + (P0-R0)
```

It closes the entered initial carbon state and is held fixed after the step.
The initial biological imbalance P0-R0 includes the small adjustment that
balances the initial photochemical major-O2 tendency. B is a conditional
initial-state closure, not an independently measured source or a fitted isotope
offset. It can become large for a high imposed initial pCO2; such experiments
describe relaxation around that imposed background.

The default m=1.00 retains Young's printed carbon coefficients. A value of 0.94
is available as an explicit sensitivity through
`UpdatedPhotosynthesisTransientInput.carbon_sink_multiplier`; it has no
independent physical calibration. Changing m recomputes B consistently. It
does not change biological production, respiration coefficients or isotope
normalizations. At 294.4 ppm with P0-R0=0, the unit-multiplier reference B is
3.268832e14 mol/year; actual run metadata include the run-specific balance.

## Slow reservoirs and domain

Slow geological O2 production and consumption are held at their balanced
initial background. Young's Section 3.6 discusses their longer, approximately
100,000-1,000,000-year response. The present experiment targets biological and
isotope relaxation conditional on that background. Its long-run result is the
equilibrium of this conditional system, not a geological steady-state forecast.
At low productivity or over long intervals, evolving weathering and organic
burial can matter and require separately specified flux laws and isotope inputs.

The joint ODE uses implicit Radau integration with positive, finite inventories
and completion checks. A trajectory leaving the supported atmospheric pO2/pCO2
domain is rejected. Boundary chemistry is used only for numerical stage probes;
accepted trajectories and exported samples must remain inside the domain.
The conditional equilibrium is obtained from a separate stationary solve of
these same five equations. The search horizon limits how long relaxation is
followed; an equilibrium time is reported only if both isotope coordinates
reach the specified tolerance within that horizon.

Results record the shared architecture, carbon inventories, initial production
and respiration rate, all carbon coefficients, multiplier, B and solver metadata.
The compatibility field `carbon_driver_po2_pal` contains the same O2 trajectory
as the global oxygen states. JSON/CSV rows and the time-response workbook expose
the two carbon inventories for inspection.

Steady forward/inverse/posterior calculations, isotope fields, ordinary state
steps and prescribed pCO2 trajectories retain their own existing formulations.
Archived v0.1.0 and dated source freezes preserve their historical behavior.

## Reference

Young, E. D., Yeung, L. Y., and Kohl, I. E. (2014). On the Delta-17O budget of
atmospheric O2. *Geochimica et Cosmochimica Acta*, 135, 102-125.
[doi:10.1016/j.gca.2014.03.026](https://doi.org/10.1016/j.gca.2014.03.026).
