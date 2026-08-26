# High-pCO2 physical-process inventory

## Purpose

This inventory separates model processes into four categories: explicit in the
released central model, retained as structural end members, implemented only
for modern validation, or still absent. It prevents a model-model residual from
being mistaken for a missing physical term and prevents a sensitivity result
from silently changing the public model.

The machine-readable source is
`model_data/literature/high_pco2_process_inventory_v1.json`.

## Current boundary

The released engine already contains native altitude-dependent O/O(1D)/O2/O3/
CO2 photochemistry, pO2- and pCO2-dependent chemical columns, a globally mixed
O2 inventory, absolute GPP, source-backed biological alternatives, and the
Adnew molecular-isoflux normalization. Its fixed ModernEarth temperature and
Kzz profiles are the main atmospheric-structure simplification.

The paired Clima calculations are currently structural end members. They demonstrate
that high-pCO2 thermal and ozone reorganization can materially weaken R7
forcing, but it crosses past Young at 30,000 ppm and uses an isothermal
stratosphere without ozone heating. It is therefore evidence that architecture
matters, not a correction to apply. Repeating the calculation at fixed total
dry major-gas pressure changes global O2 Delta-prime-17O by at most 0.0404 per
mil through 30,000 ppm and 0.2270 per mil over the full domain. The large
climate response therefore survives the pressure-convention test.

The reduced 0.1, 1 and 2 PAL cross is also complete. Low pO2 amplifies the
high-pCO2 ozone and R7 response, while high pO2 partly buffers it. At 60,000
ppm, the Clima/fixed R7 forcing ratio increases from 0.262 at 0.1 PAL to 0.451
at 1 PAL and 0.493 at 2 PAL. The pO2 interaction is therefore retained as a
material structural sensitivity rather than averaged into one correction.

Modern ERA5 latitude-height circulation, diffusion, and convection operators
are useful validation assets. They are not deep-time boundary conditions and
are not part of the public steady inverse model. Similarly, Liu's air-sea
exchange convention is a structural comparison because its atmosphere-
accessible O2 flux is not identical to total global GPP.

## Experiment order

1. Completed: repeat the six-node 1 PAL Clima-chemistry calculation with fixed
   total dry major-gas pressure. The high-pCO2 result is robust to this
   convention and both runs remain structural end members.
2. Completed: calculate a reduced 0.1 and 2 PAL cross at 300, 10,000, 30,000
   and 60,000 ppm CO2. The pO2, ozone shielding and O(1D) interaction is
   material and all scenarios converge.
3. Next: replace or bracket the isothermal
   stratosphere with ozone-heated profiles. This addresses the most important
   known weakness of the first Clima setup.
4. Treat air-sea exchange and accessible marine O2 production as a separate
   biological-flux experiment. It requires a source-backed mapping to total
   GPP and may not be tuned to Young or Liu.
5. Use the 2-D transport engine only after a paleostate circulation ensemble or
   defensible reduction exists. Present-day ERA5 fields alone cannot justify a
   Phanerozoic transport correction.

Each experiment is diagnostic. A mechanism enters the central model only if it
is independently parameterized, conserves mass and isotopes, improves more
than one validation family, and preserves modern and low-pCO2 gates.
