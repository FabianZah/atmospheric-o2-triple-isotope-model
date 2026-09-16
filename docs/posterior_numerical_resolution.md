# Posterior numerical resolution

The atmospheric forward surface, isotope definitions, measurement errors and
coordinate priors are unchanged by these numerical improvements.

## Integrating an uncertain coordinate

For Gaussian air observations (including the existing converted-spherule
observation), a three-coordinate constrained calculation integrates the
coordinate absent from the displayed pair. This is pO2 when the displayed
pair is pCO2-GPP, or GPP when solving for pO2 with uncertain pCO2 and GPP.

For each control volume [a,b], the calculation evaluates

\[
 I = \int_a^b \exp\left[-\frac12\sum_i
       \left(\frac{F_i(x)-y_i}{\sigma_i}\right)^2\right]\pi(x)\,dx.
\]

The model predictions F_i, rather than the probability density, are
approximated as affine functions over each interval. Gaussian observations
and a uniform or Gaussian coordinate constraint then admit an analytic
Gaussian integral. The implementation uses log-CDF differences for stable
tail evaluation. Subdivision compares the integral and the midpoint forward
response with their coarser approximations. It preserves separate branches
of nonmonotone forward functions.

Relevant intervals require relative integral change <= 0.001 and midpoint
forward-response error <= 0.01 measurement sigma. Intervals more than 30 log
density units below the running maximum are treated as negligible tails.
Ten subdivision levels and 12 million forward points per integration bound
the work. Failure to converge raises an error rather than returning an
apparently completed calculation. Settings and achieved errors are returned
in `coordinate_integration_diagnostics` and recorded in XLSX metadata.

The internal joint density is a cell average along this integrated
coordinate. Its probability mass includes the physical cell width. The
displayed two-dimensional density is the resulting marginal per reported
coordinate units. A logarithmic display axis does not change the prior.

Sulfate retains its separate exact isotope-transfer/process likelihood and
convergence checks. It is not replaced by a Gaussian approximation in air
space. The new analytic atmospheric-coordinate integral currently applies
to Gaussian air observations only. General sulfate calculations with several
uncertain atmospheric and process parameters still require their own
convergence and resource-budget assessment.

## Plot and export consistency

For a three-coordinate Gaussian-air calculation, the marginal field is checked
for disconnected HPD support or more than 0.5% posterior mass in single-cell-wide
HPD portions. These indicators request a finer calculation; they do not establish
that multimodality is numerical. The unchanged likelihood is recomputed with at
least 721 pCO2/pO2 points and 961 GPP points in the occupied region. GPP and pCO2
sampling is geometric, with physical-coordinate quadrature weights. Outer coarse
nodes are retained, so concentration does not narrow the prior's domain.

The calculation streams small column batches and preserves the original nuisance
axis and its adaptive integration tolerances. The refined field supplies the
solved-coordinate marginal, all constrained-coordinate summaries, HPD threshold,
and XLSX data. Real separate regions remain separate. This is a triggered dense
recomputation, not cosmetic smoothing or a guarantee of convergence everywhere.
Diagnostics record its trigger, grid, evaluation count, achieved nuisance-integral
errors, elapsed time, and changes to the median/interval. Work is bounded at one
million field nodes, 150 million forward evaluations, and 300 seconds (checked
between batches); budget failure returns an error rather than a partial field.

Repeated nuisance-coordinate evaluation reuses the fixed two-dimensional factors
of the existing tensor interpolation. The Delta17O spline coefficients and the
local quadratic delta18O values are contracted along the fixed coordinates once
per batch, leaving one-dimensional evaluations. This is an algebraic rearrangement
of the same interpolators, with no new fit, smoothing, or tolerance change.
Unsupported diagnostic surfaces use the original evaluator. Tests compare both
isotopes at random points and boundaries for all three nuisance coordinates, and
compare integrated likelihoods with optimization disabled. A numerical resource
limit is reported explicitly by the API instead of as an unexplained server error.

One completed refinement is cached in memory for identical reruns and XLSX
requests. The complete input, model object and coarse field axes identify it.
Different inputs or a restarted/changed model require a new calculation.

Narrow two-coordinate HPD ridges retain their refined grid in the returned
field. If fourfold refinement still leaves disconnected or single-cell-wide
support, both axes are concentrated over the posterior mass and original outer
nodes are retained. The bounds entering this step and the coordinate priors are
unchanged. All marginals and credible regions are recomputed using physical
trapezoidal cell weights; the internal two-coordinate limit remains two million
cells. Genuine separate modes are not joined by the algorithm.
The former rounding-based compression assigned alternating counts of
fine nodes to coarse nodes: a constant input density could become 0.5625 to
1.5625 in adjacent cells. That compression has been removed.

The 95% region is obtained by ordering density in the reported physical
coordinate measure and accumulating probability mass. The browser draws its
actual density threshold; it does not smooth an envelope or dilate a binary
mask. PNG export uses the same canvas. The returned HPD mass and density
threshold refer to the returned field.

XLSX omits only nodes with both exactly zero mass and exactly zero density,
retaining first-row/column anchors so both full axes can be recovered. All
nonzero tails remain. Missing coordinate combinations represent exact zeros;
the storage convention is recorded in the workbook.

## Reproducible checks

Run from the repository root:

```console
python -m pytest validation/test_posterior_coordinate_quadrature.py validation/test_constrained_coordinate_resolution.py
python -m pytest validation/test_posterior_surface_slice.py
python validation/audit_probability_field_resolution.py
python validation/audit_low_po2_field_refinement.py
python validation/audit_low_po2_field_refinement.py --case high-gpp
python validation/audit_fixed_po2_field_resolution.py
```

The audit uses a synthetic atmospheric target of -10 per mil with 0.015 per
mil analytical sigma, delta18O = 23.9 +/- 0.3 per mil, GPP = 100 +/- 10% modern
and pO2 = 1.0 +/- 0.2 PAL. Independent 257- and 513-node trapezoidal pO2
integrals evaluate the unchanged forward surface, separately from the new
cell integrator. These are numerical checks, not observational validation or
estimates of structural model uncertainty.

The 2026-09-08 audit found integrated absolute differences of 0.0000349
between the normalized public field and the 513-node reference, and
0.00000251 between the two dense references. The public HPD region contains
0.950068 of the independent reference probability. The old 41-node sampling
has a normalized-field absolute difference of 0.5254 in this test.

The tests also cover modern-air integration over a bounded GPP range,
independent normal and uniform constraints, narrow between-node peaks,
multiple Gaussian observations, separate physical branches, far tails,
domain-boundary solutions and the uncompressed precise high-CO2 ridge.

The low-pO2 audit uses air D17O = -8 +/- 0.015 per mil, conventional d18O =
23.9 +/- 0.3 per mil, GPP = 20 +/- 10% modern and pO2 = 0.2 +/- 0.2 PAL.
On 2026-09-08 the conditional refinement changed five grid islands to one
connected region, with pCO2 median 5,672.62 ppm and central interval
[3,369.84, 9,264.89] ppm. The 832 x 961 field contained 0.950000716 probability
in its HPD region. An independent 721 x 961 check with a different nuisance
partition agreed within 0.003 ppm on those summaries. The public-path audit
took 226 seconds locally; this cost applies only when refinement is triggered.

The subsequent timeout regression uses D17O = -11 +/- 0.015 per mil, d18O =
23.9 +/- 0.3 per mil, GPP = 180 +/- 10% modern and pO2 = 0.2 +/- 0.2 PAL.
The original full calculation exceeded its 300-second refinement limit.
With fixed tensor factors reused, the same-resolution public request completed
in approximately 119 seconds locally (110 seconds of refinement). Its 888 x 1004
field had one connected HPD region with probability 0.950000995; median pCO2 was
43,822.69 ppm and the central interval [39,698.46, 49,318.72] ppm. The calculation
used 106,245,886 evaluations under the unchanged budget and integration tolerances.
Timings depend on hardware and load; this does not guarantee every constraint
combination finishes within the budget.

For the same -11 per mil observation with pO2 fixed at 0.2 PAL, full-domain
fourfold refinement still left CO2 spacings of about 438 ppm at the posterior
median. The concentrated 893 x 1420 grid reduces that spacing to about 23 ppm.
The resulting median is 44,011.23 ppm and central interval is
[40,587.82, 48,717.42] ppm. An independent uniformly spaced, full-domain
6001 x 3201 trapezoidal calculation agrees within 8 ppm on all three quantiles;
its probability above the public HPD threshold is 0.94996868. This is a numerical
quadrature validation against the unchanged surface, not additional validation
of model physics. The reproducible audit saves its inputs and diagnostic arrays.
