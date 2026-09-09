# Conditional sulfate transfer

The Python sulfate module offers an exact isotope-atom transfer for a specified
primary-sulfate process. Its forward calculation predicts sulfate anomaly
conditional on sulfate delta18O; its inverse recovers atmospheric oxygen given
an explicit atmospheric delta18O. The console/Python calculation and the local
browser solver share the same transfer and uncertainty implementation.

All exact-transfer inputs use conventional delta18O relative to VSMOW and
logarithmic Delta-prime-17O with reference slope 0.528, in per mil. Inputs must
already have their applicable analytical calibration. No automatic partial-yield
correction is applied.

## Process assumptions

- `fraction` is the fraction of **all oxygen atoms** in the final sulfate
  inherited from atmospheric O2, after formation-stage exchange. It must refer
  to the same archive stage as the sulfate measurement/endmember.
- `background_cap_delta17_permil` is the anomaly of the effective non-air
  contribution after formation. It is not necessarily the anomaly of water.
- `alpha18_air_to_sulfate` is the isotope-ratio fractionation factor from
  atmospheric oxygen to its transferred contribution. The corresponding
  factor for 17O is either the independent `alpha17_air_to_sulfate`, with
  `theta_air_to_sulfate=null`, or the legacy
  `alpha18_air_to_sulfate**theta_air_to_sulfate`. Supplying both is rejected.
- `assumption_note` records the source, process and primary-preservation
  assumptions. Unknown resetting is not assigned a numerical correction.

There is no universal default for these coefficients. No fractionation means
both alpha18=1 and alpha17=1. With independent factors, alpha18=1 alone does
not imply a zero anomaly shift. In the legacy alpha/theta representation,
alpha18=1 sets both factors to one and theta has no numerical effect.
An experimental final incorporation fraction already includes exchange during
formation. Multiplying it by another formation-stage preservation factor would
count that exchange twice.

A separate [literature-pathway assessment](sulfate_fractionation_assessment.md)
quantifies the effects of omitting incorporation fractionation and tests this
compact transfer against the full Cao and Bao (2021) formation equations.

## Atom balance

Let `s`, `a`, and `b` be the 16O, 17O, 18O atom-fraction vectors of sulfate,
transferred atmospheric oxygen, and effective non-air oxygen. The equation is:

```text
s_i = f*a_i + (1-f)*b_i,  i = 16, 17, 18
R18_transferred = alpha18 * R18_air
R17_transferred = alpha17 * R17_air
R17_background = R17_VSMOW * exp(B/1000) * (R18_background/R18_VSMOW)**0.528
```

For the inverse, atmospheric delta18O specifies transferred R18. The only
remaining scalar is transferred 17O atom abundance. Positive atom inventories
bound that scalar, and the background relation supplies a monotone root.
The background delta18O follows from mass balance; users need not supply a
complete hypothetical air-free mineral composition. That implied delta18O
still needs to be geochemically plausible for the chosen archive.

The forward calculation instead uses the atmospheric model's paired delta18O
and Delta-prime-17O, together with measured sulfate delta18O. This avoids
assigning modern atmospheric delta18O to every ancient atmosphere. Sulfate
delta18O is a conditioning measurement, not an independently predicted
observable of this compact closure.

This exact atom-balance reduction is derived here. The experimental literature
supports the need to distinguish incorporation and fractionation; it does not
calibrate every coefficient of this reduction for all sulfate archives.

## Console example

From the repository directory, this **illustrative scenario** assumes primary
preservation, 10% incorporation, B=0 and no air-path fractionation. These are
specified assumptions, not recommended defaults for a sulfate sample:

```text
python run_model.py calculate sulfate --d17o -1 --d18o 15 --fraction 0.1 --air-d18o 23.9 --background-d17o 0 --alpha18 1 --theta 0.528 --assumptions "Illustrative preserved-primary scenario; zero B and no air-path fractionation" --output outputs/sulfate_example.json
```

The output includes the full-air isotope pair, transferred-air pair, implied
non-air pair, all assumptions, reference coordinate and atom-closure residual.
It is a conditional result, not a posterior or a calibrated pCO2 estimate.

The Python entry points are `exact_air_from_sulfate` and
`exact_sulfate_from_air` in `code/sulfate_to_air.py`.

## Atmospheric inference with uncertainty

An independently entered incorporation constraint may be fixed, a uniform
range, or a normal distribution restricted and renormalized to [0,1]. A range
is not automatically a measured probability distribution. An inverse endpoint
at zero incorporation is unbounded; the forward response at zero contains no
atmospheric information. The background anomaly independently accepts a fixed
value, normal uncertainty or uniform range in the same isotope coordinate.

`code/sulfate_uncertainty.py` evaluates the probability of the **measured sulfate**
given each atmospheric model prediction. It integrates incorporation, background
and both isotope measurement errors through the exact forward atom balance.
The atmospheric model supplies its paired anomaly and conventional delta18O at
every candidate pCO2-GPP-pO2 combination. An optional analytical correlation
between sulfate anomaly and delta18O is represented explicitly.

The statistical assumptions are:

- The two analytical errors are jointly Gaussian. Sulfate delta18O is integrated
  with a locally flat prior on its latent true value. Its conditional treatment
  reflects that this compact transfer does not independently predict sulfate
  delta18O.
- Incorporation, background and analytical errors are independent apart from
  the explicitly supplied analytical correlation.
- Air-path isotope factors and primary preservation remain conditional process
  assumptions. They are not silently calibrated or assigned distributions.
- Incompatible oxygen-atom inventories contribute zero likelihood. The
  remaining incorporation/background prior mass is not renormalized to hide
  that conflict. Implied non-air delta18O still needs archive-specific review.
- The local console inference uses uniform atmospheric-coordinate priors in
  reported units unless explicitly changed in its request. These priors are
  separate from the sulfate likelihood. Bounds and any independent geological
  constraints are recorded in the output.

The implementation analytically integrates a locally linear forward response
between exact atom-balance roots. It compares coarse and midpoint-refined cell
integrals, sums their absolute differences with the original prior weights,
and requires two successive integral refinements to agree. This prevents narrow measurement
likelihoods being missed between incorporation quadrature points. It does not
replace the inferred air distribution with a Gaussian. Unconverged integration
raises an error. Atmospheric-grid convergence is a separate check, particularly
for narrow posteriors and multiple uncertain atmospheric coordinates.

When both incorporation and non-air composition are uncertain, their estimated
response widths select the initial integration order. States that have not
converged are retried with the order reversed. Both orders evaluate the same
independent process distributions and exact atom-balance response. The retry
requires two fresh successful refinement checks at the original tolerances and
shares the request's time and root-evaluation budgets. The exported
`alternate_integration_order_states` diagnostic records how many states needed
this retry; a failure of both orders still returns no posterior.

For larger inference grids, `sulfate_likelihood_table.py` evaluates the exact
measurement likelihood in air-anomaly / conventional-air-delta18 coordinates.
The atmospheric model enters this calculation only through those two predicted
isotope values. A bicubic interpolant of `log1p(compatibility / 1e-10)` is
checked at new edge midpoints, cell centers, and requested atmospheric states;
unresolved intervals are subdivided. The compatibility is the likelihood
multiplied by the conditional analytical sigma and `sqrt(2*pi)`. Direct node
integration uses a relative tolerance of `5e-5` and absolute tolerance `2.5e-11`;
interpolation is checked at `1e-4` relative and `7.5e-11` absolute compatibility.
Interpolation stays within its checked isotope rectangle. A tail-only request
uses direct integration so that the absolute tolerance cannot create a
posterior from negligible evidence. This is numerical reuse of the sulfate
likelihood, not an additional physical model or smoothing of the posterior.

The optimized node integrator uses observation-focused quadrature panels while
retaining the full entered fraction interval and its original probability
density. Initial panel locations use a first-order transfer estimate only as a
sampling proposal; all node values use the exact isotope-atom balance. Gaussian
fraction constraints are integrated in physical fraction coordinates, with their
normalization on [0, 1] retained. Panels follow the valid atom-inventory boundary
for each latent sulfate delta18 value. Incompatible prior mass remains excluded
without renormalization. Near-100% incorporation receives extra endpoint panels.
The inner cell resolution and outer quadrature orders refine independently,
subject to the same prior-weighted error estimate.
The delta18 quadrature starts at order five and is checked at order nine before
acceptance. A Hermite rule is tried first. Unresolved near-pure-air fraction
support can retry with composite Gaussian-weighted Legendre panels split where
sulfate and transferred-air delta18 match. Failed isotope checks increase its order
independently. At a promoted order both preceding process-grid comparisons are
rechecked. Remaining failures trigger the full refinement route. Exported diagnostics record the node rules,
weighted cell-error estimate, interpolation checks, and any retries.

Checked tables are retained only within one calculation's budget context and
reused only for identical sulfate inputs inside the existing isotope bounds.
Both the root-evaluation and wall-clock limits remain active. At most two tables
are retained per request. Small inversions use direct integration; numerical
failure never returns an unfinished probability field.

The required sulfate anomaly sigma must be positive. Fixed 100% air inheritance
belongs in a joint direct-air observation treatment; it is excluded from this
conditional sulfate likelihood. Pure-air and zero-error forward/inverse
calculations remain available in the scalar transfer module.

### Console request

Use the included **synthetic example**, whose fixed atmospheric truth was
pCO2=10,000 ppm, GPP=72.5 PgC/year and pO2=0.5 PAL:

```text
python run_model.py calculate sulfate-infer --request docs/sulfate_inference_example.json --output outputs/sulfate_inference_example.json
```

The example infers pCO2 with GPP/pO2 fixed, incorporation uniformly distributed
from 0.15 to 0.25, and B=-0.02 +/- 0.03 per mil. These values illustrate the
interface; they are not recommended archive defaults. JSON isotope values are
in per mil, GPP is absolute PgC/year, and incorporation is a fraction, not a
percentage. Normal `sigma` values mean 1 SD. The `assumption_note` is required.

The result includes coordinate densities, medians, credible intervals, full
input/process metadata and likelihood-convergence diagnostics. There is no
fabricated air target or single effective air sigma. Positive air-space model
discrepancy is rejected for this observation type until its forward propagation
is implemented explicitly. This is conditional measurement/process uncertainty,
not total model uncertainty.

Python callers can pass `SulfateLikelihoodInput` as `sulfate` to
`UpdatedJointPosteriorInput` or `ConstrainedCoordinateInput`. Set direct-air
target/error fields to `None`. The latter accepts the existing fixed, normal
or range constraints on the other atmospheric coordinates.

### Browser and API

Choose **Sulfate** in the Model solver's isotope-source selector. Editable
example values are Delta-prime-17O = -0.200 per mil, delta18O = 15.000 per mil,
with illustrative analytical sigmas of 0.030 and 0.500 per mil. These are not
reference measurements. Cleared isotope fields display **Insert value** and
remain required; a blank entry is never interpreted as zero.

Air-derived oxygen is displayed as a percentage of all sulfate oxygen; its
sigma is in percentage points. The **Non-air oxygen component** anomaly is
in per mil and retains the API key `background`. These two process inputs
remain blank until supplied for the archive. Non-air delta18O is derived by
atom balance, as described above, rather than independently constrained.

The **O2 incorporation treatment** selector offers:

| Treatment | O2 contribution | Source-oxygen fractionation |
| --- | --- | --- |
| No fractionation assumed (unchanged default) | User constraint | Both factors equal one |
| Irreversible incorporation, 25 C | Fixed 25% | Peng et al. (2026), L18=-21.6 per mil, theta=0.5145 |
| Equilibrated incorporation, 25 C | Fixed 25% | Peng et al. (2026), L18=-3.5 per mil, theta=0.5199 |
| Custom isotope shifts | User constraint | Independently entered L18 and anomaly shift at slope 0.528 |

Here L18=1000*ln(alpha18). Custom settings use the logarithmic delta18 shift
and anomaly shift, both transferred O2-derived oxygen minus atmospheric O2.
The factors are `alpha18=exp(L18/1000)` and
`alpha17=exp((anomaly_shift+0.528*L18)/1000)`.
This supports zero L18 with a nonzero anomaly shift. It is not a direct
additive correction to the bulk sulfate measurement.

The Peng treatments constrain the O2 side of sulfite oxidation. They keep
that pathway's 25% incorporation fixed; they do not prescribe the effective
non-air sulfate composition, preservation, or reaction reversibility from
atmospheric pO2. The non-air component still requires an input. Switching back
to a user-constrained treatment restores the earlier incorporation fields.
Wei et al. (2026) precursor-equilibrium values have not been substituted for
the final non-air component, because sulfite-side oxidation effects remain.

Browser calculations use independent analytical errors (correlation fixed at
zero), recorded in the request and XLSX metadata. The
Python/API interface retains explicit covariance for analyses with laboratory
covariance information.
Preservation of a primary sulfate signal remains a model assumption recorded
in metadata; the form does not require a confirmation checkbox or free-text
note and the calculation does not assess preservation.

`POST /api/v1/inference/coordinate` accepts a nested `sulfate` observation
instead of the direct-air isotope fields. The observation has the same fields
as the console's `sulfate` object, plus `primary_signal_assumed`, which defaults
to true and denotes the assumed model scope. `assumption_note` is optional in
the API; the solver records a neutral preservation assumption, with any
provided note appended. Neither is treated as verified sample evidence. API
incorporation values are **fractions, 0-1**, unlike the percentage-facing form.
The ordinary atmospheric-coordinate constraint contract is unchanged.

For a named treatment, the API accepts `fractionation_treatment` as `none`,
`peng_2026_irreversible` or `peng_2026_equilibrium` with no isotope factors;
the server resolves the published coefficients. Supplied factors, if present,
must be a complete matching pair. The default `specified` treatment retains
backward-compatible alpha18/theta requests and supports independent alpha17.
Named Peng cases reject any incorporation constraint other than fixed 0.25.
The Python likelihood validates the same coefficient and incorporation
contracts, including for console requests that provide the resolved factors.

The XLSX download contains a sulfate-specific summary, the process inputs,
likelihood settings and convergence diagnostics in **Sulfate transfer**, and
the posterior arrays. Transfer metadata records the treatment, both ratio
factors, both log shifts, the anomaly shift, temperature and source DOI where
applicable. The theoretical 2SE values are not added as geological process
priors. User notes remain text, not spreadsheet formulas.

An XLSX download reuses the completed coordinate result when it remains in
the server's bounded cache. Expired or evicted results are recalculated.
See [constrained coordinate inference](constrained_coordinate_inference.md)
for the retention and resource limits.

Browser sulfate requests share one calculation budget across all internal
refinements: 600 seconds and one billion exact forward-root evaluations.
A resource limit returns HTTP 503 with code `sulfate_calculation_limit`;
neither a limit nor a convergence failure returns an unfinished interval.
Local Python/console runs are not subject to this browser budget.

Forward roots are evaluated in bounded batches, skipping only zero-weight
quadrature entries. Batching changes execution overhead, not isotope balance.
The combined-uncertainty audit exercises fixed, normal and range constraints,
including low oxygen, large isotope errors, zero-inclusive incorporation ranges,
and all three solved atmospheric coordinates. It records wall-clock time,
posterior intervals, numerical diagnostics and independent direct-likelihood
checks. Timings depend on the host; server throughput requires separate testing.

```text
python validation/audit_sulfate_uncertain_combinations.py
```

Its output records a hash of the calculation code and audit. The optional
`--resume` reuses successful cases only when that hash is unchanged. The audit
uses a 180-second per-inference test limit by default (`--max-seconds` overrides
the test allowance, within the same root-evaluation ceiling); independent reference checks are
timed separately and do not count toward that limit.

## Checks and historical comparisons

The first-order comparison implements `S = f*(A+T) + (1-f)*B + bias` in a
retained native isotope coordinate. Its analytical likelihood integrates
`p(measured sulfate | predicted air, f)` over f, rather than dividing the
measurement and replacing its result with a Gaussian air estimate. That
likelihood is a **first-order reference only**, retained alongside the exact
integration as a numerical comparison.

Historical linear/logarithmic anomalies retain their definitions. The historical
photochemical-component scope is explicitly excluded from full-air conversion.
Paired delta18O enables a change of reference slope; it does not resolve
analytical calibration or transfer uncertainties.

Run:

```text
python validation/audit_sulfate_transfer.py
python validation/audit_sulfate_transfer.py --inference --output outputs/sulfate_inference_audit.json
python -m pytest validation/test_sulfate_to_air.py validation/test_sulfate_transfer_audit.py validation/test_sulfate_uncertainty.py
```

The audit reproduces the published scalar examples and tests synthetic exact
atom mixtures. Under its stated no-fractionation scenarios, simply dividing
log-0.528 sulfate anomaly by f introduces up to 0.130 per mil error in recovered
air at -10 per mil, and 0.639 per mil at -40 per mil. Exact transfer removes
this numerical approximation. These chosen synthetic cases are neither
empirical model-error estimates nor archive uncertainty distributions.

The optional inference audit refines atmospheric grids independently of sulfate
likelihood integration. Its three synthetic cases recover the generating pCO2
inside their 95% intervals and check that the final quantile changes are below
1% of interval width. Narrow analytical-only posteriors require more atmospheric
grid points than broad process-uncertainty examples. The tests also compare
the sulfate integration to independent scalar/tensor quadrature, exercise
correlated errors and zero incorporation, and solve each atmospheric coordinate.

## Sources

- Peng, Han and Cao (2026). Theoretical evaluation of kinetic triple oxygen
  isotope fractionation during sulfite oxidation by atmospheric oxygen.
  *Geochimica et Cosmochimica Acta* 412, 64-74.
  [DOI](https://doi.org/10.1016/j.gca.2025.11.036). O2-side reaction limits;
  converted from the source's slope 0.5305 to 0.528 using both isotope factors.
- Wei, Yan and Fang (2026). Experimental determination of equilibrium
  fractionation of triple oxygen isotopes between dissolved sulfite species
  and water. *Earth and Planetary Science Letters* 679, 119862.
  [DOI](https://doi.org/10.1016/j.epsl.2026.119862). Precursor constraints;
  no final-sulfate non-air default is inferred from these alone.
- Kohl and Bao (2011). Triple-oxygen-isotope determination of molecular oxygen
  incorporation in sulfate produced during abiotic pyrite oxidation (pH=2-11).
  *Geochimica et Cosmochimica Acta* 75, 1785-1798.
  [DOI](https://doi.org/10.1016/j.gca.2011.01.003). Eq. 8, Eq. 13 and conclusions
  distinguish final source fractions from isotope fractionation.
- Bao, Lyons and Zhou (2008). Triple oxygen isotope evidence for elevated CO2
  levels after a Neoproterozoic glaciation. *Nature* 453, 504-506.
  [DOI](https://doi.org/10.1038/nature06959). SI sections 2-5 give the native
  isotope definitions and their sample-specific analytical correction.
- Cao and Bao (2013). Dynamic model constraints on oxygen-17 depletion in
  atmospheric O2 after a snowball Earth. *PNAS* 110, 14546-14550.
  [DOI](https://doi.org/10.1073/pnas.1302972110). Their Svalbard calculation uses
  an inferred weathering endmember and linear 0.52 notation.
- Waldeck et al. (2022). Calibrating the triple oxygen isotope composition of
  evaporite minerals as a proxy for marine sulfate.
  *Earth and Planetary Science Letters* 578, 117320.
  [DOI](https://doi.org/10.1016/j.epsl.2021.117320). Fig. 3 illustrates
  water-equilibrium and archive-specific resetting effects.
- Waldeck et al. (2025). Marine sulphate captures a Paleozoic transition to a
  modern terrestrial weathering environment. *Nature Communications* 16, 2087.
  [DOI](https://doi.org/10.1038/s41467-025-57282-y). Fig. 2 and accompanying
  discussion distinguish atmospheric composition and oxygen inheritance.
