# Sulfate incorporation-fractionation assessment

This reproducible sensitivity exercise compares the compact sulfate transfer
with the explicit formation pathways of [Cao and Bao (2021), *Small Triple
Oxygen Isotope Variations in Sulfate: Mechanisms and Applications*, Reviews in
Mineralogy and Geochemistry 86, 463-488](https://doi.org/10.2138/rmg.2021.86.14).
It evaluates synthetic sulfate and central atmospheric-model roots. It does
not calibrate a geological archive or assign a process-error probability.
The public model and its defaults are unchanged.

## Source constraints

Equations 13 and 15-18 and Table 1 are implemented directly in the audit:

| Pathway | Oxygen from O2 | Main constraints |
| --- | ---: | --- |
| Aqueous sulfite oxidation by dissolved O2, Eq. 16 | 25% | Sulfite-water equilibrium followed by sulfite and O2 kinetic isotope effects |
| Surface thiosulfate oxidation, Eq. 15 | 0% | Water-derived oxygen with equilibrium and kinetic effects |
| Aqueous sulfite oxidation by Fe3+, Eq. 17 | 0% | Water-derived oxygen with different kinetic effects |
| Sulfate-water equilibrium, Eq. 13 | 0% | Eq. 6 at 25 C, rather than the rounded Table 1 value |

For Eq. 16 the central oxygen-18 factors are 1.0129 for sulfite-water
equilibrium, 0.9916 for sulfite oxidation, and 0.9850 for O2 incorporation.
The authors adopt theta=0.524 for equilibrium and 0.511 for kinetics.
These are conditional estimates from selected acidic, long-duration
experiments; both exponents require assumptions (pp. 476-477).
The table footnotes and text report differing experimental isotope effects.
Consequently the coefficients are not interchangeable universal constants.

For isotope i, the printed Eq. 16 is:

```text
Ri_sulfate = 0.75 * alpha_i_eq * KIE_i_sulfite * Ri_water
          + 0.25 * KIE_i_O2 * Ri_dissolved_O2
```

The Figure 5 reproduction retains the printed ratio sum, dissolved O2 inputs
(delta18=24.2 per mil, Delta-prime-17O=-0.554 per mil at slope 0.5305), and
Eq. 18 water line. That line has Delta-prime-17O=+0.033 per mil at slope
0.528, including when water delta18=0. Water at that point on this line is
therefore not identical to the VSMOW reference composition.

Inverse experiments instead conserve the exact 16O, 17O and 18O atom
abundances, with the stated fractions referring to all oxygen atoms. Across
the tested Figure 5 range, this changes the anomaly by at most
0.00000469 per mil relative to the printed ratio sum.

## Reduction to the compact transfer

The effective non-air component is water after its equilibrium and kinetic
effects. Its anomaly B is therefore:

```text
B = Delta_water + 1000*(theta_eq - 0.528)*ln(alpha18_eq)
                + 1000*(theta_kin - 0.528)*ln(KIE18_sulfite)
  = +0.125133 per mil for the central Eq. 16 coefficients and Eq. 18 water.
```

The O2 path retains its separate fractionation. Its individual anomaly shift
is `1000*(theta_kin - 0.528)*ln(KIE18_O2)`. The complete inverse effect also
depends on the oxygen-18 atom balance and nonlinear isotope mixing, so that
single shift is not an exact correction to the sulfate measurement.

Measured sulfate delta18 and each candidate air isotope pair determine the
implied non-air delta18 by atom balance. The generating water delta18 is not
imposed as a separate inverse observation. This tests the existing compact
closure, rather than adding a hidden water-composition constraint.

Gas dissolution is kept explicit: one controlled comparison assumes identical
air and dissolved ratios solely to isolate formation effects. A second uses
constant factors derived from the paper's quoted air and dissolved isotope
pairs. This is a sensitivity bridge, not a temperature/salinity-dependent
gas-exchange model. No dissolved-air observation is silently treated as air.

## Results

For 25% incorporation, water delta18=0 on Eq. 18, air delta18=23.9 per mil,
and the controlled identity gas transfer, all anomalies below use log-0.528:

| Generating air anomaly | Generated sulfate anomaly | Air inferred omitting O2-path fractionation, same B | Air inferred also setting B=0 |
| ---: | ---: | ---: | ---: |
| -0.432 | +0.050 | -0.119 | +0.252 |
| -10.000 | -2.337 | -9.610 | -9.236 |

At the modern-sized air anomaly, omitting fractionation biases air by
+0.313 to +0.390 per mil across generating water delta18=0 to -20 per mil.
The bias is still present at very negative anomalies, although smaller as a
fraction of the total air anomaly. The water-only pathways contain no
atmospheric information in these endmember cases.

With both modelled air isotope coordinates, fixed 1 PAL O2 and 100% modern
GPP (290 PgC/year), the corresponding CO2 examples are:

| Generating pCO2, ppm | Matched transfer | Omit air-path fractionation, same B | Also set B=0 |
| ---: | ---: | ---: | ---: |
| 294 | 294 | No root | No root |
| 1,000 | 1,000 | 354 | No root |
| 10,000 | 10,000 | 9,001 | 7,968 |
| 30,000 | 30,000 | 28,009 | 26,180 |
| 60,000 | 60,000 | 56,059 | 52,687 |

These rows use water delta18=0 and identity gas transfer. "No root" means
none in the current 50-60,000 ppm domain at those fixed GPP/pO2 values; it
does not establish a physical CO2 value outside the domain. The modern
294-ppm model state is generated by the current model, not forced to -0.432.

The JSON also includes 25% GPP, 0.5 PAL O2, the second gas-transfer assumption,
and separate parameter-sensitivity scenarios. No independent Gaussian priors
are invented for the fitted kinetic coefficients or their unknown covariance.

Numerical checks:

- All 80 matched synthetic atmosphere/sulfate cases recover the generating
  CO2, with maximum error below 0.000001 ppm; this is an algebraic round-trip
  check, not independent validation of geological applicability.
- Maximum matched air-anomaly error is below 0.00000001 per mil.
- Refining root-bracketing grids from 513 to 1,025 points retains tested
  roots and no-root cases. Multiple sign-changing roots are retained.
- Four uncached central-atmosphere checks converge. Their differences from
  the accelerated surface are at most 0.000465 per mil in anomaly and
  0.00551 per mil in conventional delta18, much smaller than these transfer
  effects. They are checks at four states, not a new global surface audit.

## Implication for the interface

The existing compact transfer is sufficient to represent this pathway; extra
sulfate ODEs are unnecessary for that purpose. Incorporation fractionation
cannot generally be discarded, especially for small anomalies. Conversely,
assigning alpha18=0.985 and theta=0.511 to every sulfate is not supported.

A future named formation treatment should keep fraction, non-air component,
and fractionation mutually consistent. In particular Eq. 16 specifies f=0.25;
varying f freely with its other coefficients fixed would no longer reproduce
that pathway. Mixtures of pathways would require consistent non-air terms.
Samples with unknown resetting still need archive-specific constraints.

## Full-source assessment of the 2026 studies

The complete Wei and Peng articles, including equations, tables, figures and
discussion, have now been inspected. Peng's supplied archive contains 11
Gaussian output logs and Table S1, a geometry workbook. These were inspected
read-only; Gaussian was not rerun. Wei states that all data are in Table 1.
The audit retains source hashes and separately evaluates the two studies'
source terms. Neither has been inserted into the public model or UI defaults.

Both articles use the logarithmic reference slope 0.5305. All transferred
anomalies below use 0.528. Define `Li = 1000*ln(alpha_i)`, giving:

```text
shift_528  = L17 - 0.528*L18
shift_528  = shift_5305 + (0.5305 - 0.528)*L18
alpha_i   = exp(Li/1000)
```

In particular, a reported log fractionation of -21.6 per mil means
`alpha18=exp(-0.0216)`, not `1-0.0216`. The latter would interpret a different
isotope convention.

### O2-derived oxygen: Peng et al. (2026)

[Peng, Han and Cao (2026), *Theoretical evaluation of kinetic triple oxygen
isotope fractionation during sulfite oxidation by atmospheric oxygen*, GCA
412, 64-74](https://doi.org/10.1016/j.gca.2025.11.036) evaluates the
O2-consuming reaction `SO3 radical + O2 -> SO5 radical`, within a larger
sulfite-oxidation chain (Eq. 1). The complete pathway supplies one quarter
of sulfate oxygen from O2. The study provides intrinsic isotope effects and
a reaction-reversibility interpretation, rather than a fitted law for
selecting reversibility from atmospheric pO2.

At 25 C:

| Case | L18, per mil | Isotope exponent used | Shift in O2-derived oxygen anomaly, per mil at 0.528 |
| --- | ---: | ---: | ---: |
| Irreversible forward step | -21.6 | 0.5145 | +0.29160 |
| Reported SO5-site/O2 equilibrium | -3.5 | 0.5199 | +0.02835 |
| Fig. 7 pyrite example | -9.8 | Composite construction | +0.11932 |
| Fig. 7 apparent inverse example, lower | +7.1 | Composite construction | -0.12742 |
| Fig. 7 apparent inverse example, upper | +23.3 | Composite construction | -0.36394 |

These are individual source-oxygen shifts, before mixing with the non-air
oxygen. They are neither whole-sulfate corrections nor bounds on all samples.
The positive apparent fractionations are experimental comparison cases used
in Fig. 7, not confidence limits. Reported errors in this study are **2SE**
across theoretical configurations (four forward transition states, three
equilibrium configurations). They are not 1SD geological process priors.

The schematic Fig. 7 construction follows the backward-process slope 0.5134
from the forward kinetic endpoint, rather than from unchanged air:

```text
L17_apparent = 0.5145*(-21.6) + 0.5134*(L18_apparent + 21.6)
```

This reproduces the figure's approximately -0.154, -0.356 and -0.922 per mil
values from its assumed air value of -0.5 per mil at the native 0.5305 slope.
The separately reported equilibrium exponent and the rounded schematic line
differ by 0.00101 per mil at L18=-3.5; both remain explicitly recorded.
The figure is a source-term construction, not a reaction-rate simulation.

An important compact-model consequence: at apparent L18=0 the construction
still gives L17=-0.02376 per mil. A finite `alpha18, theta` pair cannot
represent `alpha18=1` with `alpha17!=1`. Two independent log shifts (or
isotope-ratio factors) can, without adding any ODE. The local implementation
now accepts independent factors through the scalar, batched, likelihood and
API paths, alongside backward-compatible alpha/theta inputs. The Peng forward
and equilibrium limits are conditional UI treatments; the schematic
reversibility line remains in this audit rather than becoming an automatic
pO2-dependent correction.

The supplement supports checking molecular configurations, but full kinetic
reproduction also requires the isotope-specific imaginary-frequency and
symmetry terms, not just ratios of Table 1 polynomial partition functions.
One equilibrium-conformer output (SO5-B) contains a low imaginary mode,
-15.2735 inverse cm. Its treatment would need clarification for an independent
partition-function calculation; this does not by itself invalidate the
published effective factors. We retain this as a source-reproduction question.

### Water-derived precursor: Wei et al. (2026)

[Wei, Yan and Fang (2026), *Experimental determination of equilibrium
fractionation of triple oxygen isotopes between dissolved sulfite species and
water*, EPSL 679, 119862](https://doi.org/10.1016/j.epsl.2026.119862) measures
sulfite/bisulfite-water equilibration at 12, 25, 40 and 55 C and four pH
values from 4.60 to 8.89. Table 1 reports two replicates per condition and
five water measurements, with **1SD** errors. Bisulfite denotes the sulfite
precursor species, not bisulfate in the final sulfate system.

The audit transcribes all 16 Table 1 conditions and preserves both the
printed compositions and printed differences, which can differ by 0.001
per mil when subtracting rounded values. It also retains Eqs. 1-4 and the
reported mean exponents as separate representations. At 25 C:

| Precursor | L18 from Eq. 1 or 2, per mil | Reported mean theta | Anomaly shift using mean theta | Anomaly shift using paired Eqs. 1/3 or 2/4 |
| --- | ---: | ---: | ---: | ---: |
| Bisulfite | 14.87936 | 0.5202 | -0.11606 | -0.14398 |
| Sulfite | 9.54297 | 0.5155 | -0.11929 | -0.12795 |

The last two columns are per mil at 0.528, precursor minus water. The
reported mean exponents give similar anomaly shifts despite different
oxygen-18 fractionations. However, the rounded paired equations do not
exactly recover those exponents: their 25 C differences are 0.02792 and
0.00866 per mil. Independent last-digit coefficient rounding could cover
these differences at 25 C, but the paper does not establish that explanation.
Full-precision regression coefficients and their joint covariance would
resolve this before selecting a high-precision implementation. We do not
average the representations or treat their difference as a Gaussian error.

These are precursor equilibrium effects. Final sulfate may add kinetic
effects to the sulfite-derived oxygen, so neither column alone defines the
effective non-air component B used by the compact inverse model. In
particular, substituting Wei's exponent into the shared 2021 kinetic exponent
would inadvertently change a different physical process.

Wei Section 5.1 applies a -0.1 per mil analytical correction to selected
previous datasets in its comparison. That is scale/method-specific and has
not been applied to user measurements or all sulfate data.

### Decision for a compact sulfate treatment

The atom-balance architecture remains sufficient. The new work supports
generalizing the effective O2 transfer to two isotope shifts and evaluating
conditional formation treatments; it does not require a sulfate reaction
network inside the public atmospheric solver.

Before selecting a named treatment, its O2 fraction, effective non-air
composition, O2-path fractionation and any sulfite-side kinetic effect must
describe compatible conditions. Peng alone does not map pO2 to reversibility,
and Wei alone does not specify the final non-air sulfate composition.
Preservation and pathway proportions remain archive-specific constraints.
The 2021 reproduction stays intact as a benchmark. The 2026 source terms
are not silently combined with its fitted coefficients into a new default.

## Reproduce

From the repository directory:

```text
python validation/audit_sulfate_fractionation.py
python -m pytest validation/test_sulfate_fractionation_audit.py validation/test_sulfate_to_air.py -o addopts= -q
```

Outputs are `outputs/sulfate_fractionation_audit.json`, the matching PNG,
and `outputs/sulfate_fractionation_audit_2026.png` for the source-term comparison.
JSON records every scenario, input convention, root status, residual, data ID,
and the SHA256 hashes of the surface and audit script. The diagnostic imports
the central model but changes no atmospheric parameter, UI default or posterior.
The full-source additions have 217 passing focused audit/transfer tests,
including native-coordinate Fig. 7 checks, log-fractionation conversions,
Table 1 transcription checks and preservation of the equation/theta discrepancy.
