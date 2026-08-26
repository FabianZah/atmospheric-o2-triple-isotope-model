# High-pCO2 Model Disagreement

## Question

The updated molecular model becomes more negative than digitized Young et al.
(2014) Fig. 8 above roughly 18,000 ppm, especially at 50% GPP. This audit asks
whether that discrepancy demonstrates an error in the updated model, an
imprecision in Young, or unresolved structural uncertainty in both.

It does not tune either model. The executable audit is
`validation/audit_high_pco2_competing_hypotheses.py`.

## Updated-model evidence

The updated R7 surface includes direct converged Photochem parent atmospheres
at 300, 1,000, 3,000, 10,000, 30,000, and 60,000 ppm at 1 PAL O2. The
30,000-ppm result is therefore not an extrapolation or a spline-generated
endpoint. Parent isotope-kernel scaled residuals remain below `3.1e-15`, and
the crossed response-surface holdout error is below `0.0017 per mil` in
Delta-prime-17O.

At a fixed modern isotope state, the local log elasticity of the R7 forcing
declines smoothly from `0.995` near 294 ppm to `0.917` at 30,000 ppm and
`0.868` at 60,000 ppm. This is the expected direction for increasing
O(1D)-CO2 competition and contains no hook or discontinuity.

A two-parameter competition-saturation curve describes the updated Fig. 8
responses very closely:

| GPP | effective half-saturation | fit RMSE |
|---|---:|---:|
| 100% | 48,069 ppm | 0.0020 per mil |
| 50% | 25,844 ppm | 0.0042 per mil |

These fits are diagnostics of curve shape, not forward-model equations.

## GPP reference separation

An equal percentage does not prescribe equal absolute production in the two
models. Young's Table 3 reference is 365.125 PgC/yr, whereas the updated
model's central literature reference is 290 PgC/yr. The audit now reports two
comparisons:

- own-reference: 100% or 50% of each model's selected modern reference;
- common-input: Young's absolute GPP prescribed in the updated architecture.

At 30,000 ppm, the results are:

| GPP label | own-reference updated minus Young | common-input updated minus Young |
|---|---:|---:|
| 100% | -0.614 per mil | +0.749 per mil |
| 50% | -1.585 per mil | +0.010 per mil |

The near-zero common-input 50% endpoint shows that most of its apparent
30,000-ppm offset is caused by the different GPP reference. It does not remove
the shape disagreement: the common-input 50% residual reaches approximately
+1 per mil at intermediate pCO2, and the 100% residual remains positive over
the high-pCO2 limb. Percentage-normalized curves therefore cannot by themselves
be interpreted as an architecture-only validation.

## Factorial attribution

The common-input comparison was factored into four cases using the same updated
Photochem R7 response surface: native or Adnew-normalized R7 forcing, each
combined with either Young aggregate biology or updated partitioned biology.

Across both Fig. 8 GPP levels and the complete plotted pCO2 range, changing only
the biological convention alters Delta-prime-17O by at most 0.00109 per mil.
The biological update is therefore not the source of the high-pCO2 shape
difference when absolute GPP is matched.

Without the modern molecular-isoflux normalization, the updated native R7
forcing produces residuals of approximately +5 to +7 per mil at 30,000 ppm.
Applying the source-backed Adnew normalization reduces those residuals to
+0.750 per mil for 100% GPP and +0.010 per mil for 50% GPP. The remaining
curvature is therefore localized to the molecular forcing shape or to the
assumption that the modern global transfer normalization remains constant with
pCO2. This result does not justify fitting a pCO2-dependent transfer factor.

## Young-model evidence

The equivalent fits to digitized Young Fig. 8 are:

| GPP | effective half-saturation | fit RMSE |
|---|---:|---:|
| 100% | 45,202 ppm | 0.0650 per mil |
| 50% | 21,001 ppm | 0.1844 per mil |

The 100% Young and updated saturation scales are similar. Most disagreement is
therefore not a wholesale failure of the updated O(1D)-CO2 saturation law. It
is concentrated in the stronger flattening of Young's 50% GPP high-pCO2 limb.

Young et al. provide a raster curve and short textual examples but no
high-pCO2 numerical table, uncertainty interval, or source code. Their model
also uses one representative stratospheric box with fixed photolysis,
temperature/rate conventions, and first-order stratosphere-troposphere
transport. An unreported coupling or numerical/plotting imprecision in this
extreme limb is therefore plausible.

## Updated-model limitations

The updated result is more resolved but is not independently verified at
30,000 ppm. Parent chemistry is recalculated for each pO2-pCO2 boundary, while
the Photochem ModernEarth temperature and eddy-diffusion profiles remain
unchanged. The ERA5/Adnew transfer normalization is constrained near modern
pCO2 and is then held constant across the response surface. There are no
observational high-pCO2 atmospheric isotope data against which either model
can be validated.

Consequently, numerical robustness does not establish high-pCO2 accuracy.

The surface-pressure convention was also tested with four new Photochem parent
atmospheres in which CO2 displaced N2 at fixed dry major-gas pressure. The
fixed-pressure parent grids all converged, but they strengthened the negative
R7 forcing by only 0.7%, 2.0%, and 3.7% at 10,000, 30,000, and 60,000 ppm.
After propagation through the global O2 reservoir, fixed pressure shifted
Delta-prime-17O by another -0.10 to -0.14 per mil at 30,000 ppm. It therefore
does not explain the Young disagreement and moves the own-reference comparison
slightly farther away.

## Resolved transport-shape test

The native 1-D R7 chemistry was mapped into the same annual ERA5 TEM+Kyy
transport ledger at 294, 300, 400, 10,000, and 60,000 ppm. Both geometries are
reported in the same conserved CO2 tangent-isoflux coordinate. The ERA5 source
closes against lower-boundary export to better than `1.4e-11` relative at every
node, and its effective anomaly residence time remains between `3.751` and
`3.768 yr`.

The ERA5/native source-isoflux ratio changes by at most `6.78%` relative to its
400-ppm value. To test only this shape, the ratio was normalized to unity at
400 ppm so that the independently observed Adnew modern normalization remained
unchanged. No point was fitted to Young Fig. 8. Applying the resolved transport
shape increases the combined common-input Fig. 8 RMSE from `0.689` to `0.986
per mil`. At 30,000 ppm, the 50% GPP residual changes from `+0.011` to `+0.461
per mil` and the 100% residual from `+0.750` to `+1.117 per mil`.

Steady anomaly loss and a strong pCO2-dependent transport efficiency are
therefore rejected as explanations for the Young high-pCO2 limb within this
architecture. The transport calculation is retained as a validation result,
not promoted to a correction in the production surface. Machine-readable
results are written to `outputs/era5_pco2_transfer_shape.*`.

## O(1D)-R7 curvature decomposition

The six direct 1-PAL native parent nodes from 300 to 60,000 ppm were decomposed
without fitting a response curve. Integrated O(1D) production changes by only
`+0.279%`, while the fraction of O(1D) loss proceeding through R7 rises from
`0.104%` to `14.605%`. The isotope forcing per R7 encounter strengthens by
`5.02%`; it does not collapse at high pCO2.

The total forcing elasticity closes exactly as the sum of encounter-throughput
and per-encounter-transfer elasticities. Between 30,000 and 60,000 ppm, those
terms are `0.858 + 0.031 = 0.889`. The native high-pCO2 flattening is therefore
attributed to physical saturation of O(1D) capture by CO2 as R7 competes with
other O(1D) sinks. It is not produced by numerical damping, declining O(1D)
production, weakening isotope transfer per encounter, or transport loss.
Machine-readable results are written to `outputs/o1d_pco2_decomposition.*`.

## Young R5/R7 source ambiguity

A fixed-Table-3-background algebraic audit separates the R5/R7 conventions
before they enter the coupled reconstruction. At 60,000 ppm, the fraction of
O(1D) captured by R7 is `1.87%` for Young's printed kR5 with whole `[M]`,
`1.90%` for the reversed footnote-d interpretation, and `30.08%` for the
literal footnote. The native Photochem column gives `14.61%`.

The diagnostically Table-3-balanced effective `[M]` gives `5.80%` with Young's
printed R7 rate. Combining that diagnostic reservoir with the independent Yung
et al. (1991) R7 rate gives `13.17%`, close to the native mechanism. This
explains why the reduced Young-like branch performs well, but it does not show
that Young used those unreported conventions. The balanced `[M]` value is not
derivable from Young's printed footnote and must remain labelled diagnostic.

Consequently, the reduced branch is a constrained Young-like reconstruction,
not an exact Young-only reproduction. The literal printed alternatives remain
validation bounds; the updated production model continues to use the native
Photochem/Sander mechanism. Results are written to
`outputs/young_r5_r7_capture_algebra.*`.

## Decision

- Do not apply a factor or damping term to force agreement with Young Fig. 8.
- Retain the native updated response as the current model result.
- Treat the high-pCO2 difference as structural model disagreement, not as a
  known error in either model.
- Keep own-reference and common-absolute-GPP comparisons separate; neither is
  a substitute for the other.
- Retain total pressure or pN2 as a structural-uncertainty axis, not a fitted
  high-pCO2 correction.
- Do not apply the resolved ERA5 transport shape as a fitted correction; it is
  small, independently constrained, and worsens agreement with Young Fig. 8.
- Do not tune the biological budget to repair the high-pCO2 limb. Test the
  physical pCO2 dependence of molecular anomaly production and temperature
  structure instead.
- Before claiming accuracy above 10,000 ppm, quantify sensitivity to
  temperature structure, transport/export normalization, and atmospheric
  structure or climate coupling.
- In the manuscript, use Young as a historical validation anchor over its
  better-constrained range and discuss the extreme Fig. 8 tail separately.

Machine-readable results and the comparison figure are written to
`outputs/high_pco2_competing_hypotheses.*`.
