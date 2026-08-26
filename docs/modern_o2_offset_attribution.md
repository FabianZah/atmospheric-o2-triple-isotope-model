# Modern atmospheric O2 Delta-prime-17O offset

## Result

The native fixed-parent Photochem column coupled to the unchanged Young et al.
(2014) biological budget predicts `-0.313973 per mil` at
1 PAL O2, 294 ppm CO2, and 100% Young GPP. Pack (2021) reports
`-0.432 +/- 0.015 per mil`.
The structural model-minus-observation residual is therefore
`+0.118027 per mil`, or
`7.87` times the stated analytical uncertainty.
No output offset or fitted rate is applied to the raw result.

## Exclusions

- **Numerical fixed point:** the direct native-column holdout residual is
  `1.68e-07 per mil`.
- **Fast-slow iteration:** the self-consistent correction relative to the
  frozen modern column is only `-0.000152 per mil`.
- **Response-surface interpolation:** crossed pO2-pCO2-GPP validation has a
  maximum Delta-prime-17O error of
  `0.001667 per mil`.
- **Transport dimensionality alone:** the closed annual ERA5 2-D ledger plus
  the tested GPP and biological conventions still leaves at least
  `0.079745 per mil`.
- **Matched native transport geometry:** using identical native chemistry and
  isotope boundaries, ERA5 2-D transport supplies only
  `4.97%`
  of the missing R7 forcing. Surface and local-tropopause boundary geometries
  differ by only `0.29%`.
- **O(1D) isotope source:** native R3 production differs from Young Table 3 by
  only `-0.069 per mil`, and
  the solved O(1D) pool differs by
  `+0.013 per mil`.
  The 30-50 km region supplies `67.1%`
  of native R7 encounters. Wrong O(1D) source composition is excluded.
- **CO2 anomaly source-export closure:** the unscaled R7 source and tropopause
  export close to `1.13e-11` relative.
  The exported isoflux is `2.471e+15
  per mil mol CO2 yr-1`, or `57.8%` of
  the directly comparable Adnew et al. (2025) lambda=0.528 estimate. Boering et al.
  (2004) and Koren et al. (2019) use different anomaly definitions and are retained
  only as contextual magnitudes. Numerical transport loss is excluded.
- **Global Dole closure alone:** closing delta-prime-18O and testing the joint
  literature envelope still leaves at least
  `0.076357 per mil`.
- **Explicit land-ocean biological closure:** the source-backed production,
  source-water, and respiration partition leaves at least
  `0.078862 per mil` after closing only
  atmospheric delta-prime-18O. The result uses `20`
  cases inside Bender et al.'s stated leaf-water limits and does not fit
  Delta-prime-17O.
- **294-to-400 ppm response:** the partitioned ensemble gives a 150-year shift
  of `-0.001388` to
  `-0.001373 per mil`; Young's soft
  Fig. 10 target (`-0.006 per mil`)
  lies outside this interval.
- **Self-consistent resolved R7 feedback:** coupling the ERA5/R7 tendency to the
  evolving global O2 isotope ratios changes the partitioned equilibrium by only
  `-0.000178` to
  `-0.000129 per mil`. Its coupled
  150-year shift remains `-0.001393`
  to `-0.001378 per mil`.
  Independent affine holdouts at 294 and 400 ppm have maximum relative R7
  tendency residuals of
  `2.08e-05` and
  `6.94e-05`.
- **Adnew-observed modern isoflux:** the directly comparable lambda=0.528
  observation is `1.638` times the
  resolved 400-ppm isoflux. Applying that observational forcing closes
  `17.1` to
  `27.0%` of the raw
  Pack gap across mean GPP/biology cases, but `0` of
  `8` cases overlap Pack at one sigma. The best mean
  case remains `+0.070140 per mil` too
  positive. Young and resolved R7 differ by only
   `0.066%`
   in global-O2 tendency per unit CO2 isoflux under Young's reduced convention.
   An empirical transfer multiplier is excluded, but the separately audited
   molecular material-balance limit gives
   `3` of
   `8` compact mean cases inside Pack uncertainty.
   With geometry-matched normalization, the explicit pathway gate at the mean
   Adnew isoflux overlaps Pack at 260 and 290 Pg C yr-1. Propagating Adnew's
   one-sigma uncertainty adds some Pack-compatible 320 Pg C yr-1 pathway cases.
- **Pathway-conserving land-ocean biology:** allocating Young et al.'s printed
  COX, AOX, and photorespiration fractions across land and ocean gives
  `0` Pack-overlapping cases at
  central literature inputs. Across the full published one-sigma GPP and R7
  envelope, `0` cases overlap,
  including `0` cases
  with the exact uniform global respiration exponent. The closest case uses the
  Liang et al. (2023) lower global GPP bound of
  `260.0 Pg C yr-1` and has a
  residual of `+0.062575 per mil`, but lies
  at an allocation-envelope edge and is retained only as a bound. Closing that
  case through R7 alone would require an isoflux
  `15.4` observational standard
  deviations above the Adnew et al. mean.
  Under the independently defined molecular material-balance limit, mean-isoflux
  Pack overlap occurs at
  `260, 290 Pg C yr-1`;
  the full Adnew one-sigma envelope contains solutions at
  `260, 290, 320 Pg C yr-1`.
  The architecture passes, but the parameter surface is non-unique.
- **Global-O2 GPP memory:** after an idealized step from each central GPP
  estimate to the Liang et al. lower global bound, `0`
  of `654` guardrail-valid cases have Pack-compatible
  asymptotes. Even after 5000 years the closest residual is
  `+0.043459 per mil`, and the closest
  asymptotic residual is `+0.042666
  per mil`. This excludes global-GPP uncertainty alone as the offset source.
- **External HOx/NOx/acid carriers:** complete irreversible export of the
  available HNO3 and H2O2 oxygen with an extreme
  `100 per mil` anomaly
  supplies only `1.96%`
  of the isoflux still missing after the Adnew R7 constraint. Internal carrier
  cycling is atom conserving and is not an independent global-O2 forcing.
- **Ozone-formation MIF:** Young's published `a_MIF = 1.109` sensitivity
  strengthens native R7 forcing by `1.738`
  relative to the `1.065` baseline. This is material for forward extrapolation,
  but cannot be selected from the modern O2 residual because Adnew et al.
  (2025) already observe the net modern CO2 isoflux.
- **Molecular-transfer transient:** the independently defined molecular bridge
  gives a 150-year shift range of
  `-0.003756` to
  `-0.003498 per mil` after
  propagating Adnew's one-sigma isoflux uncertainty. Every case has the correct
  negative sign and remains within Young's printed single-digit/per-meg bound.
  The minimum residual to the approximate -0.006 per mil visual anchor is
  `0.002244 per mil`;
  this is retained as a diagnostic rather than fitted.

## Current attribution

1. **The resolved R7 pathway remains low, but arbitrary kinetic repair is rejected.** Native
   O(1D) production and composition pass the Young Table 3 check, and the
   generated CO2 anomaly is exported conservatively. Young's homogeneous bulk box has
   `2.99` times the native
   encounters, while vertical CO2 isotope covariance accounts for its smaller
   per-event difference. The unscaled resolved export remains below the modern
   Adnew lambda=0.528 constraint. Liang product branching is coordinate invariant
   after site degeneracy and lowers the source; Young's printed R7a lies outside
   the Sander total-rate uncertainty. Neither convention is used to tune the
   resolved model. A historical
   diagnostic would require an R7 flux factor of
   `4.75`, but
   this factor is not a model parameter and must not be applied.
2. **The literal Young one-site transfer does not close the updated-model offset.**
   No central or one-sigma pathway allocation reaches Pack uncertainty under
   the literal reduced mapping. The
   earlier apparent low-GPP closure resulted from treating Adnew's terrestrial
   leaf-assimilation estimate as total terrestrial-plus-marine GPP; that invalid
   substitution is removed. Liang's lower global bound still leaves at least
   `+0.062575 per mil`.
3. **Observed R7 magnitude is necessary but not sufficient under the literal mapping.**
   Adnew's directly comparable modern isoflux materially strengthens the forcing,
   yet no tested central-GPP case reaches Pack uncertainty. Young's Fig. 10
   amplitude is independently reconstructed from the printed bulk-R7 ledger, so
   a generic transient multiplier or damping term is not indicated.
4. **The molecular CO2-to-O2 site-coordinate convention passes the updated-model
   architecture gates.** Young's one-site pseudo-species ledger is retained for Young
   reproduction. The equal-and-opposite molecular-isoflux limit follows Liang
   et al. (2023), Eq. 3, and overlaps the Pack interval without a fitted rate at
   the central modern GPP/isoflux case. Its propagated transient remains within
   Young's printed single-digit/per-meg bound. GPP and pathway allocation remain
   a solution surface rather than a uniquely identified parameter set.
5. **Long atmospheric memory does not rescue the global-GPP range.** Exact
   propagation to the Liang lower bound remains outside Pack after 5000 years
   and asymptotically for every tested pathway allocation. A historical hindcast
   remains useful for transient prediction, but is no longer the leading
   explanation of the steady modern offset.
5. **Resolved external carriers are too small to repair the offset.** Even an
   intentionally permissive complete-export bound leaves more than 98% of the
   missing forcing. These states remain valuable for local chemistry and
   conservation, but adding their positive branches to global O2 would double
   count internal oxygen cycling.
6. **Ozone source-law uncertainty belongs in prediction intervals.** The
   published Young bracket has a large effect on calculated R7 forcing, while
   the temperature dependence of the individual 17O ozone-formation KIEs is
   still unmeasured. It is therefore propagated as structural uncertainty and
   is not calibrated against Pack.

## Policy

The Pack value may anchor observation-referenced differentials for reporting,
but the raw baseline, Pack value, structural residual, and model differential
must all be exported. The Pack residual must not alter an ODE, reaction rate,
fractionation exponent, or response derivative unless an independent physical
constraint justifies that change.
