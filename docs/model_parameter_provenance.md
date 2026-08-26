# Model Parameter Provenance

This table separates strict Young printed values, the current Young-like reproduction branch, and updated-model candidate values. Updated parameters should be chosen from literature or explicit scenario assumptions, not by minimizing deviation from Young.

## Branch Policy

| branch | purpose | criterion |
|---|---|---|
| strict Young printed | audit the paper-derived equations and constants | use Young printed/text values wherever known |
| Young-like reproduction | reproduce published Young figure behavior where printed information is incomplete | may use explicitly flagged effective values, but must not be treated as pure Young input |
| updated model | apply newer literature constraints and scenario inputs | may diverge from Young if the divergence follows from documented updated values |
| sensitivity tests | quantify parameter influence around either branch | vary one parameter or a documented parameter set with provenance |

## Parameter Ledger

| parameter | process | strict Young value | Young-like reproduction value | updated candidate values | status | source basis | interpretation |
|---|---|---:|---:|---|---|---|---|
| `a_mif` | stratospheric O3/O(1D)/CO2 MIF source strength | 1.065 | 1.065 | global-a_MIF Pack candidate: 1.0725484294; biosphere candidates: 1.065 | global-a_MIF route rejected as preferred updated mechanism | Young et al., 2014 for 1.065; Pack 2021 target used only to test whether a_MIF should be retuned | Changing this value globally matches Pack but over-propagates into high-pCO2 Fig. 8 behavior. Keep Young value unless newer photochemical literature directly justifies a different mechanism. |
| `beta_respiration_17` | biospheric respiration 17O/18O fractionation exponent | 0.5149 | 0.515 | respiration-beta Pack candidate: 0.514; water-beta candidate keeps 0.515 | updated sensitivity candidate, not final | Young derives global beta 0.5149; Young also reports AOX beta 0.514, alternative Blunier beta 0.5143, and uncertainty of at least about 0.0005 | A value near 0.5140 can match Pack mechanistically, but shifts Fig. 9 more negative. Use as a respiration-pathway sensitivity unless literature supports it as the updated mean. |
| `evapotranspiration_beta_17` | source-water / evapotranspiration 17O/18O exponent | 0.52 | 0.524 | water-beta Pack candidate: 0.521; respiration-beta candidate keeps 0.524 | public updated_physical default candidate | Young et al., 2014 derive global bwater=0.520, report direct leaf-transpiration bwater about 0.519, and discuss measured transpiration beta values from about 0.522 to 0.514 depending humidity | evapotranspiration_beta_17=0.521 is a literature-bounded source-water/transpiration sensitivity inside the Young-discussed range. It also matches the Pack modern-air target about as well as beta_respiration_17=0.514 while leaving respiration beta unchanged. Treat it as an updated-model default candidate to test, not as a number selected by Young-fit preservation. |
| `alpha_respiration_18` | biospheric respiration 18O fractionation | 0.982125319191 | 0.982800982801 | not selected as current Pack mechanism | branch-separating parameter | Young Table 2/Section 3.4 gives alpha_r = 1/1.0182; reproduction branch uses effective value | This parameter must be kept visibly separated because the current reproduction branch uses an effective value rather than the strict printed Young value. |
| `evapotranspiration_alpha_18` | source-water / evapotranspiration 18O enrichment | 1.00525 | 1.0058 | not selected as current Pack mechanism | branch-separating parameter | Young footnote/text gives 1.00525; reproduction branch uses effective value | This is another place where the current Young-like reproduction branch is effective rather than strictly printed. It should be reported as such in validation materials. |
| `explicit_lower_box_net_export_rate_per_year` | lower-stratospheric CO2 export/isoflux handling | not printed as explicit lower-box ODE | none in core Young-like O2 branch | 0.959699184825 | updated CO2-isoflux candidate | Adnew et al., 2025 modern CO2 Delta'17O isoflux target; not a Young printed equation | This is an updated-model extension for modern CO2 export diagnostics. It should not be used to claim exact Young ODE reproduction. |
| `o2_d17o_calibration_mode` | reported-output O2 Delta'17O calibration | none | none | updated_physical_from_validated_young_local_pack_anchor: pack2021_validated_young_local_offset; mechanistic candidates: none | diagnostic only | Pack 2021 modern O2 Delta'17O target -0.432 per mil | Useful to diagnose reference alignment, but not a final scientific mechanism unless the Pack-Young difference is explicitly interpreted as a reporting-scale offset. |
| `p_o2_pal` | atmospheric O2 level | 1 PAL in modern Young Fig. 7/Fig. 8 calculations | scenario input; default 1 PAL | scenario/user input; Phanerozoic working range currently 0.1-2.0 PAL | sensitivity/scenario variable | Young modern baseline; Mills et al., 2023 for Phanerozoic pO2 context | This is not a fitted Pack parameter. It is a major user-controlled paleo boundary condition. |
| `gpp_scale` | global primary production / photosynthetic O2 throughput | Young 100% scale, Table 3 convention | scenario input; internally mapped to Young scale | scenario/user input with selectable normalization | sensitivity/scenario variable | Young et al., 2014; Beerling 1999 and Liang et al., 2023 for context/normalization options | Updated GPP changes should be sensitivity/scenario choices, not hidden branch constants. |

## Immediate Consequence

The current `young_reproduction_po2_feedback_candidate` is a Young-like effective reproduction branch, not a strict printed-input branch. In particular, the biosphere/source-water isotope parameters differ from the printed Young inventory. This is acceptable for a validation candidate only if the manuscript and code make the distinction explicit.

For the updated model, the current leading mechanistic candidate is source-water/evapotranspiration beta around `0.521`, but this should be accepted only after direct literature re-reading and not merely because it preserves Young-like behavior.
