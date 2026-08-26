# GPP Normalization Policy

Last refreshed: 2026-06-15.

The model uses a Young-style photosynthetic O2 flux internally. User-facing
GPP inputs are entered as percent of a selected modern reference and are then
converted to the internal Young scale before the isotope model is solved.

## Decision

The public default remains:

```text
young_2014
```

This default is interpreted as the Young et al. (2014) gross O2-production
scale. It is retained because:

- it is the internal flux scale used to reproduce Young et al. behavior,
- it keeps validation and application runs traceable to the same model
  convention,
- it is nearly identical to the Beerling (1999) total gross O2-production
  value after stoichiometric conversion.

Liang et al. (2023), Beerling (1999), and custom global values remain available
as explicit normalization choices. They are not hidden model constants; they
are scenario/reference choices and are written into exports. Adnew et al.
(2025) estimate terrestrial leaf-assimilation GPP and are retained as a
terrestrial constraint, not a total global O2-production normalization.

## Modern Reference Values

| key | label | modern reference | ratio to Young scale | role |
|---|---|---:|---:|---|
| `young_2014` | Young et al., 2014 gross O2-production scale | 365.125 PgC/yr | 1.000 | default internal/public scale |
| `beerling_1999` | Beerling, 1999 Table 2 total | 367.527 PgC/yr | 1.0066 | near-equivalent gross O2-production cross-check |
| `liang_2023` | Liang et al., 2023 | 290 PgC/yr | 0.7942 | modern-reference sensitivity |
| `custom` | Custom | user-defined | user-defined | explicit user scenario |

## Export Rule

Every exported scenario should include:

- `gpp_user_percent_modern`,
- `gpp_normalization`,
- `gpp_normalization_role`,
- `gpp_normalization_label`,
- `gpp_modern_reference_pgC_per_year`,
- `gpp_requested_pgC_per_year`,
- `gpp_internal_young_scale`.

This lets a user report both the intuitive input, such as "25% modern GPP",
and the exact absolute reference used by the model.

## Manuscript Language

Recommended wording:

> Unless otherwise stated, GPP percentages are reported relative to the Young
> et al. (2014) gross O2-production scale used by the model. Alternative
> modern GPP estimates are treated as explicit sensitivity normalizations and
> are recorded with each model run.
