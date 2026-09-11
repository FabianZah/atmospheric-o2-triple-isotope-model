# High-pCO2 process scope

OXYTIB includes altitude-dependent O/O(1D)/O2/O3/CO2 photochemistry,
pO2- and pCO2-dependent chemical columns, a globally mixed oxygen inventory,
absolute GPP and source-backed biological alternatives. Its forcing
normalization uses molecular oxygen balance and the Adnew CO2 isoflux.

## Central assumptions

The native column uses prescribed temperature and vertical-mixing profiles.
The atmosphere responds chemically to pO2 and pCO2 while those structural
profiles remain specified. The biological budget is expressed in total
global GPP. The published domain is 0.1-2 PAL pO2, 50-60,000 ppm pCO2 and
18.256264-850 Pg C per year GPP.

## Structural evidence

- Paired Clima/chemistry calculations quantify thermal and ozone sensitivity.
- Fixed-total-dry-gas and additive-CO2 experiments isolate pressure convention.
- End members at 0.1 and 2 PAL test the oxygen-climate interaction.
- Ozone-heated RCE experiments test the modern climate-profile boundary.
- Marine-accessibility calculations test the mapping from marine production
  to atmospheric oxygen throughput.
- ERA5 latitude-height transport provides modern transport evidence.

The climate and marine-accessibility experiments are retained as sensitivity
evidence. The integrated assessment records their modern and cross-model
qualifications. Present-day ERA5 circulation is a modern constraint, rather
than a reconstructed paleo-circulation field.

## Interpretation and reproduction

The evidence separates source chemistry, thermal/transport structure and
biological boundary conditions. A multi-model residual alone cannot identify
which mechanism causes a discrepancy. Conservation, independent source
parameters and multiple validation families support the interpretation.

See [structural uncertainty](structural_uncertainty_policy.md),
[model validation](publication_model_acceptance.md) and the versioned
`model_data/validation_evidence/` records. The process inventory in
`model_data/literature/high_pco2_process_inventory_v1.json` is a provenance
record; the operational model is defined by the model contract.
