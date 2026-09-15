# Scientific input attribution

Numerical inputs retain their original source terms. Model-generated response
surfaces and benchmark outputs are identified separately from observations.

| File or collection | Source and terms | Transformation |
|---|---|---|
| `validation/reference_data/adnew_2025_stratoclim_table_s3.csv` | Adnew, G. A., Koren, G., Mehendale, N., Gromov, S., Krol, M., and Roeckmann, T. (2025), [AMT 18, 2701-2719](https://doi.org/10.5194/amt-18-2701-2025), supplement Table S3, CC BY 4.0 | Selected numerical columns and air-mass classification; original measurements retain their reference scale. |
| `code/data/era5_l137_hybrid_coefficients.csv` | ECMWF, [L137 model level definitions](https://confluence.ecmwf.int/spaces/UDOC/pages/108117123/L137+model+level+definitions), [public-content terms](https://www.ecmwf.int/en/terms-use), CC BY 4.0 | The model-level coefficients are tabulated as CSV for pressure reconstruction. |
| `model_data/literature/banerjee_2026_supporting_tables.csv` | Banerjee et al. (2026), [USAP-DC 602070](https://doi.org/10.15784/602070), CC BY 4.0 | Workbook normalization preserves source exclusions and flags inconsistencies; see the adjacent README and importer. |
| `model_data/literature/brandon_2020_termination_v_delta17.csv` | Brandon et al. (2020), [PANGAEA 914283](https://doi.org/10.1594/PANGAEA.914283), CC BY 4.0 | Selected corrected measurements, with source isotope convention retained. |
| `model_data/literature/yang_2022_pangaea_941483.csv` | Yang et al. (2022), [PANGAEA 941483](https://doi.org/10.1594/PANGAEA.941483), CC BY 4.0 | Normalized column names; numerical observations unchanged. The source isotope convention is retained. |
| `model_data/literature/bereiter_2015_noaa_composite.txt` | Bereiter et al. (2015), [NOAA corrected Antarctic CO2 composite](https://doi.org/10.25921/n8y4-bp27) | Original public NOAA archive file, including its attribution, metadata and use constraints. |
| `model_data/literature/ishidoya_2025_annual_delta18.csv` | Ishidoya et al., [Zenodo 14221768](https://doi.org/10.5281/zenodo.14221768), CC BY 4.0 | Annual means from archived monthly values; raw-data weighting alternatives and hashes are recorded in the source manifest. |
| `model_data/validation_evidence/` | OXYTIB calculated benchmark reports, with the cited observational/model sources in each report | Canonical report export removes machine-local paths and preserves numerical results. These files are outputs, not raw measurements. |
| `model_data/literature/liu_2021_reference_grid.json` | OXYTIB external-model runs using the Liu et al. (2021) architecture; see `multimodel_benchmark_registry_v1.json` for source identifiers | Only the external model's outputs and coordinates are extracted from the hashed comparison report. OXYTIB predictions are recalculated separately. |
| `validation/reference_data/climate/` | OXYTIB numerical atmospheric experiments using native Photochem 0.6.7 and Clima | Native atmosphere arrays and portable manifests, with original artifact hashes; program source is obtained separately under its original license. OXYTIB-authored numerical outputs are CC BY 4.0. |

The Young curve coordinates are OXYTIB digitizations, with the source paper
identified in the validation tables. Figure scans and extracted article text
are excluded. Large external Photochem, Clima, and reanalysis inputs retain
their source attribution and distribution terms; the corresponding scientific
programs are obtained separately under their own licenses.

For individually imported ice-core sources, exact identifiers and hashes are
also recorded in `model_data/literature/ice_core_external_sources.json`.
OXYTIB authorship of a transformation does not replace attribution to the
original measurements or external numerical model.
