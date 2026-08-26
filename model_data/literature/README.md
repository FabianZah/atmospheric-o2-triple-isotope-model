# Literature validation data

`multimodel_benchmark_registry_v1.json` records the provenance, independence,
available quantitative material, and ingestion status of each external model or
observational benchmark. Registry entries are validation targets, not fitted
calibration data.

## Ishidoya et al. (2025)

`ishidoya_2025_annual_delta18.csv` contains annual atmospheric O2 delta18
changes derived from the authors' Zenodo archive (DOI
`10.5281/zenodo.14221768`, CC BY 4.0). The primary series is the equal-weight
annual mean of their prepared Figure 7 monthly values. Two independently
recomputed raw-data annualizations are retained as a weighting sensitivity.
`ishidoya_2025_source_manifest.json` records exact source-file checksums and
the known duplicated 2018 fractional-year row without modifying the archive.

Regenerate both files with:

```powershell
python validation/import_ishidoya_2025_zenodo.py --source-dir $HOME/Downloads
```

## Banerjee et al. (2026)

`banerjee_2026_supporting_tables.csv` is normalized from the authors' archived
USAP Data Center workbook, doi:10.15784/602070 (CC BY 4.0), associated with
Banerjee et al. (2026), doi:10.1029/2025JD045900. The archive MD5 was verified
as `2c0e4d4090f58cb790e098647141b219`; import is reproducible with
`validation/import_banerjee_2026_usap_archive.py`.

The source reports O2 Delta-17O in ppm relative to modern air with
lambda=0.518. It is not an absolute VSMOW Delta-prime-17O value. Use
`code/isotope_reference_frames.py` for comparison with model output.

The remaining source-table anomaly is retained in `source_note`:

- Table S1, depth 143.03 m: replicate 1 is 32.15 ppm but the printed average is
  29.42 ppm despite no printed replicate 2.
The archive supplies the complete reconstructed value at 144.415 m and also
preserves every author-supplied exclusion from the pristine CO2 dataset.

Equation 7 in the main paper prints a positive regression coefficient, but the
reported negative correlation, Figure 1, and Table S1 require
`CO2 = 266.58 - 1.6131 * Delta-17O(ppm)`. Table S2's reconstructed values instead
follow the exact archive-encoded relation
`CO2 = 256.9758 - 1.0862 * Delta-17O(ppm)`; its methodological attribution is
not stated in the workbook.
The new-data reconstructed values match the rounded printed relation within
0.001 ppm; the legacy relation matches its stored values to machine precision.
The minus-sign relation is the only operational implementation. The printed
plus-sign form is retained solely as source-provenance evidence in the audit.
Model validation must use the raw
measured isotope and CO2 columns, not either reconstructed-CO2 column.

The physical and accelerated updated-model domains now extend to 50 ppm. The
paired raw observations are applied by
`validation/audit_banerjee_2026_updated_model.py`. This infers GPP rather than
fitting model parameters; Banerjee's pointwise GPP estimates are not present in
the archived workbook and are therefore not used as numerical targets.
