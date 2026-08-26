# Licensing

This project is dual-licensed, the standard arrangement for reproducible
research software with an associated dataset.

| Component | License | Files |
|---|---|---|
| Source code | **MIT** (see `LICENSE`) | `code/**.py`, `validation/**.py`, `conftest.py` |
| Data, documentation, figures | **CC-BY-4.0** (see below) | `docs/**`, generated `outputs/**` data/figures, digitized curve CSVs |

SPDX: `MIT AND CC-BY-4.0`.

## Code — MIT

Permissive: reuse, modify, and redistribute with attribution and the license
notice. See `LICENSE`.

## Data, documentation, and figures — CC-BY-4.0

The reconstructed datasets (e.g. digitized Young Fig. 7/Fig. 8 curve CSVs in
`outputs/`), the documentation in `docs/`, and the generated figures are
licensed under the Creative Commons Attribution 4.0 International License
(CC-BY-4.0): you may share and adapt them for any purpose with attribution.

- Human-readable summary: https://creativecommons.org/licenses/by/4.0/
- Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode

If a publisher or archive (e.g. Zenodo) requires the verbatim legal text in the
repository, drop it into `LICENSE-CC-BY-4.0.txt`.

## Third-party material — NOT covered by the above licenses

The following are **not** ours to relicense and must be handled separately:

- **`outputs/young_digitization_sources/*.png`** — cropped scans of figures from
  Young et al. (2014), *Geochimica et Cosmochimica Acta* 135, 102-125
  (Elsevier, copyrighted). These are used only by the optional raster-overlay
  scripts. They are **excluded from the public release** (see `.gitignore`); do
  not redistribute them. The published pipeline reads the cached digitized CSV
  (`outputs/young_fig8_digitized_curves.csv`), which is our derived data, not the
  scans.
- **`reference_texts/*.txt`** — extracted text of cited third-party papers,
  retained locally for the reconstruction. Not redistributed (gitignored).
- **Photochem example inputs** — Photochem is GPL-3.0 licensed. Its
  ModernEarth atmosphere file is downloaded separately and verified by
  checksum; it is not redistributed under this project's MIT/CC-BY licenses.
- **ERA5 TEM derived data** - the optional Serva (2022) monthly
  transformed-Eulerian-mean archive contains modified Copernicus Climate Change
  Service information. It is downloaded from Zenodo, checksum-verified, and
  retained only under ignored `external_data/`; it is not redistributed under
  this project's MIT/CC-BY licenses.

When citing or reusing the reconstructed data, please also cite Young et al.
(2014) as the original model source, and any datasets digitized from other
papers as noted in `docs/`.

## How to attribute

See `CITATION.cff` for the citation metadata.
