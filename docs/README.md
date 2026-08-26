# Atmospheric O2 Triple-Isotope Model Documentation

This documentation describes the released model, its public inversion
interface, validation evidence, uncertainty layers, and deployment. The
authoritative machine-readable model identity is
`model_data/publication_model_contract_v1.json`.

The model predicts atmospheric O2 Delta-prime-17O and delta-prime-18O from
pCO2, pO2, and gross primary production (GPP). In inversion mode it solves one
of those physical coordinates while requiring independent constraints on the
other two.

## Start here

- `publication_model_definition.md`: equations, architecture, and fixed public
  model definition.
- `publication_model_acceptance.md`: integrated evidence table, accepted
  claims, and scope limits.
- `spherule_inversion_workflow.md`: I-type cosmic-spherule conversion,
  uncertainty propagation, and constrained inversion.
- `server_deployment.md`: local and independent Linux-server deployment.
- `GPP_NORMALIZATION_POLICY.md`: relationship between absolute and relative
  GPP reporting.
- `validation_evidence_bundle.md`: provenance and integrity policy for the
  compact evidence distributed with the release.

## Public model

The accepted model couples a resolved Photochem R1-R7 atmospheric column to a
conservative global atmospheric-O2 and biological-turnover budget. Its
versioned numerical surface accelerates repeated calculations without output
offsets, smoothing, or extrapolation.

The operational domain is:

| coordinate | minimum | maximum |
|---|---:|---:|
| pO2 | 0.10 PAL | 2.00 PAL |
| pCO2 | 50 ppm | 60,000 ppm |
| GPP | 18.256 PgC/yr | 850 PgC/yr |

Inputs outside this domain are rejected. The public model does not claim that
one isotope measurement uniquely determines pCO2, pO2, and GPP together.

## Run the interface

From the repository root:

```powershell
python -m pip install -r code/requirements-api.txt
python run_model.py api --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the browser application or
`http://127.0.0.1:8000/docs` for the generated HTTP API schema. The browser and
API call the same framework-independent service functions.

Stable public components are:

- `code/updated_molecular_forward_model.py`: central mechanistic calculation.
- `code/updated_output_surface.py`: validated numerical accelerator.
- `code/updated_output_surface_inverse.py`: deterministic one-coordinate
  inversion.
- `code/updated_constrained_pco2_posterior.py`: constrained coordinate
  probability calculations.
- `code/updated_molecular_transient.py` and
  `code/updated_photosynthesis_transient.py`: declared step experiments.
- `code/public_model_service.py` and `code/web_api.py`: public service boundary.
- `web/`: independent HTML, CSS, and JavaScript interface.

## Verify the release

Run the package smoke test and integrated scientific decision:

```powershell
python run_model.py smoke
python run_model.py acceptance
```

Install `code/requirements-dev.txt` to run the complete pytest suite. Continuous
integration checks the publication contract, forward and inverse calculations,
constrained posteriors, realistic spherule inversion, isotope-field contours,
transients, uncertainty separation, and the historical Young et al. (2014)
response anchor.

The compact evidence bundle is versioned under
`model_data/validation_evidence/`. Native Photochem, ERA5, Clima, and
published-model calculations used during model development are represented by
the curated evidence records; their large third-party inputs are not required
to run or verify the released model.

## Uncertainty policy

Measurement/proxy, parameter, numerical, and structural uncertainties remain
separate. Published-model differences and climate end members are validation
or structural-sensitivity evidence; they are not alternative public models and
are not silently added as a Gaussian error term.

## Source organization

The repository contains the operational model and the dependency closure
needed by its regression and historical-anchor tests. Exploratory download
pipelines, discarded model branches, manuscript planning notes, private
literature files, and generated development outputs are intentionally excluded.
