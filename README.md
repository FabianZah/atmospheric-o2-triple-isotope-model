# OXYTIB

**Atmospheric oxygen triple-isotope budget and inference model**

OXYTIB links atmospheric O₂ triple-isotope composition to pCO₂, pO₂, and
gross primary production (GPP). It provides steady forward calculations,
constrained inference of one coordinate, isotope fields, time-response
experiments, and traceable scientific exports.

The hosted application is available at
[mycompton.de/oxytib](https://mycompton.de/oxytib/).

## Scientific model

The deterministic model couples altitude-resolved R1-R7 photochemistry to a
conservative global atmospheric-O₂ and biological-turnover budget. A versioned
output surface accelerates repeated evaluation while preserving the validated
central calculation.

OXYTIB predicts atmospheric O₂ Δ′¹⁷O and δ′¹⁸O for specified pCO₂, pO₂, and
GPP. Inference solves one coordinate from an isotope observation and
independent constraints on the other two. The accepted operational domain is:

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| pO₂ | 0.10 PAL | 2.00 PAL |
| pCO₂ | 50 ppm | 60,000 ppm |
| GPP | 18.256 Pg C yr⁻¹ | 850 Pg C yr⁻¹ |

The machine-readable definition in
[`model_data/publication_model_contract_v1.json`](model_data/publication_model_contract_v1.json)
pins the model identity, numerical data, domain, modern reference state,
uncertainty policy, and scientific evidence.

## Use OXYTIB

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code/requirements-api.txt
```

Run a steady forward calculation:

```powershell
python run_model.py calculate forward --po2 1 --pco2 294 --gpp 290
```

Infer pCO₂ at fixed pO₂ and GPP:

```powershell
python run_model.py calculate infer `
  --target-d17o -0.426 --sigma 0.004 `
  --solve-for pCO2 --po2 1 --gpp 290
```

Export a pCO₂ step response:

```powershell
python run_model.py calculate transient `
  --experiment pCO2 --initial-pco2 280 --final-value 420 `
  --duration 12000 --format xlsx --output oxytib_pco2_step.xlsx
```

Run a gradual pCO₂ trajectory using the historical-reference endpoints:

```powershell
python run_model.py calculate transient `
  --experiment pCO2-trajectory --initial-pco2 285.5 --final-value 422.8 `
  --trajectory-duration 174 --duration 12000 `
  --format xlsx --output oxytib_pco2_trajectory.xlsx
```

The endpoint preset uses the IPCC AR6 estimate for 1850 and the NOAA Global
Monitoring Laboratory global mean for 2024. Both endpoints, duration, and the
linear or smooth interpolation are editable.

Every calculation uses the same service functions and versioned model data as
the hosted application. JSON is the default console format; time responses can
also be exported as CSV or XLSX. Command-specific options are listed with, for
example, `python run_model.py calculate infer --help`.

## Validate the release

Install the release-validation dependencies and run the integrated checks:

```powershell
python -m pip install -r code/requirements-dev.txt
python run_model.py validate
```

This verifies runtime data integrity, steady and transient calculations,
forward-to-inverse closure, API and browser assets, and the integrated
scientific acceptance decision. Exact direct dependency versions from the
accepted environment are recorded in `code/requirements-tested.txt`.

The release-candidate protocol for independent domain review is provided in
[`docs/expert_review_guide.md`](docs/expert_review_guide.md). Review findings
can be recorded with the accompanying
[`feedback template`](docs/expert_review_feedback_template.md).

## Evidence

Validation combines modern atmospheric O₂ observations, ice-core behavior,
published productivity constraints, numerical holdouts, conservation checks,
and comparisons with published atmospheric-isotope models. Young et al. (2014)
provides the foundational atmospheric O₂ budget architecture and published
response curves; later literature constrains photochemistry, biological
fractionation, proxy conversion, modern isotope composition, and structural
sensitivity.

The compact evidence bundle is stored under
`model_data/validation_evidence/`. The integrated assessment is summarized in
[`docs/publication_model_acceptance.md`](docs/publication_model_acceptance.md),
and the scientific model is defined in
[`docs/publication_model_definition.md`](docs/publication_model_definition.md).

## Repository guide

- `run_model.py`: stable console entry point.
- `code/`: model, inference, transient, export, and API implementation.
- `web/`: hosted browser interface.
- `model_data/`: versioned runtime surfaces, contracts, and evidence records.
- `validation/`: regression tests and reproducible scientific audits.
- `docs/`: model definition, methods, uncertainty, validation, and deployment.
- `deploy/`: Docker and reverse-proxy configuration for an independent server.

The release contains the operational dependency closure and validation
evidence needed to reproduce the published calculations. Large third-party
datasets, copyrighted literature, exploratory downloads, and generated local
outputs remain outside the release runtime.

## Citation and license

Citation records are supplied as [`CITATION.cff`](CITATION.cff),
[`CITATION.bib`](CITATION.bib), and [`CITATION.ris`](CITATION.ris). The DOI will
be added to these records when the v0.1.0 archive is deposited.

Source code is MIT licensed. Project documentation and original model data are
CC BY 4.0. Third-party scientific inputs retain their original licenses; see
[`LICENSING.md`](LICENSING.md).
