# OXYTIB

**Atmospheric oxygen triple-isotope budget and inference model**

OXYTIB links atmospheric O₂ triple-isotope composition to pCO₂, pO₂, and
gross primary production (GPP). It provides steady forward calculations,
constrained inference of one coordinate, isotope fields, time-response
experiments, and traceable scientific exports.

The hosted application is available at
[mycompton.de/oxytib](https://mycompton.de/oxytib/).

## Scientific model

The deterministic model couples altitude-resolved oxygen, ozone and
carbon-dioxide photochemistry to a conservative global atmospheric-O₂ and
biological-turnover budget. A precomputed model grid makes repeated evaluations
faster by interpolation, with accuracy checked against direct model calculations.

OXYTIB predicts atmospheric O₂ Δ′¹⁷O and δ′¹⁸O for specified pCO₂, pO₂, and
GPP. Inference solves one coordinate from an isotope observation and
independent constraints on the other two. The accepted operational domain is:

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| pO₂ | 0.10 PAL | 2.00 PAL |
| pCO₂ | 50 ppm | 60,000 ppm |
| GPP | 18.256 Pg C yr⁻¹ | 850 Pg C yr⁻¹ |

PAL means present atmospheric level; 1 PAL corresponds to the model's modern
O₂ reference of 21.2%. In the interface, 100% modern GPP is 290 Pg C yr⁻¹.

The machine-readable definition in
[`model_data/publication_model_contract_v1.json`](model_data/publication_model_contract_v1.json)
pins the model identity, numerical data, domain, modern reference state,
uncertainty policy, and scientific evidence.

## Use OXYTIB

The release is tested with Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code/requirements-api.txt -c code/requirements-api-lock.txt
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv`
and activate it with `source .venv/bin/activate`. The remaining commands are
the same.

Run a steady forward calculation:

```powershell
python run_model.py calculate forward --po2 1 --pco2 294 --gpp 290
```

In the web solver, **O₂ isotope composition** predicts Δ′¹⁷O and conventional
δ¹⁸O (VSMOW). Fixed, 1σ, or range constraints on pCO₂, GPP, and pO₂ can be
propagated to isotope intervals. The [forward-constraint guide](docs/forward_isotope_constraints.md)
describes the calculation and direct Python/API usage.

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
python -m pip install -r code/requirements-dev.txt -c code/requirements-api-lock.txt
python run_model.py validate
```

This checks runtime data integrity, a forward-to-inverse smoke calculation,
API and browser assets, and the scientific acceptance decision using the
versioned evidence reports. Run the numerical regression suite separately:

```powershell
python -m pytest validation -q
```

Regenerate the bundled-input comparisons and their plots:

```powershell
python run_model.py reproduce
```

Individual comparisons can be selected, for example `python run_model.py
reproduce cao-bao yang`. Logs, input hashes, and completion status are saved
in `outputs/reproduction/manifest.json`. The
[reproduction guide](docs/reproduction.md) distinguishes these calculations
from external climate/chemistry runs and evidence aggregation.

These installation commands use the production pins in
`code/requirements-api-lock.txt`, matching the CI constraints.
`code/requirements-tested.txt` also records the direct package versions from
the isolated local verification environment.

## Evidence

Validation combines modern atmospheric O₂ observations, ice-core behavior,
published productivity constraints, independent interpolation tests, conservation checks,
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

The release contains the code, dependencies, model data and validation
evidence needed to reproduce the published calculations. Large third-party
datasets, copyrighted literature, exploratory downloads, and generated local
outputs remain outside the release runtime.

## Citation and license

Zahnow, F. (2026). *OXYTIB: Atmospheric oxygen triple-isotope budget and inference
model* (v0.1.0) [Software]. Zenodo.
[doi:10.5281/zenodo.22819964](https://doi.org/10.5281/zenodo.22819964).

This DOI identifies the archived v0.1.0 release. Downloadable citation records
are supplied as [`CITATION.cff`](CITATION.cff), [`CITATION.bib`](CITATION.bib),
and [`CITATION.ris`](CITATION.ris).

Source code is MIT licensed. Project documentation and original model data are
CC BY 4.0. Third-party scientific inputs retain their original licenses; see
[`LICENSING.md`](LICENSING.md).
