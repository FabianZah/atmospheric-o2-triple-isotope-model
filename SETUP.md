# Setup and reproducibility

The accepted model runs locally from the repository. No external download,
account, compiled chemistry package, or copyrighted paper is required for the
normal forward model, inverse tools, isotope fields, transients, or public web
application.

## 1. Create an environment

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code/requirements-api.txt
```

For release validation and the historical response anchor, install:

```powershell
python -m pip install -r code/requirements-dev.txt
```

`code/requirements-tested.txt` records the exact direct dependency versions
used for the accepted local verification. It is useful for reproducing that
software stack; `code/requirements-api.txt` is the public server installation
file.
Production Docker, HTTPS, update, and rollback instructions are in
[`docs/server_deployment.md`](docs/server_deployment.md).

## 2. Run calculations

The console entry point is the primary local interface:

```powershell
python run_model.py calculate forward --po2 1 --pco2 294 --gpp 290
python run_model.py calculate infer --target-d17o -0.426 --solve-for pCO2 --po2 1 --gpp 290
python run_model.py calculate transient --experiment pCO2-trajectory --initial-pco2 285.5 --final-value 422.8 --trajectory-duration 174 --duration 12000
```

Use `python run_model.py calculate --help` and the command-specific help for
all inputs and export options.

## 3. Start the browser interface

```powershell
python run_model.py api --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the browser interface or
`http://127.0.0.1:8000/docs` for the generated API documentation.

The hosted OXYTIB application is the recommended graphical interface. Running
the browser application locally is useful for deployment checks and software
development.

## 4. Verify the installation

```powershell
python run_model.py validate
```

The smoke test is a fast end-to-end runtime check. The acceptance command
regenerates the integrated report and its local JSON, CSV, and PNG artifacts.
Its scientific inputs are the versioned reports in
`model_data/validation_evidence/`; ignored development outputs are not needed.

## 5. Run the publication regression tests

```powershell
python -m pytest `
  validation/test_publication_package_smoke.py `
  validation/test_publication_model_contract.py `
  validation/test_publication_model_acceptance.py `
  validation/test_updated_molecular_forward_model.py `
  validation/test_updated_forward_surface.py `
  validation/test_updated_output_surface.py `
  validation/test_updated_output_surface_release.py `
  validation/test_updated_output_surface_inverse.py `
  validation/test_updated_output_surface_posterior.py `
  validation/test_updated_output_surface_joint_posterior.py `
  validation/test_updated_uncertainty_layers.py `
  validation/test_uncertainty_contract.py `
  validation/test_updated_molecular_transient.py `
  validation/test_updated_photosynthesis_transient.py `
  validation/test_updated_pco2_trajectory_transient.py `
  validation/test_public_cli.py `
  validation/test_web_api.py `
  validation/test_public_deployment.py -q
```

The historical Young et al. (2014) response anchor can be regenerated
separately:

```powershell
python validation/audit_young_acceptance_gate.py
```

## Layout and import policy

The accepted snapshot retains a flat `code/` and `validation/` namespace to
avoid changing the audited scientific import graph. `conftest.py` configures
the paths for pytest, and each standalone scientific script contains its own
repository bootstrap. Normal users should use `run_model.py`.

Generated reports go to `outputs/`. Versioned runtime inputs are under
`model_data/`. Third-party literature, ERA5 files, and native Photochem or
Clima products remain outside the normal runtime and are not redistributed.

## Scientific provenance

The released numerical surfaces and compact validation evidence are versioned
under `model_data/`. Their architecture, acceptance criteria, and evidence
integrity policy are documented in:

- `docs/publication_model_definition.md`
- `docs/publication_model_acceptance.md`
- `docs/validation_evidence_bundle.md`

Do not commit supplied PDFs, manuscript files, credentials, or downloaded
third-party datasets. The repository policy is defined in `.gitignore` and
`LICENSING.md`.
