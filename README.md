# Atmospheric O2 triple-isotope model

This repository contains a literature-updated model of the atmospheric O2
triple-oxygen-isotope budget, its inverse tools, an independent HTTP API, and
the validation evidence used to define the publication model. The public user
interface is a framework-independent browser application served by FastAPI.

The model calculates atmospheric O2 Delta-prime-17O and delta-prime-18O for
specified pCO2, pO2, and gross primary production (GPP). It can solve for one
of those physical coordinates when the other two are constrained. It does not
claim that one isotope measurement uniquely determines pCO2, pO2, and GPP
simultaneously.

The hosted interface is available at
[mycompton.de/atmo-mod](https://mycompton.de/atmo-mod/).

## Publication model

The manuscript and public interface use one deterministic model: an
altitude-resolved Photochem R1-R7 atmospheric column coupled to a conservative
global atmospheric-O2 and biological-turnover budget. The versioned output
surface accelerates repeated evaluation without applying smoothing, output
offsets, or extrapolation.

The accepted operational domain is:

- pO2: 0.1-2.0 PAL
- pCO2: 50-60,000 ppm
- absolute GPP: 18.256-850 PgC yr-1

The exact model identity, runtime data, domain, modern reference state, and
claim limits are pinned in
[`model_data/publication_model_contract_v1.json`](model_data/publication_model_contract_v1.json).
The compact machine-readable evidence used by the acceptance decision is
versioned under `model_data/validation_evidence/` with a SHA-256 manifest.
The integrated scientific decision is recorded in
[`docs/publication_model_acceptance.md`](docs/publication_model_acceptance.md).

## Quick start

Python 3.11 or newer is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code/requirements-api.txt
python run_model.py api --host 127.0.0.1 --port 8000
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.
The API command is otherwise the same. Open `http://127.0.0.1:8000/` for the
independent browser interface or `http://127.0.0.1:8000/docs` for generated API
documentation.
Constrained solutions can be exported as a compact CSV summary or as an XLSX
workbook containing provenance, inputs, the marginal posterior, and any joint
probability field. Time-response experiments can be exported as an XLSX
workbook containing the experiment inputs, complete time series, equilibrium
metadata, solver settings, and model provenance. The intended proxy sequence
and uncertainty separation are documented in
[`docs/spherule_inversion_workflow.md`](docs/spherule_inversion_workflow.md).

For a Linux server, use the hardened Compose and Caddy bundle under `deploy/`;
see [`docs/server_deployment.md`](docs/server_deployment.md).

## Verify the snapshot

The fast smoke test checks runtime data, two forward evaluations, a
forward-to-inverse round trip, the API and browser assets, and the integrated
acceptance verdict:

```powershell
python run_model.py smoke
```

Run the complete integrated acceptance report with:

```powershell
python run_model.py acceptance
```

Developer and CI dependencies are installed with:

```powershell
python -m pip install -r code/requirements-dev.txt
python -m pytest validation/test_publication_package_smoke.py `
  validation/test_publication_model_contract.py `
  validation/test_publication_model_acceptance.py -q
```

Exact direct dependency versions from the accepted local environment are
listed in `code/requirements-tested.txt`. See [`SETUP.md`](SETUP.md) for the
full reproducibility sequence.

## Evidence and scope

Young et al. (2014) remains an important historical response anchor. Published
model comparisons, observational tests, Clima experiments, and marine-access
experiments are validation evidence or structural-sensitivity end members;
they are not alternative public models and do not silently alter the central
calculation. Measurement, parameter, numerical, and structural uncertainty
remain separate.

Historical reconstruction scripts are retained for scientific provenance.
They are not normal user entry points. Optional external datasets and compiled
Photochem experiments are documented under `docs/` and are not required to run
the accepted model or its interface.

## Repository layout

- `code/`: model, inverse tools, HTTP API, and scientific plotting tools.
- `web/`: independent browser interface served by the HTTP API.
- `model_data/`: versioned runtime surfaces, contracts, and compact metadata.
- `validation/`: tests, audits, published-model comparisons, and provenance.
- `docs/`: methods, validation evidence, limitations, and developer notes.
- `outputs/`: generated local reports and figures; normally not versioned.
- `.github/workflows/`: release regression and container CI.

The flat `code/` layout is intentionally retained for this release to preserve
the audited import graph. The repository includes that import closure and the
release-validation suite, while excluding exploratory workflows, discarded
branches, private literature, and generated development outputs.
`run_model.py` is the stable public launcher.

## Citation and license

Citation metadata are supplied as [`CITATION.cff`](CITATION.cff),
[`CITATION.bib`](CITATION.bib), and [`CITATION.ris`](CITATION.ris). Source code
is MIT licensed; project documentation and reconstructed data are CC BY 4.0.
Third-party scientific inputs retain their original licenses and are handled
as described in [`LICENSING.md`](LICENSING.md).
Release changes are listed in [`CHANGELOG.md`](CHANGELOG.md).
