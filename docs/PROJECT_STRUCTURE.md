# Project structure

## Release layout

The first publication release intentionally retains the audited flat module
layout:

```text
run_model.py                 stable public launcher
code/                        model, inverse tools, API, and runtime dependencies
web/                         independent browser interface
model_data/                  versioned runtime surfaces and contracts
validation/                  release tests and acceptance generators
docs/                        scientific and software documentation
outputs/                     generated local artifacts
.github/workflows/           release and container CI
```

This is a conservative reproducibility decision. Moving the scientific modules
into a conventional installable package would alter a large import graph just
before publication without changing the model. `run_model.py` provides stable
calculation, API, and validation commands for normal use.

## Stable public components

The public model identity is defined by
`model_data/publication_model_contract_v1.json`. Its central implementation is:

- `code/updated_molecular_forward_model.py`
- `code/updated_output_surface.py`
- `code/updated_output_surface_inverse.py`
- `code/updated_output_surface_posterior.py`
- `code/updated_output_surface_joint_posterior.py`
- `code/updated_molecular_transient.py`
- `code/updated_photosynthesis_transient.py`
- `code/updated_pco2_trajectory_transient.py`
- `code/updated_uncertainty_layers.py`
- `code/public_model_service.py`
- `code/public_cli.py`
- `code/web_api.py`
- `web/index.html`, `web/styles.css`, and `web/app.js`

The contract includes SHA-256 identities for the central source and runtime
data files. `validation/audit_publication_model_acceptance.py` is the
integrated release decision.

## Publication boundary

The repository retains the 57-module operational dependency closure and the
additional validation modules required by the release tests and
published-model response anchors. Compact validation records and digitized
reference datasets are included where they are direct inputs to the release
contract. Exploratory download scripts, private source documents, and generated
local outputs remain outside the publication repository.

A later software-only release may move modules into an installable package.
That refactor should occur after the scientific release tag and must preserve
contract hashes through an explicitly reviewed version change and full
regression comparison.
