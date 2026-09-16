# Repository structure

```text
run_model.py                 console and API entry point
code/                        scientific model, inference, exports and API
web/                         browser interface and bundled equation renderer
model_data/                  versioned numerical inputs and evidence
validation/                  regression tests and scientific audits
docs/                        methods, model use and deployment
deploy/                      Docker and reverse-proxy configuration
outputs/                     generated local reports (untracked)
.github/workflows/           scientific, software and container CI
```

## Scientific implementation

The model identity, implementation paths and data checksums are defined in
[the model contract](../model_data/publication_model_contract_v1.json).

- `public_model_service.py` connects the console, API and model.
- `updated_molecular_forward_model.py` defines the direct physical-model calculation.
- `updated_output_surface.py` provides faster evaluation by interpolation in a precomputed model grid.
- The inverse, posterior and transient modules implement the corresponding
  calculations using the same physical model.
- `spherule_to_air_d17o.py`, `sulfate_to_air.py` and the sulfate uncertainty
  modules implement proxy transfer.
- `model_result_workbook.py` builds scientific exports.
- `web_api.py`, `public_cli.py` and `web/` expose the model.

Python modules use a flat namespace. `run_model.py` resolves the repository
paths; `conftest.py` supplies the paths for regression tests.

## Reproducibility

Runtime data, observation/reference inputs, validation evidence and their
supporting code are version controlled. The historical Young response
calculation and structural sensitivity records provide comparison evidence
for the model. Tests distinguish these comparisons from the operational
forward and inverse calculations.

Generated reports belong in `outputs/`. Copyrighted papers, personal
manuscripts, credentials and exploratory downloads remain outside the
public repository. See [licensing](../LICENSING.md) for third-party data terms.
