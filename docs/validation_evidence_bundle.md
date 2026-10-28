# Validation evidence bundle

The integrated publication-model acceptance decision reads compact,
machine-readable evidence from `model_data/validation_evidence/corrected_core_20261019/`. These files
are versioned release inputs. They allow a clean checkout and GitHub CI to
reproduce the acceptance verdict without private literature files, downloaded
external datasets, or a developer's ignored `outputs/` directory.

## Contents

`manifest.json` records each report's:

- repository path;
- LF-normalized SHA-256 digest;
- generating validation script;
- scientific role;
- corresponding development report path.

The manifest also binds all model code and numerical bundles by hash. A changed
core invalidates that binding. The parent directory retains the archived
v0.1.0 evidence and its original manifest.

The bundle includes the central release scorecard, Liu and Cao-Bao model
comparisons, the Luz inverse-architecture comparison, Yang/Banerjee and Brandon
observational checks, the marine-accessibility sensitivity, and the separated
uncertainty-layer audit and low-CO2 predictive assessment. Clima records are
explicitly marked as archived-chemistry experiments, with no current-model
applicability. They supply historical context, not numerical uncertainty for
the corrected core.

## Regeneration policy

Development audits continue to write detailed products under `outputs/`.
After intentionally regenerating and reviewing those audits, refresh the
canonical bundle with:

```powershell
python validation/export_current_model_evidence.py
python validation/export_publication_model_contract.py
python run_model.py acceptance
```

The current exporter requires regenerated comparison reports. It removes
machine-local artifact paths and fails if it encounters
an unexpected absolute path. Refreshing the bundle changes contract hashes and
therefore requires scientific review; it is not an automatic application
startup step.

`export_uncertainty_contract.py` belongs to the archived climate-evidence
workflow and rejects the corrected atmospheric core. Retain the supplied
current uncertainty contract unless a new scientific update is independently
validated.

## Integrity check

```powershell
python -m pytest validation/test_validation_evidence_bundle.py validation/test_current_model_provenance.py -q
```

The tests verify archived and current evidence identities, portable paths,
source-data integrity and current-core binding.
