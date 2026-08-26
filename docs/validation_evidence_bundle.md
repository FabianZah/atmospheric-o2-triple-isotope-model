# Validation evidence bundle

The integrated publication-model acceptance decision reads compact,
machine-readable evidence from `model_data/validation_evidence/`. These files
are versioned release inputs. They allow a clean checkout and GitHub CI to
reproduce the acceptance verdict without private literature files, downloaded
external datasets, or a developer's ignored `outputs/` directory.

## Contents

`manifest.json` records each report's:

- repository path;
- SHA-256 digest and byte count;
- generating validation script;
- scientific role;
- corresponding development report path.

The bundle includes the central release scorecard, Liu and Cao-Bao model
comparisons, the Luz inverse-architecture comparison, Yang/Banerjee and Brandon
observational checks, the marine-accessibility sensitivity, and the separated
uncertainty-layer audit. It also carries the reviewed Clima pressure/pO2
structural end members and the low-CO2 predictive-error candidate required to
rebuild the uncertainty contract from a clean checkout.

## Regeneration policy

Development audits continue to write detailed products under `outputs/`.
After intentionally regenerating and reviewing those audits, refresh the
canonical bundle with:

```powershell
python validation/export_validation_evidence_bundle.py
python validation/export_uncertainty_contract.py
python validation/export_publication_model_contract.py
python run_model.py acceptance
```

The exporter removes machine-local artifact paths and fails if it encounters
an unexpected absolute path. Refreshing the bundle changes contract hashes and
therefore requires scientific review; it is not an automatic application
startup step.

## Integrity check

```powershell
python -m pytest validation/test_validation_evidence_bundle.py -q
```

The test verifies every digest and size, rejects absolute local paths, and
requires every evidence-matrix source report to resolve inside the bundle.
