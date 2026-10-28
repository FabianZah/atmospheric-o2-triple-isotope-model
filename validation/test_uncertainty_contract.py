"""Prevent archived atmospheric experiments from relabelling current uncertainty."""
import json
from pathlib import Path

import pytest

from export_uncertainty_contract import run

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "model_data/validation_evidence"


def test_archived_exporter_rejects_corrected_core_without_overwriting(tmp_path):
    output = tmp_path / "protected.json"
    output.write_text("preserve this contract")
    with pytest.raises(ValueError, match="archived climate evidence"):
        run(ROOT / "model_data/literature/multimodel_evidence_matrix_v1.json",
            EVIDENCE / "clima_global_o2_response.json",
            ROOT / "model_data/updated_r7_response_surface_v1.json", output)
    assert output.read_text() == "preserve this contract"


def test_current_uncertainty_is_portable_and_keeps_layers_separate():
    contract = json.loads((ROOT / "model_data/uncertainty/updated_o2_uncertainty_layers_v1.json").read_text())
    response = json.loads((ROOT / "model_data/updated_r7_response_surface_v1.json").read_text())
    assert contract["central_model_data_id"] == response["model_data_id"]
    assert not contract["policy"]["combined_public_confidence_interval_available"]
    assert not contract["policy"]["inter_model_spread_used_as_Gaussian_sigma"]
    assert not contract["policy"]["numerical_guardrails_used_as_probability_distributions"]
    for path in contract["source_files"].values():
        if path.startswith("https://doi.org/"):
            continue
        assert path.startswith("model_data/") or path.startswith("docs/")
        assert (ROOT / path).is_file()
    assert ":/" not in json.dumps([p for p in contract["source_files"].values() if not p.startswith("https://")])
