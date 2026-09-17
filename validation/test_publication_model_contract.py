from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "validation") not in sys.path:
    sys.path.insert(0, str(ROOT / "validation"))

from export_publication_model_contract import run  # noqa: E402
from export_publication_model_contract import _sha256  # noqa: E402


def test_stored_contract_hashes_match_supplied_sources():
    contract = json.loads((ROOT / "model_data/publication_model_contract_v1.json").read_text())
    assert contract['source_hash_convention'] == 'SHA-256; Python and JSON text normalized from CRLF to LF'
    for record in contract['source_files'].values():
        assert _sha256(ROOT / record['path']) == record['sha256'], record['path']


def test_source_hashes_are_portable_but_detect_content_changes(tmp_path):
    source = tmp_path / 'example.py'
    source.write_bytes(b'x = 1\ny = 2\n')
    expected = _sha256(source)
    source.write_bytes(b'x = 1\r\ny = 2\r\n')
    assert _sha256(source) == expected
    source.write_bytes(b'x = 1\ny = 3\n')
    assert _sha256(source) != expected


def test_publication_contract_has_one_consistent_model() -> None:
    output = ROOT / "outputs" / f"_pytest_publication_contract_{uuid4().hex}.json"
    try:
        contract = run(output_path=output)

        assert contract["schema_version"] == 1
        assert contract["status"] == "active_publication_model"
        policy = contract["single_model_policy"]
        assert policy["release_model_count"] == 1
        assert policy["public_interface_model_count"] == 1
        assert policy["historical_reconstructions_role"] == "validation evidence"
        assert policy["structural_endmembers_role"] == (
            "structural sensitivity evidence"
        )

        deterministic = contract["deterministic_model"]
        assert deterministic["model_data_id"] == "updated_r7_response_surface_v1"
        accelerator = deterministic["numerical_accelerator"]
        assert accelerator["surface_data_id"] == (
            "updated_molecular_output_surface_v1"
        )
        assert accelerator["extrapolation_permitted"] is False
        assert accelerator["output_offset_applied"] is False
        assert accelerator["smoothing_applied"] is False
        assert deterministic["transient_solvers"] == {
            "atmospheric_state_step": "code/updated_molecular_transient.py",
            "photosynthesis_step": "code/updated_photosynthesis_transient.py",
            "gradual_pCO2_trajectory": "code/updated_pco2_trajectory_transient.py",
        }

        assert contract["operational_domain"]["pCO2_ppm"]["minimum"] == 50.0
        assert contract["operational_domain"]["pCO2_ppm"]["maximum"] == 60000.0
        assert contract["modern_reference_state"]["raw_model_reporting"] == (
            "unadjusted mechanistic state"
        )
        assert contract["reporting_policy"]["observation_reference_offset_applied"] is False
        assert "observation_referenced_definition" not in contract["reporting_policy"]
        from public_model_service import result_envelope
        assert set(contract["reporting_policy"]["required_api_envelope_fields"]) == set(
            result_envelope({}, calculation="contract_test"))

        uncertainty = contract["uncertainty"]
        assert uncertainty["layers_remain_separate"]
        assert not uncertainty["combined_public_confidence_interval_available"]
        assert not uncertainty["whole_domain_structural_Gaussian_sigma_available"]
        assert "structural sensitivity" in uncertainty["clima_role"]

        assert len(contract["validation"]["evidence"]) == 7
        assert contract["validation"]["evidence_bundle_id"] == (
            "publication_validation_evidence_v1"
        )
        assert contract["validation"]["integrated_acceptance_generator"] == (
            "validation/audit_publication_model_acceptance.py"
        )
        assert contract["validation"]["integrated_acceptance_report"] == (
            "docs/publication_model_acceptance.md"
        )
        assert all(
            len(item["sha256"]) == 64
            for item in contract["source_files"].values()
        )
        assert {
            "state_step_transient_implementation",
            "photosynthesis_step_transient_implementation",
            "pco2_trajectory_transient_implementation",
            "forward_constraints_implementation",
            "public_service_implementation",
        } <= set(contract["source_files"])
        assert all(
            item["path"].startswith("model_data/validation_evidence/")
            for name, item in contract["source_files"].items()
            if name.startswith("validation_")
            and name not in {
                "validation_evidence_matrix",
                "validation_evidence_exporter",
            }
        )
        assert json.loads(output.read_text(encoding="utf-8")) == contract
    finally:
        output.unlink(missing_ok=True)
