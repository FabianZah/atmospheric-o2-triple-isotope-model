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


def test_publication_contract_has_one_consistent_model() -> None:
    output = ROOT / "outputs" / f"_pytest_publication_contract_{uuid4().hex}.json"
    try:
        contract = run(output_path=output)

        assert contract["schema_version"] == 1
        assert contract["status"] == "active_publication_model"
        assert contract["single_model_policy"]["manuscript_model_count"] == 1
        assert contract["single_model_policy"]["public_ui_model_count"] == 1
        assert not contract["single_model_policy"][
            "structural_endmembers_are_alternative_public_models"
        ]
        assert not contract["single_model_policy"][
            "diagnostic_output_offsets_or_damping_are_permitted"
        ]

        deterministic = contract["deterministic_model"]
        assert deterministic["model_data_id"] == "updated_r7_response_surface_v1"
        accelerator = deterministic["numerical_accelerator"]
        assert accelerator["surface_data_id"] == (
            "updated_molecular_output_surface_v1"
        )
        assert accelerator["extrapolation_permitted"] is False
        assert accelerator["output_offset_applied"] is False
        assert accelerator["smoothing_applied"] is False

        assert contract["operational_domain"]["pCO2_ppm"]["minimum"] == 50.0
        assert contract["operational_domain"]["pCO2_ppm"]["maximum"] == 60000.0
        assert contract["modern_reference_state"][
            "raw_model_is_modified_to_match_observation"
        ] is False
        assert contract["reporting_policy"][
            "observation_referenced_output_is_same_model"
        ]

        uncertainty = contract["uncertainty"]
        assert uncertainty["layers_remain_separate"]
        assert not uncertainty["combined_public_confidence_interval_available"]
        assert not uncertainty["whole_domain_structural_Gaussian_sigma_available"]
        assert "not a central-model correction" in uncertainty["clima_role"]

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
