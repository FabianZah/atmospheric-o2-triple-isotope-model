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

from export_uncertainty_contract import run  # noqa: E402


EVIDENCE = ROOT / "model_data" / "validation_evidence"


def test_exported_uncertainty_contract_is_portable() -> None:
    output = ROOT / "outputs" / f"_pytest_uncertainty_{uuid4().hex}.json"
    try:
        contract = run(
            ROOT / "model_data" / "literature" / "multimodel_evidence_matrix_v1.json",
            EVIDENCE / "clima_global_o2_response.json",
            ROOT / "model_data" / "updated_r7_response_surface_v1.json",
            output,
            EVIDENCE / "yang_lowco2_predictive_error.json",
            EVIDENCE / "clima_global_o2_response_fixed_total.json",
            EVIDENCE / "clima_pressure_convention_audit.json",
            EVIDENCE / "clima_global_o2_response_0p1pal.json",
            EVIDENCE / "clima_global_o2_response_2pal.json",
            EVIDENCE / "clima_po2_cross_audit.json",
        )

        assert contract["schema_version"] == 1
        assert not contract["policy"]["combined_public_confidence_interval_available"]
        assert not contract["policy"]["inter_model_spread_used_as_Gaussian_sigma"]
        assert contract["source_files"]["evidence_matrix"].startswith("model_data/")
        assert all(
            path.startswith("model_data/")
            for path in contract["source_files"].values()
        )
        assert ":/" not in json.dumps(contract["source_files"])
        climate = contract["layers"]["structural"]["climate_endmember"]
        assert climate["central_model_changed"] is False
        assert climate["interpolation_permitted"] is False
        alternatives = contract["layers"]["structural"][
            "climate_pressure_alternatives"
        ]
        assert len(alternatives) == 1
        assert alternatives[0]["pressure_convention"] == "fixed_total_dry_major"
        robustness = contract["layers"]["structural"][
            "climate_pressure_robustness"
        ]
        assert robustness["status"] == (
            "pressure_robust_structural_endmember_not_promoted"
        )
        assert robustness["maximum_absolute_D17O_difference_full_domain"] < 0.25
        po2_endmembers = contract["layers"]["structural"][
            "climate_po2_endmembers"
        ]
        assert {row["domain"]["pO2_PAL"][0] for row in po2_endmembers} == {
            0.1,
            2.0,
        }
        assert contract["layers"]["structural"]["climate_po2_interaction"][
            "status"
        ] == "po2_interaction_material_structural_endmembers_not_promoted"
        low_co2 = next(
            row
            for row in contract["domain_specific_discrepancy_candidates"]
            if row["id"] == "yang_banerjee_low_co2_tracking"
        )
        assert 0.0 < low_co2["candidate_excess_predictive_sigma_permil"] < 0.05
        assert not low_co2["promoted_to_public_default"]
    finally:
        output.unlink(missing_ok=True)
