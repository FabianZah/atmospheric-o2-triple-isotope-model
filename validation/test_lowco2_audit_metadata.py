"""Low-CO2 validation supports native full-domain and historical expanded grids."""
import json
from types import SimpleNamespace

import pytest
import audit_updated_output_surface_lowco2 as audit


@pytest.mark.parametrize("expanded", [False, True])
def test_lowco2_audit_keeps_identity_and_accepts_complete_domain_metadata(monkeypatch, tmp_path, expanded):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    previous = dict(cap_delta17_maximum_absolute_residual_permil=.004,
                    central_delta18_maximum_absolute_residual_permil=.014,
                    delta18_acceleration_validated=True)
    bundle = dict(surface_data_id="test_current_core", validation=(
        dict(previous_domain_validation=previous) if expanded else previous))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(bundle))
    values = {key: 0. for key in audit.SURFACE_FIELDS}
    monkeypatch.setattr(audit, "UpdatedMolecularOutputSurface", lambda b: SimpleNamespace(pco2_nodes=[50.,100.]))
    monkeypatch.setattr(audit, "run_updated_forward", lambda *a, **k: None)
    monkeypatch.setattr(audit, "_live_fields", lambda r: values)
    monkeypatch.setattr(audit, "_interpolated_fields", lambda *a: values)
    monkeypatch.setattr(audit, "_plot", lambda *a: None)
    promoted = tmp_path / "validated.json"
    result = audit.run(candidate, tmp_path / "report.json", promoted)
    assert result["status"] == "accepted"
    saved = json.loads(promoted.read_text())
    assert saved["surface_data_id"] == "test_current_core"
    assert saved["validation"]["cap_delta17_maximum_absolute_residual_permil"] == .004
    assert saved["validation"]["central_delta18_maximum_absolute_residual_permil"] == .014
