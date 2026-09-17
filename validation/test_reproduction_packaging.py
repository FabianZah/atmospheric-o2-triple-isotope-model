"""Release reproducibility, source separation and dependency contract tests."""

import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import reproduce_benchmarks as reproduction
from modern_isotope_reference import modern_reference_isotope_compositions

ROOT = reproduction.ROOT


def test_reference_extraction_preserves_compositions_and_excludes_unused_port():
    oxygen, carbon = modern_reference_isotope_compositions()
    assert oxygen.delta18_prime_permil == 23.600
    assert oxygen.cap_delta17_prime_permil == -0.432
    assert carbon.cap_delta17_prime_permil == -0.2186969696969697
    assert carbon.delta18_prime_permil == pytest.approx(1000 * math.log1p(41.78933333333333 / 1000), abs=1e-12)
    assert not (ROOT / "code/photochem_two_stream.py").exists()
    assert not (ROOT / "code/modern_isotope_column.py").exists()
    assert not (ROOT / "code/modern_photolysis.py").exists()
    assert not (ROOT / "code/observation_referenced_isotope.py").exists()


def test_every_evidence_generator_is_supplied_and_importable():
    manifest = json.loads((ROOT / "model_data/validation_evidence/manifest.json").read_text())
    for name in {row["generator"] for row in manifest["files"].values()}:
        assert (ROOT / name).is_file(), name
        importlib.import_module(Path(name).stem)
    for scripts in reproduction.BENCHMARKS.values():
        for name in scripts:
            importlib.import_module(Path(name).stem)


def test_native_climate_inputs_match_archived_artifact_hashes():
    inventory = json.loads((ROOT / "validation/reference_data/climate/inventory.json").read_text())
    assert len(inventory["files"]) == 20
    for row in inventory["files"]:
        data = (ROOT / row["path"]).read_bytes()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]


def test_liu_reference_grid_excludes_oxytib_predictions():
    from merge_liu_2021_gpp_grid import DEFAULT_INPUT, load_columns
    columns = load_columns(DEFAULT_INPUT)
    assert len(columns) == 15
    assert sum(len(row["scenarios"]) for row in columns) == 90
    for column in columns:
        for row in column["scenarios"]:
            assert not any("updated" in key or "oxytib" in key for key in row)


def test_bundled_ice_core_inputs_match_archived_hashes():
    expected = {
        "yang_2022_pangaea_941483.csv": "82933b451e9b77ec897ad1137618ab56bbe23185b7d01b2a57c4ae6a29f06649",
        "bereiter_2015_noaa_composite.txt": "57e6a3855b818c359fe566f921204734c97f912827b192430d92fd32a517c56d",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "model_data/literature" / name).read_bytes()).hexdigest() == digest


def test_reproduction_records_success_and_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(reproduction, "ROOT", tmp_path)
    monkeypatch.setattr(reproduction, "input_inventory", lambda: [])
    monkeypatch.setattr(reproduction.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert reproduction.run(["fig8"])["status"] == "completed"
    monkeypatch.setattr(reproduction.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match="failed"):
        reproduction.run(["fig8"])
    saved = json.loads((tmp_path / "outputs/reproduction/manifest.json").read_text())
    assert saved["status"] == "incomplete"
    assert saved["runs"][0]["returncode"] == 2


@pytest.mark.parametrize("timeout", [0, -1])
def test_reproduction_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="must be positive"):
        reproduction.run(["fig8"], timeout_seconds=timeout)


def test_acceptance_default_leaves_tracked_documentation_untouched():
    from audit_publication_model_acceptance import DEFAULT_DOC
    assert DEFAULT_DOC.parent == ROOT / "outputs"


def test_api_documentation_assets_have_exact_versions():
    content = (ROOT / "web/api-docs.html").read_text()
    assert "swagger-ui-dist@5/" not in content
    assert content.count("swagger-ui-dist@5.32.15/") == 2


@pytest.mark.parametrize("status", ["failed", "grid_refinement_required", None])
def test_numerical_reproduction_rejects_unaccepted_reports(status):
    from reproduce_numerical_audits import require_accepted
    with pytest.raises(RuntimeError, match="scientific acceptance gates"):
        require_accepted({"status": status}, "Test audit")
    require_accepted({"status": "accepted"}, "Test audit")


@pytest.mark.parametrize("passed", [True, False])
def test_scorecard_cli_exit_reflects_formal_gates(monkeypatch, passed):
    import audit_updated_molecular_release_scorecard as scorecard
    monkeypatch.setattr("sys.argv", ["scorecard"])
    monkeypatch.setattr(scorecard, "run", lambda **kwargs: {
        "formal_gate_summary": {"passed": int(passed), "failed": int(not passed),
                                "all_passed": passed}})
    if passed:
        scorecard.main()
    else:
        with pytest.raises(SystemExit) as error:
            scorecard.main()
        assert error.value.code == 1
