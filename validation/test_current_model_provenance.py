"""Source identity, historical separation and portable marine cache contracts."""
import json
from pathlib import Path

import pytest

import export_current_model_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]


def test_current_comparisons_are_bound_to_current_core():
    assert evidence.evidence_matches_current_core()
    manifest = json.loads((evidence.TARGET / "manifest.json").read_text())
    model_id = json.loads((ROOT / "model_data/updated_r7_response_surface_v1.json").read_text())["model_data_id"]
    assert manifest["model_data_id"] == model_id
    for name, row in manifest["files"].items():
        if name.startswith("clima_"):
            assert row["applicable_model_data_id"] is None
            assert row["evidence_scope"] == "archived_chemistry_structural_experiment"
        else:
            assert row["applicable_model_data_id"] == model_id
            assert row["evidence_scope"] == "current_core_comparison"


def test_core_change_invalidates_evidence_binding(monkeypatch):
    original = evidence.sha
    monkeypatch.setattr(evidence, "sha", lambda p:
                        "0"*64 if Path(p).name == "updated_molecular_forward_model.py" else original(p))
    assert not evidence.evidence_matches_current_core()


@pytest.mark.parametrize("name", ["audit_cao_bao_2013_benchmark.py", "young_fig8_digitized_curves.csv"])
def test_comparison_source_change_invalidates_binding(monkeypatch, name):
    original = evidence.sha
    monkeypatch.setattr(evidence, "sha", lambda p: "0"*64 if Path(p).name == name else original(p))
    assert not evidence.evidence_matches_current_core()


def test_imported_response_and_modern_cache_preserve_numerical_source():
    manifest = json.loads((ROOT / "model_data/model_input_provenance_v1.json").read_text())
    for row in manifest["original_imports"][:2]:
        assert evidence.sha(ROOT / row["path"]) == row["sha256_lf_normalized_at_import"]


def test_uncertainty_provenance_resolves_inside_portable_package():
    contract = json.loads((ROOT / "model_data/uncertainty/updated_o2_uncertainty_layers_v1.json").read_text())
    for source in contract["source_files"].values():
        if source.startswith("https://doi.org/"):
            continue
        path = (ROOT / source).resolve()
        assert path.is_relative_to(ROOT)
        assert path.is_file()


def test_marine_cache_source_hashes_are_portable_and_current():
    bundle = json.loads((ROOT / "model_data/marine_molecular_output_surface_v1.json").read_text())
    assert bundle["construction"]["source_hash_convention"] == "SHA-256 of LF-normalized text"
    for relative, expected in bundle["construction"]["source_hashes"].items():
        assert "\\" not in relative
        assert evidence.sha(ROOT / relative) == expected
    assert bundle["validation"]["passed"]
    assert bundle["validation"]["cap_delta17_maximum_absolute_residual_permil"] < .005
    assert bundle["validation"]["central_delta18_maximum_absolute_residual_permil"] < .015


def test_marine_scope_counts_and_fixed_equilibrium():
    registry = json.loads((ROOT / "model_data/biosphere_configurations_v1.json").read_text())
    marine = registry["configurations"]["marine_endmember"]
    assert "152" in marine["numerical_validation"]["scope"]
    assert "14" in marine["numerical_validation"]["scope"]
    assert marine["reference_sea_surface_temperature_c"] == 20.
    assert marine["surface_respiration_fraction"] + marine["interior_respiration_fraction"] == pytest.approx(1.)


def test_banerjee_inversion_search_uses_declared_model_domain():
    from audit_banerjee_2026_updated_model import GPP_BOUNDS_PGC_PER_YEAR
    bundle = json.loads((ROOT / "model_data/updated_r7_response_surface_v1.json").read_text())
    assert tuple(bundle["domain"]["gpp_pgC_per_year"]) == GPP_BOUNDS_PGC_PER_YEAR


@pytest.mark.parametrize("reverse", [False, True])
def test_liu_unavailable_curve_is_unassessed_not_a_direction_failure(reverse):
    from merge_liu_2021_gpp_grid import shape_metrics
    rows = []
    for gpp in (5., 50., 100.):
        for co2 in (300., 3000., 30000.):
            reference = -co2 / gpp
            value = float("nan") if gpp == 5. else reference * (-1 if reverse else 1)
            rows.append(dict(pO2_PAL=1., pCO2_ppm=co2,
                GPP_percent_of_own_reference=gpp,
                Liu_cap_delta17_prime_permil=reference,
                updated_cap_delta17_prime_permil=value,
                Liu_response_from_300ppm_100pct_permil=reference,
                updated_response_from_300ppm_100pct_permil=value))
    metrics = shape_metrics(rows)
    assert metrics["paired_finite_state_count"] == 6
    assert metrics["updated_unavailable_state_count"] == 3
    assert metrics["unassessed_pCO2_curve_count"] == 1
    assert metrics["all_pCO2_direction_checks_pass"] is (not reverse)
    assert metrics["all_GPP_direction_checks_pass"] is (not reverse)
