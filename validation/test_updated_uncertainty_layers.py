from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_output_surface import UpdatedOutputSurfaceInput  # noqa: E402
from updated_uncertainty_layers import decompose_updated_uncertainty  # noqa: E402


def test_uncertainty_layers_remain_separate_at_modern_conditions() -> None:
    result = decompose_updated_uncertainty(
        UpdatedOutputSurfaceInput(1.0, 294.0, 290.0),
        measurement_sigma_permil=0.015,
        measurement_source="Pack (2021)",
    )

    assert result.measurement.probabilistic
    assert not result.source_isoflux_parameter.probabilistic
    assert not result.biological_parameter.probabilistic
    assert not result.numerical.probabilistic
    assert not result.whole_domain_structural_sigma_available
    assert not result.combined_public_confidence_interval_available
    assert result.numerical.total_lower_margin_permil >= 0.0
    assert result.numerical.total_upper_margin_permil >= 0.0
    assert result.structural_endmember_points == ()


@pytest.mark.parametrize("po2", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("biosphere", ["modern_reference", "marine_endmember"])
def test_old_clima_offsets_are_excluded_from_corrected_model(po2, biosphere) -> None:
    result = decompose_updated_uncertainty(
        UpdatedOutputSurfaceInput(po2, 30000.0, 290.0, biosphere),
    )
    assert result.structural_endmember_points == ()
    assert not result.whole_domain_structural_sigma_available


def test_positive_measurement_sigma_requires_source() -> None:
    with pytest.raises(ValueError, match="traceable source"):
        decompose_updated_uncertainty(
            UpdatedOutputSurfaceInput(), measurement_sigma_permil=0.01
        )


def test_fixed_marine_parameter_scope_is_explicit() -> None:
    result = decompose_updated_uncertainty(
        UpdatedOutputSurfaceInput(1., 294., 290., "marine_endmember"),
    )
    assert "unassessed" in result.biological_parameter.interpretation
    assert "fixed marine" in result.crossed_parameter_envelope.interpretation
    assert result.biological_parameter.lower_distance_permil == 0.
    assert result.biological_parameter.upper_distance_permil == 0.
    assert result.source_isoflux_parameter.lower_distance_permil > 0.


def test_uncertainty_contract_cannot_be_mixed_with_another_core(tmp_path) -> None:
    from updated_uncertainty_layers import DEFAULT_UNCERTAINTY_CONTRACT_PATH
    contract = json.loads(DEFAULT_UNCERTAINTY_CONTRACT_PATH.read_text())
    contract["central_model_data_id"] = "different_chemistry"
    path = tmp_path / "foreign_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="identities differ"):
        decompose_updated_uncertainty(UpdatedOutputSurfaceInput(), contract_path=path)
