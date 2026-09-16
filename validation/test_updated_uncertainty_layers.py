from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def test_exact_clima_node_is_reported_without_interpolation() -> None:
    result = decompose_updated_uncertainty(
        UpdatedOutputSurfaceInput(1.0, 30000.0, 290.0),
    )

    assert len(result.structural_endmember_points) == 2
    points = {
        point.pressure_convention: point
        for point in result.structural_endmember_points
    }
    assert set(points) == {"additive_co2", "fixed_total_dry_major"}
    assert points["additive_co2"].endmember_id == (
        "clima_0p5p11_1pal_additive_co2_v1"
    )
    assert all(not point.interpolation_applied for point in points.values())
    assert np.isclose(
        points["additive_co2"].offset_from_central_permil,
        3.092283101529705,
    )
    assert np.isclose(
        points["fixed_total_dry_major"].offset_from_central_permil,
        3.0734650918169386,
    )


def test_positive_measurement_sigma_requires_source() -> None:
    with pytest.raises(ValueError, match="traceable source"):
        decompose_updated_uncertainty(
            UpdatedOutputSurfaceInput(), measurement_sigma_permil=0.01
        )


@pytest.mark.parametrize("po2", [0.1, 2.0])
def test_exact_nonmodern_po2_clima_node_is_reported(po2: float) -> None:
    result = decompose_updated_uncertainty(
        UpdatedOutputSurfaceInput(po2, 30000.0, 290.0),
    )

    assert len(result.structural_endmember_points) == 1
    point = result.structural_endmember_points[0]
    assert point.pressure_convention == "additive_co2"
    assert not point.interpolation_applied
