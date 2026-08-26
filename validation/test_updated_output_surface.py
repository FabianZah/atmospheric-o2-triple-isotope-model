from __future__ import annotations

import numpy as np
import pytest

from updated_output_surface import (
    SURFACE_FIELDS,
    UpdatedMolecularOutputSurface,
    UpdatedOutputSurfaceInput,
)


def _synthetic_bundle() -> dict[str, object]:
    po2 = np.asarray([0.1, 0.5, 1.0, 2.0])
    pco2 = np.asarray([294.0, 1000.0, 10000.0, 60000.0])
    gpp = np.asarray([1.0, 100.0, 290.0, 850.0])
    x, y, z = np.meshgrid(po2, pco2, np.log(gpp), indexing="ij")
    central = 0.2 * x - 0.0007 * y + 0.4 * z
    widths = {
        "source_isoflux": 0.01,
        "biological_process": 0.02,
        "combined_process": 0.03,
        "model_guardrail": 0.04,
    }
    fields = {
        "central_delta18_prime_permil": (20.0 + 0.1 * x + 0.2 * z).tolist(),
        "central_cap_delta17_prime_permil": central.tolist(),
    }
    for prefix, width in widths.items():
        fields[f"{prefix}_lower_cap_delta17_permil"] = (central - width).tolist()
        fields[f"{prefix}_upper_cap_delta17_permil"] = (central + width).tolist()
    return {
        "schema_version": 1,
        "surface_data_id": "synthetic",
        "upstream_model_data_id": "synthetic-kernel",
        "axes": {
            "po2_pal": po2.tolist(),
            "pco2_ppm": pco2.tolist(),
            "gpp_pgC_per_year": gpp.tolist(),
        },
        "fields": fields,
        "validation": {
            "cap_delta17_maximum_absolute_residual_permil": 0.005,
        },
    }


def test_cubic_surface_is_exact_for_affine_function_in_declared_coordinates() -> None:
    surface = UpdatedMolecularOutputSurface(_synthetic_bundle())
    request = UpdatedOutputSurfaceInput(0.5, 5000.0, 100.0)
    result = surface.evaluate(request)
    expected = 0.2 * 0.5 - 0.0007 * 5000.0 + 0.4 * np.log(100.0)
    assert result.central_cap_delta17_prime_permil == pytest.approx(expected)
    assert result.accelerated_model_guardrail_interval_cap_delta17_permil == pytest.approx(
        (expected - 0.045, expected + 0.045)
    )
    assert result.extrapolation_applied is False
    assert result.central_delta18_acceleration_validated is False


def test_training_nodes_are_reproduced_exactly() -> None:
    bundle = _synthetic_bundle()
    surface = UpdatedMolecularOutputSurface(bundle)
    request = UpdatedOutputSurfaceInput(1.0, 1000.0, 290.0)
    result = surface.evaluate(request)
    expected = bundle["fields"]["central_cap_delta17_prime_permil"][2][1][2]
    assert result.central_cap_delta17_prime_permil == pytest.approx(expected, abs=1.0e-10)


def test_dedicated_delta18_surface_is_exact_at_training_nodes() -> None:
    bundle = _synthetic_bundle()
    po2 = np.asarray(bundle["axes"]["po2_pal"])
    pco2 = np.asarray(bundle["axes"]["pco2_ppm"])
    gpp = np.asarray(bundle["axes"]["gpp_pgC_per_year"])
    x, y, z = np.meshgrid(po2, np.log(pco2), np.log(gpp), indexing="ij")
    values = 20.0 + 0.1 * x + 0.02 * y + 0.2 * z
    bundle["delta18_surface"] = {
        "axes": bundle["axes"],
        "values": values.tolist(),
    }
    bundle["validation"]["delta18_acceleration_validated"] = True
    result = UpdatedMolecularOutputSurface(bundle).evaluate(
        UpdatedOutputSurfaceInput(1.0, 1000.0, 290.0)
    )
    assert result.central_delta18_prime_permil == pytest.approx(values[2, 1, 2])
    assert result.central_delta18_acceleration_validated is True


@pytest.mark.parametrize(
    "surface_input",
    (
        UpdatedOutputSurfaceInput(0.09, 1000.0, 290.0),
        UpdatedOutputSurfaceInput(1.0, 60001.0, 290.0),
        UpdatedOutputSurfaceInput(1.0, 1000.0, 851.0),
    ),
)
def test_surface_rejects_extrapolation(surface_input) -> None:
    with pytest.raises(ValueError, match="outside output-surface domain"):
        UpdatedMolecularOutputSurface(_synthetic_bundle()).evaluate(surface_input)


def test_surface_rejects_invalid_interval_order() -> None:
    bundle = _synthetic_bundle()
    bundle["fields"]["model_guardrail_lower_cap_delta17_permil"][0][0][0] = 999.0
    with pytest.raises(ValueError, match="interval order"):
        UpdatedMolecularOutputSurface(bundle)


def test_surface_schema_requires_exact_field_set() -> None:
    bundle = _synthetic_bundle()
    del bundle["fields"][SURFACE_FIELDS[0]]
    with pytest.raises(ValueError, match="fields differ"):
        UpdatedMolecularOutputSurface(bundle)
