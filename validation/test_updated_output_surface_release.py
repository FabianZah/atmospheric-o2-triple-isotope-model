"""Release checks for the versioned updated-model D17O accelerator."""

from __future__ import annotations

import pytest

from updated_molecular_forward_model import UpdatedForwardInput, run_updated_forward
from updated_output_surface import (
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)


def test_release_metadata_and_domain() -> None:
    surface = load_updated_output_surface()
    assert surface.surface_data_id == "updated_molecular_output_surface_v1"
    assert surface.upstream_model_data_id == "updated_r7_response_surface_v1"
    assert surface.domain == {
        "po2_pal": (0.1, 2.0),
        "pco2_ppm": (50.0, 60000.0),
        "gpp_pgC_per_year": (18.256263999999998, 850.0),
    }
    assert surface.delta18_acceleration_validated is True
    assert surface.maximum_holdout_cap_delta17_residual_permil == pytest.approx(
        0.004940448570907918
    )


@pytest.mark.parametrize(
    "inputs",
    (
        (1.0, 50.0, 290.0),
        (1.0, 75.0, 290.0),
        (1.0, 200.0, 290.0),
        (1.0, 294.0, 290.0),
        (0.5, 4200.0, 182.56264),
        (2.0, 60000.0, 850.0),
    ),
)
def test_release_nodes_reproduce_live_delta17_and_expand_guardrail(inputs) -> None:
    surface = load_updated_output_surface()
    accelerated = surface.evaluate(UpdatedOutputSurfaceInput(*inputs))
    live = run_updated_forward(UpdatedForwardInput(*inputs))

    assert accelerated.central_cap_delta17_prime_permil == pytest.approx(
        live.central_cap_delta17_prime_permil, abs=5.0e-5
    )
    assert accelerated.central_delta18_prime_permil == pytest.approx(
        live.central_delta18_prime_permil, abs=0.05
    )
    assert accelerated.interpolated_kernel_guardrail_interval_cap_delta17_permil == pytest.approx(
        live.model_guardrail_interval_cap_delta17_permil, abs=5.0e-5
    )
    interpolation_error = accelerated.output_surface_interpolation_guardrail_permil
    assert accelerated.accelerated_model_guardrail_interval_cap_delta17_permil == pytest.approx(
        (
            accelerated.interpolated_kernel_guardrail_interval_cap_delta17_permil[0]
            - interpolation_error,
            accelerated.interpolated_kernel_guardrail_interval_cap_delta17_permil[1]
            + interpolation_error,
        )
    )
    assert accelerated.central_delta18_acceleration_validated is True
