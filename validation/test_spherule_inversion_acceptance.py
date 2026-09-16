"""End-to-end acceptance tests for the intended cosmic-spherule workflow."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spherule_to_air_d17o import spherule_d17o_from_air
from web_api import app


client = TestClient(app)
SPHERULE_DELTA18_PERMIL = 43.269
SPHERULE_CAP_DELTA17_SIGMA_PERMIL = 0.060
SPHERULE_DELTA18_SIGMA_PERMIL = 0.500


def _synthetic_spherule_target(
    *, pco2_ppm: float, gpp_pgC_per_year: float, po2_pal: float
) -> tuple[float, float, float]:
    forward = client.post(
        "/api/v1/forward",
        json={
            "p_o2_pal": po2_pal,
            "p_co2_ppm": pco2_ppm,
            "gpp_pgC_per_year": gpp_pgC_per_year,
        },
    )
    assert forward.status_code == 200, forward.text
    air_target = forward.json()["result"]["central_cap_delta17_prime_permil"]
    spherule_target = spherule_d17o_from_air(
        air_target, SPHERULE_DELTA18_PERMIL
    )
    converted = client.post(
        "/api/v1/proxy/spherule-to-air",
        json={
            "cap_delta17_spherule_permil": spherule_target,
            "delta18_spherule_permil": SPHERULE_DELTA18_PERMIL,
            "cap_delta17_sigma_permil": SPHERULE_CAP_DELTA17_SIGMA_PERMIL,
            "delta18_sigma_permil": SPHERULE_DELTA18_SIGMA_PERMIL,
        },
    )
    assert converted.status_code == 200, converted.text
    result = converted.json()["result"]
    assert result["cap_delta17_air_o2_permil"] == pytest.approx(air_target)
    assert result["analytical_sigma_permil"] > SPHERULE_CAP_DELTA17_SIGMA_PERMIL
    return (
        result["cap_delta17_air_o2_permil"],
        result["analytical_sigma_permil"],
        spherule_target,
    )


@pytest.mark.parametrize(
    ("solve_for", "constraints", "truth"),
    [
        (
            "pCO2",
            {
                "gpp_constraint": {
                    "kind": "normal",
                    "center": 72.5,
                    "sigma": 14.5,
                },
                "po2_constraint": {"kind": "range", "lower": 0.4, "upper": 0.6},
            },
            8000.0,
        ),
        (
            "GPP",
            {
                "pco2_constraint": {
                    "kind": "normal",
                    "center": 8000.0,
                    "sigma": 1000.0,
                },
                "po2_constraint": {"kind": "range", "lower": 0.4, "upper": 0.6},
            },
            72.5,
        ),
        (
            "pO2",
            {
                "pco2_constraint": {
                    "kind": "normal",
                    "center": 8000.0,
                    "sigma": 1000.0,
                },
                "gpp_constraint": {"kind": "range", "lower": 58.0, "upper": 87.0},
            },
            0.5,
        ),
    ],
)
def test_realistic_spherule_recovers_each_coordinate_with_declared_constraints(
    solve_for: str, constraints: dict, truth: float
) -> None:
    air_target, analytical_sigma, _ = _synthetic_spherule_target(
        pco2_ppm=8000.0,
        gpp_pgC_per_year=72.5,
        po2_pal=0.5,
    )
    response = client.post(
        "/api/v1/inference/coordinate",
        json={
            "solve_for": solve_for,
            "target_air_cap_delta17_permil": air_target,
            "measurement_sigma_permil": analytical_sigma,
            **constraints,
            "pco2_grid_size": 181,
            "gpp_grid_size": 81,
            "po2_grid_size": 17,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    lower, upper = result["equal_tailed_credible_interval"]
    assert lower <= truth <= upper, (
        f"{solve_for} truth {truth:g} is outside the propagated interval "
        f"[{lower:g}, {upper:g}]"
    )
    assert sum(result["solve_marginal_probability_mass"]) == pytest.approx(1.0)
    expected_status = (
        "solve_boundary_sensitive"
        if result["solve_boundary_sensitive"]
        else "posterior_computed"
    )
    assert result["status"] == expected_status
    assert (
        "No log-uniform coordinate prior or probabilistic structural model "
        "discrepancy is included."
    ) in result["probability_scope"]


@pytest.mark.parametrize("pco2_ppm", [50.0, 60000.0])
def test_realistic_spherule_preserves_true_pco2_domain_edge_density_modes(
    pco2_ppm: float,
) -> None:
    air_target, analytical_sigma, _ = _synthetic_spherule_target(
        pco2_ppm=pco2_ppm,
        gpp_pgC_per_year=290.0,
        po2_pal=1.0,
    )
    response = client.post(
        "/api/v1/inference/coordinate",
        json={
            "solve_for": "pCO2",
            "target_air_cap_delta17_permil": air_target,
            "measurement_sigma_permil": analytical_sigma,
            "gpp_constraint": {"kind": "fixed", "center": 290.0},
            "po2_constraint": {"kind": "fixed", "center": 1.0},
            "pco2_grid_size": 181,
            "gpp_grid_size": 17,
            "po2_grid_size": 17,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    axis = result["solve_axis"]
    mass = result["solve_marginal_probability_mass"]
    density = result["solve_marginal_density"]
    mode = axis[density.index(max(density))]
    assert mode == pytest.approx(pco2_ppm)
    assert axis[0] == pytest.approx(50.0)
    assert axis[-1] == pytest.approx(60000.0)
    assert sum(mass) == pytest.approx(1.0)
    assert result["numerical_refinement_applied"] is False
