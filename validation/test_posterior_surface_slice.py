"""The faster nuisance-coordinate evaluation must preserve the original surface."""
import numpy as np
import pytest

from posterior_surface_slice import COORDINATES, prepare_coordinate_slice
from updated_output_surface import load_updated_output_surface
from posterior_coordinate_quadrature import integrated_coordinate_log_likelihood
from test_posterior_coordinate_quadrature import _request


@pytest.mark.parametrize("coordinate", COORDINATES)
@pytest.mark.parametrize("include_delta18", (False, True))
def test_contraction_matches_existing_surface_on_random_points_and_edges(coordinate, include_delta18):
    surface = load_updated_output_surface()
    rng = np.random.default_rng(102)
    axes = (surface.po2_nodes, surface.pco2_nodes, surface.gpp_nodes)
    base = {c: rng.choice(a, 257) for c, a in zip(COORDINATES, axes)}
    prepared = prepare_coordinate_slice(surface, base, coordinate, include_delta18)
    axis = axes[COORDINATES.index(coordinate)]
    ids = rng.integers(0, 257, 3001)
    values = np.r_[axis, np.exp(rng.uniform(np.log(axis[0]), np.log(axis[-1]), len(ids)-len(axis)))]
    values = np.clip(values, axis[0], axis[-1])
    points = {c: a[ids] for c, a in base.items()}
    points[coordinate] = values
    kwargs = dict(zip(("p_o2_pal", "p_co2_ppm", "gpp_pgC_per_year"), (points[c] for c in COORDINATES)))
    d17, d18 = prepared.evaluate(ids, values)
    np.testing.assert_allclose(d17, surface.evaluate_central_cap_delta17_grid(**kwargs), atol=1e-11, rtol=1e-12)
    if include_delta18:
        np.testing.assert_allclose(d18, surface.evaluate_central_delta18_prime_grid(**kwargs), atol=1e-11, rtol=1e-12)
    else:
        assert d18 is None
    with pytest.raises(ValueError, match="domain"):
        prepared.evaluate(ids[:1], np.array([axis[-1]+1]))


def test_complete_integral_matches_unoptimized_evaluator(monkeypatch):
    surface = load_updated_output_surface()
    request = _request(target=-11)
    request.po2_prior_mean, request.po2_prior_sigma = .2, .2
    z = np.linspace(.1, 1., 33)
    fields = {"pO2": z[None, :], "pCO2": np.linspace(18000, 60000, 91)[:, None], "GPP": 522.}
    fast, _ = integrated_coordinate_log_likelihood(surface, request, fields, z, "pO2")
    monkeypatch.setattr("posterior_coordinate_quadrature.prepare_coordinate_slice", lambda *args: None)
    original, _ = integrated_coordinate_log_likelihood(surface, request, fields, z, "pO2")
    # Compare posterior mass, including extreme tails, without division by zero.
    shift = np.max(original)
    old, new = np.exp(original-shift), np.exp(fast-shift)
    assert np.sum(abs(new-old))/np.sum(old) < 1e-8


def test_resolution_limit_has_a_useful_public_error(monkeypatch):
    from fastapi.testclient import TestClient
    import web_api
    from posterior_coordinate_quadrature import PosteriorResolutionError

    def fail(_request):
        raise PosteriorResolutionError("internal time budget diagnostics")

    monkeypatch.setattr(web_api, "constrained_coordinate", fail)
    with TestClient(web_api.app) as client:
        response = client.post("/api/v1/inference/coordinate", json={
            "solve_for": "pCO2", "target_air_cap_delta17_permil": -11,
            "measurement_sigma_permil": .015,
            "gpp_constraint": {"kind": "normal", "center": 522, "sigma": 29},
            "po2_constraint": {"kind": "normal", "center": .2, "sigma": .2},
        })
    assert response.status_code == 503
    assert response.json()["code"] == "posterior_resolution_limit"
    assert "No result was returned" in response.json()["detail"]
