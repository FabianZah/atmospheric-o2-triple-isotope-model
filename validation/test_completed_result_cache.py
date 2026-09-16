"""Completed-result reuse must preserve numbers, metadata and resource limits."""

from copy import deepcopy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import random

from fastapi.testclient import TestClient
from openpyxl import load_workbook
import pytest

from completed_result_cache import CompletedResultCache, completed_result_key
from model_result_workbook import build_coordinate_inference_workbook
import web_api


def test_round_trip_is_exact_and_callers_cannot_mutate_stored_results():
    cache = CompletedResultCache()
    original = {"values": [0.0, -0.0, 0.0000000000001, 614.1437230995808],
                "mask": [True, False], "note": "test", "missing": None}
    assert cache.put("a", original)
    assert cache.get("a") == original
    first = cache.get("a")
    first["values"][0] = 99
    original["mask"][0] = False
    assert cache.get("a")["values"][0] == 0
    assert cache.get("a")["mask"][0] is True


def test_expiry_is_absolute_and_clear_discards_everything():
    clock = [0.]
    cache = CompletedResultCache(ttl_seconds=10, clock=lambda: clock[0])
    cache.put("a", {"result": 1})
    clock[0] = 9
    assert cache.get("a") is not None
    clock[0] = 10
    assert cache.get("a") is None
    assert cache._bytes == 0
    cache.put("b", {"result": 2})
    cache.clear()
    assert cache.get("b") is None
    assert cache._bytes == 0


def test_lru_eviction_and_byte_budget_are_enforced():
    cache = CompletedResultCache(max_entries=2)
    for key in ("a", "b"):
        cache.put(key, {"result": key})
    cache.get("a")
    cache.put("c", {"result": "c"})
    assert cache.get("b") is None
    assert cache.get("a") and cache.get("c")
    size = cache._bytes
    bounded = CompletedResultCache(max_bytes=size - 1)
    bounded.put("a", {"result": "a"})
    bounded.put("b", {"result": "b"})
    assert bounded._bytes <= bounded.max_bytes
    assert bounded.get("a") is None


def test_oversize_and_nonfinite_results_are_not_stored():
    cache = CompletedResultCache(max_result_bytes=100)
    assert not cache.put("large", {"values": list(range(1000))})
    assert cache.get("large") is None
    randomizer = random.Random(1)
    cache = CompletedResultCache(max_bytes=100)
    assert not cache.put("incompressible", {"values": [randomizer.random() for _ in range(1000)]})
    assert cache._bytes == 0
    with pytest.raises(ValueError):
        cache.put("invalid", {"value": float("nan")})
    assert cache.get("invalid") is None


def test_concurrent_writes_and_reads_remain_bounded():
    cache = CompletedResultCache(max_entries=4, max_bytes=5000)
    def job(i):
        cache.put(str(i), {"value": i})
        result = cache.get(str(i))
        assert result is None or result == {"value": i}
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(job, range(40)))
    assert len(cache._entries) <= 4
    assert cache._bytes <= cache.max_bytes


def test_key_is_order_independent_and_includes_inputs_and_model():
    key = completed_result_key({"a": 1, "b": 2}, {"version": "v1"})
    assert key == completed_result_key({"b": 2, "a": 1}, {"version": "v1"})
    assert key != completed_result_key({"a": 1, "b": 3}, {"version": "v1"})
    assert key != completed_result_key({"a": 1, "b": 2}, {"version": "v2"})


@pytest.fixture
def fresh_export_cache(monkeypatch):
    cache = CompletedResultCache()
    monkeypatch.setattr(web_api, "_COORDINATE_EXPORT_CACHE", cache)
    return cache


def air_request():
    return {
        "solve_for": "pCO2", "target_air_cap_delta17_permil": -0.432,
        "measurement_sigma_permil": 0.015,
        "gpp_constraint": {"kind": "fixed", "center": 290.0},
        "po2_constraint": {"kind": "fixed", "center": 1.0},
    }


def workbook_cells(content):
    book = load_workbook(BytesIO(content), data_only=False)
    values = {}
    for sheet in book:
        rows = []
        for row in sheet.values:
            if row[0] == "generated_utc":
                assert datetime.fromisoformat(row[1]).tzinfo is not None
                row = (row[0], "<export timestamp>", *row[2:])
            rows.append(row)
        values[sheet.title] = rows
    return values


@pytest.mark.parametrize("coordinate", ["pCO2", "GPP", "pO2"])
@pytest.mark.parametrize("source", ["Direct air O2", "I-type cosmic spherule", "Sulfate"])
def test_export_reuses_completed_calculation_and_matches_fresh_workbook(
    monkeypatch, fresh_export_cache, coordinate, source,
):
    from test_sulfate_web_api import payload
    request = payload(coordinate) if source == "Sulfate" else air_request()
    if source != "Sulfate" and coordinate != "pCO2":
        request["solve_for"] = coordinate
        request.pop("gpp_constraint" if coordinate == "GPP" else "po2_constraint")
        request["pco2_constraint"] = {"kind": "fixed", "center": 294.0}
    context = {"isotope_source": source}
    if source == "I-type cosmic spherule":
        context["spherule"] = {
            "cap_delta17_permil": -0.66, "cap_delta17_sigma_permil": 0.06,
            "delta18_permil": 43.269, "delta18_sigma_permil": 0.5,
        }
    with TestClient(web_api.app) as client:
        response = client.post("/api/v1/inference/coordinate", json=request)
        assert response.status_code == 200, response.text
        expected = workbook_cells(build_coordinate_inference_workbook(response.json(), context))
        def should_not_run(_request):
            pytest.fail("export repeated a completed inference")
        monkeypatch.setattr(web_api, "constrained_coordinate", should_not_run)
        for _ in range(2):
            download = client.post("/api/v1/export/coordinate.xlsx",
                                   json={"inference": request, "context": context})
            assert download.status_code == 200, download.text
            assert workbook_cells(download.content) == expected


@pytest.mark.parametrize("reason", ["changed_input", "changed_model", "expired"])
def test_changed_or_expired_results_are_recalculated(monkeypatch, fresh_export_cache, reason):
    request = air_request()
    context = {"isotope_source": "Direct air O2"}
    calls = []
    original = web_api.constrained_coordinate
    def counted(inputs):
        calls.append(inputs)
        return original(inputs)
    monkeypatch.setattr(web_api, "constrained_coordinate", counted)
    with TestClient(web_api.app) as client:
        assert client.post("/api/v1/inference/coordinate", json=request).status_code == 200
        if reason == "changed_input":
            request["measurement_sigma_permil"] = 0.03
        elif reason == "changed_model":
            metadata = deepcopy(web_api.model_metadata())
            metadata["publication_model_id"] += "_new"
            monkeypatch.setattr(web_api, "model_metadata", lambda: metadata)
        else:
            fresh_export_cache.clock = lambda: float("inf")
        response = client.post("/api/v1/export/coordinate.xlsx",
                               json={"inference": request, "context": context})
        assert response.status_code == 200, response.text
    assert len(calls) == 2


def test_failed_calculation_is_not_cached_and_client_results_are_rejected(monkeypatch, fresh_export_cache):
    def fail(_inputs):
        raise ValueError("unfinished calculation")
    monkeypatch.setattr(web_api, "constrained_coordinate", fail)
    with TestClient(web_api.app) as client:
        response = client.post("/api/v1/inference/coordinate", json=air_request())
        assert response.status_code == 422
        assert fresh_export_cache._bytes == 0
        response = client.post("/api/v1/export/coordinate.xlsx", json={
            "inference": air_request(), "context": {"isotope_source": "Direct air O2"},
            "result": {"posterior_median": 999},
        })
        assert response.status_code == 422


def test_explicit_calculate_still_runs_the_model(monkeypatch, fresh_export_cache):
    calls = []
    original = web_api.constrained_coordinate
    def counted(inputs):
        calls.append(inputs)
        return original(inputs)
    monkeypatch.setattr(web_api, "constrained_coordinate", counted)
    with TestClient(web_api.app) as client:
        for _ in range(2):
            assert client.post("/api/v1/inference/coordinate", json=air_request()).status_code == 200
    assert len(calls) == 2


def test_joint_field_and_updated_export_context_reuse_exact_server_result(monkeypatch, fresh_export_cache):
    request = air_request()
    request["gpp_constraint"] = {"kind": "normal", "center": 290., "sigma": 29.}
    with TestClient(web_api.app) as client:
        result = client.post("/api/v1/inference/coordinate", json=request).json()
        def should_not_run(_inputs):
            pytest.fail("joint-field export repeated inference")
        monkeypatch.setattr(web_api, "constrained_coordinate", should_not_run)
        for source in ["Direct air O2", "I-type cosmic spherule"]:
            context = {"isotope_source": source}
            if source == "I-type cosmic spherule":
                context["spherule"] = {
                    "cap_delta17_permil": -0.66, "cap_delta17_sigma_permil": 0.06,
                    "delta18_permil": 43.269, "delta18_sigma_permil": 0.5,
                }
            response = client.post("/api/v1/export/coordinate.xlsx",
                                   json={"inference": request, "context": context})
            assert response.status_code == 200, response.text
            expected = workbook_cells(build_coordinate_inference_workbook(result, context))
            assert "Joint probability" in expected
            assert workbook_cells(response.content) == expected


def test_large_streamed_result_round_trip_and_capacity_fallback(monkeypatch, fresh_export_cache):
    payload = {"values": list(range(40000))}
    assert fresh_export_cache.put("large", payload)
    assert fresh_export_cache.get("large") == payload
    tiny = CompletedResultCache(max_result_bytes=10)
    monkeypatch.setattr(web_api, "_COORDINATE_EXPORT_CACHE", tiny)
    calls = []
    original = web_api.constrained_coordinate
    def counted(inputs):
        calls.append(inputs)
        return original(inputs)
    monkeypatch.setattr(web_api, "constrained_coordinate", counted)
    with TestClient(web_api.app) as client:
        assert client.post("/api/v1/inference/coordinate", json=air_request()).status_code == 200
        response = client.post("/api/v1/export/coordinate.xlsx", json={
            "inference": air_request(), "context": {"isotope_source": "Direct air O2"},
        })
        assert response.status_code == 200, response.text
    assert len(calls) == 2
