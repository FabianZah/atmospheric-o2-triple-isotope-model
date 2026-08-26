"""Regression test for the public package entry points and runtime data."""

from __future__ import annotations

from smoke_publication_package import build_smoke_report


def test_publication_package_smoke() -> None:
    report = build_smoke_report()
    assert report["status"] == "pass"
    assert report["runtime_dependencies"]["PyYAML"]
    assert report["publication_model_id"] == "atmospheric_o2_triple_isotope_model_v1"
    assert report["public_interface"] == "FastAPI with static browser frontend"
    assert abs(report["inverse_roundtrip_pCO2_ppm"] - 294.0) < 1.0e-5
