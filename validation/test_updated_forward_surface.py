"""Regression gate for the updated forward model's physical surface shape."""

from __future__ import annotations

from pathlib import Path

from audit_updated_forward_surface import run


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)


def test_updated_surface_has_physical_shape_and_unique_tested_roots() -> None:
    report = run(output_path=ROOT / "outputs" / "updated_forward_surface_test.json")
    assert report["status"] == "shape_and_inverse_gates_passed"
    assert all(report["gates"].values())
    assert report["candidate_promoted_to_public_default"] is False
    assert report["maximum_fixed_point_residual_permil"] <= 1.0e-8
    assert report["remaining_promotion_gates"]
