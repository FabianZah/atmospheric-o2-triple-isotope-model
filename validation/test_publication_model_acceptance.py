"""Regression tests for the integrated publication-model acceptance audit."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "validation") not in sys.path:
    sys.path.insert(0, str(ROOT / "validation"))

from audit_publication_model_acceptance import SOURCE_PATHS, build_report  # noqa: E402


def test_acceptance_sources_are_versioned_release_inputs() -> None:
    for name, path in SOURCE_PATHS.items():
        if name != "contract":
            assert path.is_relative_to(ROOT / "model_data" / "validation_evidence")
        assert path.is_file()


def test_publication_model_has_no_release_blockers() -> None:
    report = build_report()
    assert report["release_blocker_count"] == 0
    assert report["verdict"] == (
        "accepted_for_steady_forward_inverse_and_declared_step_response"
    )


def test_qualified_evidence_does_not_mutate_the_central_model() -> None:
    report = build_report()
    assert report["central_model_changed_by_audit"] is False
    statuses = {row["gate"]: row["status"] for row in report["rows"]}
    assert statuses["Young Fig. 8 historical response anchor"] == "qualified"
    assert statuses["Marine O2 accessibility"] == "excluded"
    assert statuses["pCO2-dependent climate profile"] == "excluded"


def test_claim_boundary_keeps_inverse_nonuniqueness_explicit() -> None:
    report = build_report()
    unsupported = " ".join(report["claims_not_supported"])
    assert "unique simultaneous recovery" in unsupported
    assert "whole-domain Gaussian" in unsupported
