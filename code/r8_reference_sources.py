"""Candidate provenance for the tiny R8c 17O convention factor."""

from __future__ import annotations

from dataclasses import dataclass

from isotopes import R17_VSMOW


# The current Young-reproduction branch uses this small multiplier only on R8c.
# It is equivalent to using a slightly more precise 17RSMOW value in the
# water-source term while retaining the rounded reporting convention elsewhere.
R8C_REFERENCE_RATIO_FACTOR = 1.000180
R8C_EQUIVALENT_R17_SMOW = R17_VSMOW * R8C_REFERENCE_RATIO_FACTOR


@dataclass(frozen=True)
class R8ReferenceConvention:
    key: str
    label: str
    r17_smow: float
    factor_vs_rounded: float
    note: str


def r8_reference_conventions() -> tuple[R8ReferenceConvention, ...]:
    return (
        R8ReferenceConvention(
            "rounded_printed",
            "Rounded printed 17RSMOW",
            R17_VSMOW,
            1.0,
            "Common printed convention used in Young text and many isotope references.",
        ),
        R8ReferenceConvention(
            "candidate_precise_r8c",
            "Candidate precise 17RSMOW for R8c source term",
            R8C_EQUIVALENT_R17_SMOW,
            R8C_REFERENCE_RATIO_FACTOR,
            "Numerically equivalent to the current R8c convention factor; hidden-code provenance not confirmed.",
        ),
    )
