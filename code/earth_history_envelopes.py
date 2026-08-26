"""Practical Earth-history input envelopes for model scenarios."""

from __future__ import annotations

from dataclasses import dataclass


PCO2_RECOMMENDED_MAX_PPM = 30000.0
PCO2_EXPLORATORY_MAX_PPM = 100000.0


@dataclass(frozen=True)
class EnvelopeStatus:
    level: str
    message: str


def pco2_envelope_status(pco2_ppm: float) -> EnvelopeStatus:
    """Return a user-facing pCO2 envelope classification.

    The threshold is a practical modeling guard rail, not a geological limit.
    It keeps Young-range validation separate from exploratory extrapolation
    that may be needed to explain very low Delta'17O values.
    """

    if pco2_ppm <= PCO2_RECOMMENDED_MAX_PPM:
        return EnvelopeStatus(
            "recommended",
            "pCO2 is within the main Young Fig. 8 validation range and current recommended working envelope.",
        )
    if pco2_ppm <= PCO2_EXPLORATORY_MAX_PPM:
        return EnvelopeStatus(
            "exploratory",
            "pCO2 is outside Young Fig. 8; use this as exploratory extrapolation for low Delta'17O, not exact calibration.",
        )
    return EnvelopeStatus(
        "outside_configured_range",
        "pCO2 exceeds the configured exploratory range; extend plots deliberately and document the extrapolation.",
    )
