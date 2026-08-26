"""pO2 conversion helpers and Phanerozoic working ranges.

The Young-style model uses `p_o2_pal` relative to its modern Table-3 O2
reservoir. Mills et al. (2023) report atmospheric O2 in percent. Keeping the
conversion explicit avoids mixing percent O2, Mills-relative PAL, and
Young/Table-3 PAL in later Phanerozoic sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass


YOUNG_TABLE3_MODERN_O2_PERCENT = 21.2
MILLS_MODERN_O2_PERCENT = 20.6449666166846

MILLS_PHANEROZOIC_MIN_PERCENT = 0.0585816586819024
MILLS_PHANEROZOIC_MAX_PERCENT = 41.8852363914043
MILLS_PHANEROZOIC_MID_MIN_PERCENT = 1.78634812474959
MILLS_PHANEROZOIC_MID_MAX_PERCENT = 37.2382265730289

MIDDLE_ORDOVICIAN_MID_MEAN_PERCENT = 8.067416720656917
MIDDLE_ORDOVICIAN_MIN_MEAN_PERCENT = 4.39716189592825
MIDDLE_ORDOVICIAN_MAX_MEAN_PERCENT = 22.50477725216155

# Practical model-audit envelope requested for Phanerozoic use. The lower
# bound is intentionally above the most extreme Mills lower envelope; those
# very low values remain edge-case sensitivity tests.
MODEL_PO2_WORKING_MIN_PAL = 0.1
MODEL_PO2_WORKING_MAX_PAL = 2.1


@dataclass(frozen=True)
class O2PercentEquivalent:
    p_o2_pal: float
    young_percent_o2: float
    mills_percent_o2: float


def pal_to_percent(p_o2_pal: float, *, convention: str = "young") -> float:
    """Convert PAL to atmospheric O2 percent."""

    if convention == "young":
        return p_o2_pal * YOUNG_TABLE3_MODERN_O2_PERCENT
    if convention == "mills":
        return p_o2_pal * MILLS_MODERN_O2_PERCENT
    raise ValueError("convention must be 'young' or 'mills'")


def percent_to_pal(o2_percent: float, *, convention: str = "young") -> float:
    """Convert atmospheric O2 percent to PAL."""

    if convention == "young":
        return o2_percent / YOUNG_TABLE3_MODERN_O2_PERCENT
    if convention == "mills":
        return o2_percent / MILLS_MODERN_O2_PERCENT
    raise ValueError("convention must be 'young' or 'mills'")


def percent_equivalent(p_o2_pal: float) -> O2PercentEquivalent:
    """Return percent-O2 equivalents under both modern-O2 conventions."""

    return O2PercentEquivalent(
        p_o2_pal=p_o2_pal,
        young_percent_o2=pal_to_percent(p_o2_pal, convention="young"),
        mills_percent_o2=pal_to_percent(p_o2_pal, convention="mills"),
    )


def in_working_po2_envelope(p_o2_pal: float) -> bool:
    return MODEL_PO2_WORKING_MIN_PAL <= p_o2_pal <= MODEL_PO2_WORKING_MAX_PAL
