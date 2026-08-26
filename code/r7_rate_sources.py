"""Candidate provenance for the R7 O(1D)+CO2 throughput convention.

The Young reconstruction currently needs an R7 throughput multiplier near
2.38. This module keeps the candidate literature-rate explanations separate
from fitted model behavior, so the branch policy can distinguish a rate-source
choice from an arbitrary tuning factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from young_model_inventory import PARAMETERS


YOUNG_PRINTED_R7A_CM3_S = PARAMETERS["k_R7a_cm3_s"]
CURRENT_R7_THROUGHPUT_FACTOR = 2.3836445078
YUNG_1991_PREEXP_CM3_S = 7.4e-11
YUNG_1991_ACTIVATION_K = 120.0
YUNG_1991_TEXT_298K_CM3_S = 1.1e-10


@dataclass(frozen=True)
class R7RateSource:
    key: str
    label: str
    k_cm3_s: float
    note: str

    @property
    def factor_vs_young(self) -> float:
        return self.k_cm3_s / YOUNG_PRINTED_R7A_CM3_S


def yung_1991_rate_cm3_s(temperature_k: float) -> float:
    """Yung et al. (1991) k = 7.4e-11 exp(120/T) cm3 s-1."""

    return YUNG_1991_PREEXP_CM3_S * exp(YUNG_1991_ACTIVATION_K / temperature_k)


def yung_equivalent_temperature_k(rate_cm3_s: float) -> float:
    """Temperature that would give a rate in the Yung et al. expression."""

    return YUNG_1991_ACTIVATION_K / log(rate_cm3_s / YUNG_1991_PREEXP_CM3_S)


def rate_sources() -> tuple[R7RateSource, ...]:
    current_implied_rate = CURRENT_R7_THROUGHPUT_FACTOR * YOUNG_PRINTED_R7A_CM3_S
    return (
        R7RateSource(
            "young_printed_table2",
            "Young Table 2 printed kR7a",
            YOUNG_PRINTED_R7A_CM3_S,
            "Printed in Young et al. Table 2 for R7a at 220 K.",
        ),
        R7RateSource(
            "current_candidate_factor",
            "Current candidate factor applied to Young kR7a",
            current_implied_rate,
            "Current best static/dynamic candidate: 2.3836445078 times Young kR7a.",
        ),
        R7RateSource(
            "yung_1991_text_298k",
            "Yung et al. 1991 text value near 298 K",
            YUNG_1991_TEXT_298K_CM3_S,
            "Yung text reports about 1.1e-10 cm3 s-1 at 298 K for O(1D)+CO2.",
        ),
        R7RateSource(
            "yung_1991_formula_298k",
            "Yung et al. 1991 formula at 298 K",
            yung_1991_rate_cm3_s(298.0),
            "k = 7.4e-11 exp(120/T), evaluated at 298 K.",
        ),
        R7RateSource(
            "yung_1991_formula_220k",
            "Yung et al. 1991 formula at 220 K",
            yung_1991_rate_cm3_s(220.0),
            "k = 7.4e-11 exp(120/T), evaluated at Young's stated stratospheric temperature.",
        ),
    )
