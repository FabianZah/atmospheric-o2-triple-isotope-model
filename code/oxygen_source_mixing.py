"""Isotope-ratio mixing for oxygen contributed by two source reservoirs.

This module contains only the mechanical two-source mass balance. It is valid
when the supplied source fraction represents the mixing fraction used for the
17O/16O and 18O/16O ratios and any process fractionation has already been
accounted for. Process-specific transfer models, including pyrite oxidation to
sulfate, belong in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from isotopes import (
    R17_VSMOW,
    R18_VSMOW,
    cap_delta17_from_primes,
    conventional_delta_from_prime,
)


REFERENCE_SLOPE = 0.528


@dataclass(frozen=True)
class OxygenIsotopeComposition:
    """Oxygen-isotope composition in logarithmic delta notation, in per mil."""

    delta18_prime_permil: float
    cap_delta17_permil: float
    reference_slope: float = REFERENCE_SLOPE

    @classmethod
    def from_delta18_and_cap_delta17(
        cls,
        delta18_permil: float,
        cap_delta17_permil: float,
        *,
        reference_slope: float = REFERENCE_SLOPE,
    ) -> "OxygenIsotopeComposition":
        if delta18_permil <= -1000.0:
            raise ValueError("delta18O must be greater than -1000 per mil")
        delta18_prime = 1000.0 * log(1.0 + delta18_permil / 1000.0)
        return cls(delta18_prime, cap_delta17_permil, reference_slope)

    @classmethod
    def from_ratios(
        cls,
        ratio17: float,
        ratio18: float,
        *,
        reference_slope: float = REFERENCE_SLOPE,
    ) -> "OxygenIsotopeComposition":
        if ratio17 <= 0.0 or ratio18 <= 0.0:
            raise ValueError("oxygen-isotope ratios must be positive")
        delta17_prime = 1000.0 * log(ratio17 / R17_VSMOW)
        delta18_prime = 1000.0 * log(ratio18 / R18_VSMOW)
        return cls(
            delta18_prime,
            cap_delta17_from_primes(
                delta17_prime,
                delta18_prime,
                slope=reference_slope,
            ),
            reference_slope,
        )

    @property
    def delta17_prime_permil(self) -> float:
        return self.cap_delta17_permil + self.reference_slope * self.delta18_prime_permil

    @property
    def delta18_permil(self) -> float:
        return conventional_delta_from_prime(self.delta18_prime_permil)

    @property
    def delta17_permil(self) -> float:
        return conventional_delta_from_prime(self.delta17_prime_permil)

    @property
    def ratio17(self) -> float:
        return R17_VSMOW * exp(self.delta17_prime_permil / 1000.0)

    @property
    def ratio18(self) -> float:
        return R18_VSMOW * exp(self.delta18_prime_permil / 1000.0)


def mix_oxygen_sources(
    source_a: OxygenIsotopeComposition,
    source_b: OxygenIsotopeComposition,
    fraction_a: float,
) -> OxygenIsotopeComposition:
    """Return the ratio-space mixture of two oxygen sources."""

    if not 0.0 <= fraction_a <= 1.0:
        raise ValueError("source fraction must be between 0 and 1")
    if source_a.reference_slope != source_b.reference_slope:
        raise ValueError("source compositions must use the same reference slope")
    fraction_b = 1.0 - fraction_a
    return OxygenIsotopeComposition.from_ratios(
        fraction_a * source_a.ratio17 + fraction_b * source_b.ratio17,
        fraction_a * source_a.ratio18 + fraction_b * source_b.ratio18,
        reference_slope=source_a.reference_slope,
    )


def recover_source_a(
    mixture: OxygenIsotopeComposition,
    source_b: OxygenIsotopeComposition,
    fraction_a: float,
) -> OxygenIsotopeComposition:
    """Recover source A from a two-source mixture and known source B."""

    if not 0.0 < fraction_a <= 1.0:
        raise ValueError("source fraction must be greater than 0 and at most 1")
    if mixture.reference_slope != source_b.reference_slope:
        raise ValueError("mixture and source composition must use the same reference slope")
    fraction_b = 1.0 - fraction_a
    return OxygenIsotopeComposition.from_ratios(
        (mixture.ratio17 - fraction_b * source_b.ratio17) / fraction_a,
        (mixture.ratio18 - fraction_b * source_b.ratio18) / fraction_a,
        reference_slope=mixture.reference_slope,
    )
