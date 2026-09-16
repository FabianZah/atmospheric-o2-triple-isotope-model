"""Source-backed modern isotope reference compositions."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p

import numpy as np

from isotopes import R17_VSMOW, R18_VSMOW


@dataclass(frozen=True)
class PrimeIsotopeComposition:
    """One oxygen-isotope composition in logarithmic per-mil notation."""

    delta18_prime_permil: float
    cap_delta17_prime_permil: float
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta18_prime_permil):
            raise ValueError("delta18-prime must be finite")
        if not np.isfinite(self.cap_delta17_prime_permil):
            raise ValueError("Delta-prime-17O must be finite")
        if not self.source:
            raise ValueError("isotope composition requires provenance")

    @classmethod
    def from_delta18_and_cap_delta17(
        cls,
        *,
        delta18_permil: float,
        cap_delta17_prime_permil: float,
        source: str,
    ) -> "PrimeIsotopeComposition":
        """Convert conventional delta18O to the solver's logarithmic notation."""
        if not np.isfinite(delta18_permil) or delta18_permil <= -1000.0:
            raise ValueError("delta18 must be finite and greater than -1000 per mil")
        return cls(
            delta18_prime_permil=1000.0 * log1p(delta18_permil / 1000.0),
            cap_delta17_prime_permil=cap_delta17_prime_permil,
            source=source,
        )

    @property
    def delta17_prime_permil(self) -> float:
        return self.cap_delta17_prime_permil + 0.528 * self.delta18_prime_permil

    @property
    def ratio17(self) -> float:
        return R17_VSMOW * exp(self.delta17_prime_permil / 1000.0)

    @property
    def ratio18(self) -> float:
        return R18_VSMOW * exp(self.delta18_prime_permil / 1000.0)


def modern_reference_isotope_compositions() -> tuple[
    PrimeIsotopeComposition, PrimeIsotopeComposition
]:
    """Return the source-backed O2 and upper-tropospheric CO2 boundaries."""
    o2 = PrimeIsotopeComposition(
        delta18_prime_permil=23.600,
        cap_delta17_prime_permil=-0.432,
        source=(
            "Barkan and Luz (2011) delta18-prime as quoted by Young et al. "
            "(2014); Pack (2021) Delta-prime-17O"
        ),
    )
    co2 = PrimeIsotopeComposition.from_delta18_and_cap_delta17(
        delta18_permil=41.78933333333333,
        cap_delta17_prime_permil=-0.2186969696969697,
        source=(
            "Adnew et al. (2025) Supplement Table S2 mean for n=33 CARIBIC "
            "upper-tropospheric samples selected by N2O >= 313.5 ppb; "
            "delta18O=41.7893 per mil VSMOW converted to delta18-prime; "
            "Delta-prime-17O=-0.218697 per mil, lambda_ref=0.528"
        ),
    )
    return o2, co2
