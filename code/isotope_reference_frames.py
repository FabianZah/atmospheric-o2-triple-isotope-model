"""Exact conversions between absolute and modern-air oxygen isotope frames.

The updated forward model reports logarithmic isotope compositions relative
to VSMOW with lambda=0.528. Banerjee et al. (2026) report Delta-17O of O2
relative to modern air with lambda=0.518 and in ppm. These coordinates must be
converted before model-data comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


MODEL_LAMBDA = 0.528
BANERJEE_LAMBDA = 0.518


@dataclass(frozen=True)
class PrimeIsotopeComposition:
    """Logarithmic oxygen isotope composition in per mil."""

    delta18_prime_permil: float
    cap_delta17_prime_permil: float
    lambda_ref: float = MODEL_LAMBDA

    def __post_init__(self) -> None:
        values = (
            self.delta18_prime_permil,
            self.cap_delta17_prime_permil,
            self.lambda_ref,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("isotope composition values must be finite")


def convert_cap_delta17_slope(
    cap_delta17_prime_permil: float,
    delta18_prime_permil: float,
    *,
    from_lambda: float,
    to_lambda: float,
) -> float:
    """Convert Delta-prime-17O between reference slopes at fixed isotope ratios."""

    values = (
        cap_delta17_prime_permil,
        delta18_prime_permil,
        from_lambda,
        to_lambda,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("conversion values must be finite")
    return cap_delta17_prime_permil + (from_lambda - to_lambda) * delta18_prime_permil


def relative_cap_delta17_ppm(
    sample: PrimeIsotopeComposition,
    reference: PrimeIsotopeComposition,
    *,
    relative_lambda: float = BANERJEE_LAMBDA,
) -> float:
    """Return sample Delta-17O relative to reference air, in ppm.

    Both input compositions may use any explicit absolute lambda convention.
    They are first converted to the sample convention and then differenced in
    the requested modern-air-relative coordinate.
    """

    if not isfinite(relative_lambda):
        raise ValueError("relative_lambda must be finite")
    reference_cap_in_sample_frame = convert_cap_delta17_slope(
        reference.cap_delta17_prime_permil,
        reference.delta18_prime_permil,
        from_lambda=reference.lambda_ref,
        to_lambda=sample.lambda_ref,
    )
    relative_permil = (
        sample.cap_delta17_prime_permil
        - reference_cap_in_sample_frame
        + (sample.lambda_ref - relative_lambda)
        * (sample.delta18_prime_permil - reference.delta18_prime_permil)
    )
    return 1000.0 * relative_permil


def absolute_cap_delta17_from_relative_ppm(
    relative_cap_delta17_ppm_value: float,
    *,
    sample_delta18_prime_permil: float,
    reference: PrimeIsotopeComposition,
    absolute_lambda: float = MODEL_LAMBDA,
    relative_lambda: float = BANERJEE_LAMBDA,
) -> float:
    """Recover absolute Delta-prime-17O from a modern-air-relative value.

    The sample delta-prime-18O is required. A modern-air-relative Delta-17O
    value alone does not uniquely define an absolute isotope composition when
    the relative and absolute reference slopes differ.
    """

    values = (
        relative_cap_delta17_ppm_value,
        sample_delta18_prime_permil,
        absolute_lambda,
        relative_lambda,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("conversion values must be finite")
    reference_cap = convert_cap_delta17_slope(
        reference.cap_delta17_prime_permil,
        reference.delta18_prime_permil,
        from_lambda=reference.lambda_ref,
        to_lambda=absolute_lambda,
    )
    return (
        reference_cap
        + relative_cap_delta17_ppm_value / 1000.0
        + (relative_lambda - absolute_lambda)
        * (sample_delta18_prime_permil - reference.delta18_prime_permil)
    )
