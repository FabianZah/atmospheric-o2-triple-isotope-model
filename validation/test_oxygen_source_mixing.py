from __future__ import annotations

import pytest

from oxygen_source_mixing import (
    OxygenIsotopeComposition,
    mix_oxygen_sources,
    recover_source_a,
)


def test_ratio_space_mixing_preserves_endmembers() -> None:
    atmospheric = OxygenIsotopeComposition.from_delta18_and_cap_delta17(23.9, -0.432)
    non_atmospheric = OxygenIsotopeComposition.from_delta18_and_cap_delta17(-8.0, 0.0)

    pure_atmospheric = mix_oxygen_sources(atmospheric, non_atmospheric, 1.0)
    pure_non_atmospheric = mix_oxygen_sources(atmospheric, non_atmospheric, 0.0)

    assert pure_atmospheric.delta18_permil == pytest.approx(atmospheric.delta18_permil)
    assert pure_atmospheric.cap_delta17_permil == pytest.approx(atmospheric.cap_delta17_permil)
    assert pure_non_atmospheric.delta18_permil == pytest.approx(non_atmospheric.delta18_permil)
    assert pure_non_atmospheric.cap_delta17_permil == pytest.approx(
        non_atmospheric.cap_delta17_permil
    )


@pytest.mark.parametrize("fraction", [0.05, 0.10, 0.25, 0.60, 0.95])
def test_known_endmember_is_recovered_after_forward_mixing(fraction: float) -> None:
    atmospheric = OxygenIsotopeComposition.from_delta18_and_cap_delta17(23.9, -0.432)
    non_atmospheric = OxygenIsotopeComposition.from_delta18_and_cap_delta17(-12.5, 0.035)

    mixture = mix_oxygen_sources(atmospheric, non_atmospheric, fraction)
    recovered = recover_source_a(mixture, non_atmospheric, fraction)

    assert recovered.delta18_permil == pytest.approx(atmospheric.delta18_permil, abs=1.0e-10)
    assert recovered.cap_delta17_permil == pytest.approx(
        atmospheric.cap_delta17_permil,
        abs=1.0e-10,
    )


def test_delta_prime_anomaly_is_not_averaged_directly() -> None:
    source_a = OxygenIsotopeComposition.from_delta18_and_cap_delta17(120.0, -8.0)
    source_b = OxygenIsotopeComposition.from_delta18_and_cap_delta17(-80.0, 2.0)
    fraction = 0.35

    mixture = mix_oxygen_sources(source_a, source_b, fraction)
    direct_delta_average = (
        fraction * source_a.cap_delta17_permil
        + (1.0 - fraction) * source_b.cap_delta17_permil
    )

    assert abs(mixture.cap_delta17_permil - direct_delta_average) > 0.01


def test_recovery_rejects_nonphysical_solution() -> None:
    mixture = OxygenIsotopeComposition.from_delta18_and_cap_delta17(-900.0, 0.0)
    source_b = OxygenIsotopeComposition.from_delta18_and_cap_delta17(500.0, 0.0)

    with pytest.raises(ValueError, match="ratios must be positive"):
        recover_source_a(mixture, source_b, 0.01)
