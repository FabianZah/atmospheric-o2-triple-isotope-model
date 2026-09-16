"""Observation-reference a mechanistic isotope differential without altering it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ObservationReferencedIsotope:
    raw_modern_permil: float
    raw_scenario_permil: float
    observed_reference_permil: float
    observed_reference_uncertainty_permil: float
    reference_source: str

    @property
    def mechanistic_differential_permil(self) -> float:
        return self.raw_scenario_permil - self.raw_modern_permil

    @property
    def observation_referenced_permil(self) -> float:
        return (
            self.observed_reference_permil
            + self.mechanistic_differential_permil
        )

    @property
    def structural_baseline_residual_permil(self) -> float:
        return self.raw_modern_permil - self.observed_reference_permil

    def as_dict(self) -> dict[str, float | str]:
        result = asdict(self)
        result.update(
            {
                "mechanistic_differential_permil": (
                    self.mechanistic_differential_permil
                ),
                "observation_referenced_permil": (
                    self.observation_referenced_permil
                ),
                "structural_baseline_residual_permil": (
                    self.structural_baseline_residual_permil
                ),
            }
        )
        return result


def observation_reference(
    *,
    raw_modern_permil: float,
    raw_scenario_permil: float,
    observed_reference_permil: float,
    observed_reference_uncertainty_permil: float,
    reference_source: str,
) -> ObservationReferencedIsotope:
    values = (
        raw_modern_permil,
        raw_scenario_permil,
        observed_reference_permil,
        observed_reference_uncertainty_permil,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("isotope reference values must be finite")
    if observed_reference_uncertainty_permil < 0.0:
        raise ValueError("observational uncertainty must be non-negative")
    if not reference_source.strip():
        raise ValueError("reference_source must be non-empty")
    return ObservationReferencedIsotope(
        raw_modern_permil=float(raw_modern_permil),
        raw_scenario_permil=float(raw_scenario_permil),
        observed_reference_permil=float(observed_reference_permil),
        observed_reference_uncertainty_permil=float(
            observed_reference_uncertainty_permil
        ),
        reference_source=reference_source.strip(),
    )
