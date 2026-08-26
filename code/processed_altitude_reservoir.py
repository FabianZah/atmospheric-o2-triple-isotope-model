"""Diagnostic high-altitude processed CO2 reservoir utilities.

This module is deliberately small and diagnostic. It does not alter the O2
isotope ODE solution. It represents an additional high-altitude CO2 isotope
component that can be mixed into the reported/observed stratospheric and export
CO2 signatures.

The purpose is to make the altitude-history hypothesis explicit and
reproducible before adding a full ODE column/reservoir.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

from isotopes import R17_VSMOW, R18_VSMOW, cap_delta17_from_primes
from young_model_inventory import TABLE3_TARGETS


def finite_exposure_activation(residence_time_years: float, exchange_lifetime_years: float) -> float:
    """Return the reacted fraction for first-order finite O(1D)-CO2 exposure."""

    if residence_time_years < 0.0 or exchange_lifetime_years <= 0.0:
        raise ValueError("residence time must be non-negative and exchange lifetime must be positive")
    return 1.0 - exp(-residence_time_years / exchange_lifetime_years)


@dataclass(frozen=True)
class ColumnProcessedFraction:
    """Column-weighted high-altitude processed CO2 contribution.

    Young states that the 10-25 km stratospheric column is about 0.5e25 cm^-2
    and the 25-40 km column is about 0.05e25 cm^-2. This helper turns that
    printed column split into a transparent processed/lower export fraction.
    """

    lower_column_cm2: float = 0.5e25
    upper_column_cm2: float = 0.05e25
    upper_activation: float = 0.0

    @property
    def upper_column_fraction(self) -> float:
        total = self.lower_column_cm2 + self.upper_column_cm2
        if self.lower_column_cm2 < 0.0 or self.upper_column_cm2 < 0.0 or total <= 0.0:
            raise ValueError("column densities must be non-negative with positive total")
        return self.upper_column_cm2 / total

    @property
    def processed_fraction(self) -> float:
        if not 0.0 <= self.upper_activation <= 1.0:
            raise ValueError("upper activation must be between 0 and 1")
        return self.upper_column_fraction * self.upper_activation

    def activation_for_processed_fraction(self, processed_fraction: float) -> float:
        if processed_fraction < 0.0:
            raise ValueError("processed fraction must be non-negative")
        return processed_fraction / self.upper_column_fraction

    def additional_major_mixed_d17o(
        self,
        base_d17o_permil: float,
        processed_d17o_permil: float,
    ) -> float:
        """Return mixed Delta'17O if processed CO2 adds major export flux."""

        fraction = self.processed_fraction
        return (base_d17o_permil + fraction * processed_d17o_permil) / (1.0 + fraction)

    def fixed_reported_mixed_d17o(
        self,
        base_d17o_permil: float,
        processed_d17o_permil: float,
    ) -> float:
        """Return reported Delta'17O if processed CO2 adds isotope numerator only."""

        return base_d17o_permil + self.processed_fraction * processed_d17o_permil

    def additional_major_anomaly_flux(
        self,
        lower_major_flux_mol_per_year: float,
        base_d17o_permil: float,
        processed_d17o_permil: float,
    ) -> float:
        """Return anomaly flux for lower export plus processed parallel export."""

        fraction = self.processed_fraction
        return lower_major_flux_mol_per_year * (base_d17o_permil + fraction * processed_d17o_permil)


@dataclass(frozen=True)
class ProcessedSignatureSource:
    """Diagnostic O(1D)-derived source for a high-altitude CO2 component.

    This is not yet the branch-level R7 isotope ODE. It records an explicit
    assumption for how much of an O(1D) anomaly is transferred into a processed
    CO2 component so audits can distinguish calculated signatures from imposed
    signatures.
    """

    o1d_d17o_permil: float
    activation: float
    transfer_efficiency: float = 1.0
    background_d17o_permil: float = 0.0
    mode: str = "inherit_o1d_delta"

    def processed_d17o_permil(self) -> float:
        """Return the processed CO2 Delta'17O value in per mil."""

        if not all(
            isfinite(value)
            for value in (
                self.o1d_d17o_permil,
                self.activation,
                self.transfer_efficiency,
                self.background_d17o_permil,
            )
        ):
            raise ValueError("processed signature inputs must be finite")
        if self.activation < 0.0 or self.transfer_efficiency < 0.0:
            raise ValueError("activation and transfer efficiency must be non-negative")

        transferred = self.activation * self.transfer_efficiency
        if self.mode == "inherit_o1d_delta":
            return transferred * self.o1d_d17o_permil
        if self.mode == "relax_to_o1d_delta":
            return self.background_d17o_permil + transferred * (
                self.o1d_d17o_permil - self.background_d17o_permil
            )
        raise ValueError(f"unknown processed signature mode: {self.mode}")


@dataclass(frozen=True)
class CO2IsotopeSignature:
    """Isotope-resolved CO2 signature in delta-prime notation."""

    delta17_prime_permil: float
    delta18_prime_permil: float

    @property
    def cap_delta17_permil(self) -> float:
        return cap_delta17_from_primes(self.delta17_prime_permil, self.delta18_prime_permil)

    @property
    def r17(self) -> float:
        return R17_VSMOW * exp(self.delta17_prime_permil / 1000.0)

    @property
    def r18(self) -> float:
        return R18_VSMOW * exp(self.delta18_prime_permil / 1000.0)


@dataclass(frozen=True)
class O1DIsotopeTransferSource:
    """Diagnostic isotope-resolved O(1D)-to-CO2 transfer source."""

    o1d_delta17_prime_permil: float
    o1d_delta18_prime_permil: float
    activation: float
    transfer_efficiency: float = 1.0
    background: CO2IsotopeSignature | None = None
    mode: str = "inherit_o1d_primes"

    def processed_signature(self) -> CO2IsotopeSignature:
        if self.activation < 0.0 or self.transfer_efficiency < 0.0:
            raise ValueError("activation and transfer efficiency must be non-negative")
        transferred = self.activation * self.transfer_efficiency
        if self.mode == "inherit_o1d_primes":
            return CO2IsotopeSignature(
                transferred * self.o1d_delta17_prime_permil,
                transferred * self.o1d_delta18_prime_permil,
            )
        if self.mode == "relax_to_o1d_primes":
            if self.background is None:
                raise ValueError("relax_to_o1d_primes requires a background signature")
            return CO2IsotopeSignature(
                self.background.delta17_prime_permil
                + transferred * (self.o1d_delta17_prime_permil - self.background.delta17_prime_permil),
                self.background.delta18_prime_permil
                + transferred * (self.o1d_delta18_prime_permil - self.background.delta18_prime_permil),
            )
        raise ValueError(f"unknown O1D transfer mode: {self.mode}")


def mix_co2_isotope_signatures(
    base: CO2IsotopeSignature,
    processed: CO2IsotopeSignature,
    fraction: float,
) -> CO2IsotopeSignature:
    """Mix two CO2 signatures by molecule fraction and return d-prime values."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    r17 = (1.0 - fraction) * base.r17 + fraction * processed.r17
    r18 = (1.0 - fraction) * base.r18 + fraction * processed.r18
    return CO2IsotopeSignature(
        1000.0 * log(r17 / R17_VSMOW),
        1000.0 * log(r18 / R18_VSMOW),
    )


def required_fraction_isotope_resolved(
    base: CO2IsotopeSignature,
    processed: CO2IsotopeSignature,
    target_d17o_permil: float,
    *,
    tolerance: float = 1.0e-10,
    max_iter: int = 80,
) -> float:
    """Return molecule fraction needed to match target Delta'17O by bisection."""

    f0 = base.cap_delta17_permil - target_d17o_permil
    f1 = processed.cap_delta17_permil - target_d17o_permil
    if abs(f0) < tolerance:
        return 0.0
    if f0 * f1 > 0.0:
        raise ValueError("target is not bracketed by base and processed signatures")
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = mix_co2_isotope_signatures(base, processed, mid).cap_delta17_permil - target_d17o_permil
        if abs(fm) < tolerance:
            return mid
        if f0 * fm <= 0.0:
            hi = mid
        else:
            lo = mid
            f0 = fm
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ProcessedReservoirMix:
    """A diagnostic mixture of lower-box and high-altitude processed CO2."""

    processed_d17o_permil: float
    fraction: float

    def mix(self, base_d17o_permil: float) -> float:
        """Return the mixed Delta'17O value in per mil."""

        return (1.0 - self.fraction) * base_d17o_permil + self.fraction * self.processed_d17o_permil


def required_fraction(base_d17o_permil: float, processed_d17o_permil: float, target_d17o_permil: float) -> float:
    """Return processed fraction needed to move a base signature to target."""

    denominator = processed_d17o_permil - base_d17o_permil
    if abs(denominator) < 1.0e-12:
        raise ValueError("processed and base signatures are indistinguishable")
    return (target_d17o_permil - base_d17o_permil) / denominator


def joint_required_fraction(
    *,
    base_strat_d17o_permil: float,
    base_export_d17o_permil: float,
    base_flux_permil_mol_per_year: float,
    processed_d17o_permil: float,
) -> float:
    """Return fraction that jointly fits Young CO2_strat and CO2 flux.

    The two required fractions are nearly identical for the current lower-box
    candidate. Averaging them gives one diagnostic mixture fraction while still
    allowing the audit table to report each requirement separately.
    """

    flux_per_permil = base_flux_permil_mol_per_year / max(base_export_d17o_permil, 1.0e-30)
    flux_target_d17o = TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"] / flux_per_permil
    f_strat = required_fraction(
        base_strat_d17o_permil,
        processed_d17o_permil,
        TABLE3_TARGETS["D17_CO2_strat_permil"],
    )
    f_flux = required_fraction(base_export_d17o_permil, processed_d17o_permil, flux_target_d17o)
    return 0.5 * (f_strat + f_flux)


def apply_processed_reservoir_mix(
    outputs: dict,
    processed_d17o_permil: float,
    fraction: float,
) -> dict:
    """Return CO2 diagnostics with a processed altitude component mixed in."""

    mix = ProcessedReservoirMix(processed_d17o_permil, fraction)
    mixed_strat = mix.mix(float(outputs["CO2_strat_D17O_permil"]))
    mixed_export = mix.mix(float(outputs["CO2_export_D17O_permil"]))
    base_export = max(float(outputs["CO2_export_D17O_permil"]), 1.0e-30)
    mixed_flux = float(outputs["CO2_strat_D17O_flux_permil_mol_per_year"]) * mixed_export / base_export
    return {
        **outputs,
        "CO2_strat_D17O_permil_unmixed": outputs["CO2_strat_D17O_permil"],
        "CO2_export_D17O_permil_unmixed": outputs["CO2_export_D17O_permil"],
        "CO2_strat_D17O_flux_permil_mol_per_year_unmixed": outputs[
            "CO2_strat_D17O_flux_permil_mol_per_year"
        ],
        "CO2_strat_D17O_permil": mixed_strat,
        "CO2_export_D17O_permil": mixed_export,
        "CO2_strat_D17O_flux_permil_mol_per_year": mixed_flux,
        "processed_altitude_CO2_D17O_permil": processed_d17o_permil,
        "processed_altitude_fraction": fraction,
        "processed_altitude_signature_mode": "diagnostic_mixture",
    }
