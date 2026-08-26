"""Usable diagnostic interface for the Young model reconstruction.

This module collects the current model variants in one place. The variants are
named to keep paper-derived constants separate from diagnostic adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from isotopes import R17_VSMOW, R18_VSMOW
from model_runner import (
    IsotopeSummary,
    initialize_finite_geosphere_burial_state,
    isotope_summaries,
    scaled_table3_state,
)
from reactions import Reaction, derivative
from consistent_o2_budget import consistent_rp_o2
from solve_coupled_isotope_subsystem import solve_subsystem
from solve_fixed_reservoir_isotopes import SolveResult as FixedReservoirSolveResult
from solve_fixed_reservoir_isotopes import solve_fixed_reservoir_isotopes
from solve_full_27 import solve_full_27
from solve_full_atmosphere import SolveResult, solve_full_atmosphere
from table3_state import SPECIES_ORDER, TABLE3_MOLES
from young_model_inventory import PARAMETERS, TABLE3_TARGETS
from young_reactions import TABLE3_CO2_TROP, executable_reactions


R5_TABLE3_BALANCED_MOLES = 5.589807e18

CALIBRATION_FACTORS = {
    # Two-factor CO2 Delta17O diagnostic fit, with R7c fixed by the CO18O
    # R7/transport audit.
    "R7c": 0.98166954,
    "R7g": 1.02552547,
    "R8c": 1.00010952,
}

R7_SOURCE_BALANCE_FACTORS = {
    # Factors implied by the printed Table 3 stratospheric CO2 isotopologue
    # R7/transport balance. These are branch-specific convention diagnostics,
    # not fitted free parameters.
    "R7c": 0.98166954,
    "R7g": 1.02783670,
}

DEFAULT_R7_THROUGHPUT_FACTOR = 2.25
DEFAULT_R8C_FACTOR = 1.0
R7_TRANSFER_BRANCHES = frozenset({"R7c", "R7d", "R7g", "R7h"})
YOUNG_UPPER_COLUMN_FRACTION = 1.0 / 11.0
LIANG_EXCHANGE_TIME_MODERN_YR = 1.0e8 / (365.25 * 24 * 60 * 60)
MODERN_RP_O2 = PARAMETERS["k_respiration_per_year"] * 3.80e19
R7_TRANSFER_ENHANCEMENT_GROUPS = {
    "all_transfer": R7_TRANSFER_BRANCHES,
    "incoming_heavy": frozenset({"R7c", "R7g"}),
    "incoming_17": frozenset({"R7g"}),
    "outgoing_heavy": frozenset({"R7d", "R7h"}),
}

MODEL_VARIANTS = (
    "paper",
    "r7_balance_diagnostic",
    "r7_balance_throughput_diagnostic",
    "r7_throughput_diagnostic",
    "r7_yung_branching_diagnostic",
    "table3_diagnostic",
    "young_behavior_diagnostic",
)
SOLVE_MODES = ("fixed_reservoir", "young_steady", "full_atmosphere", "full_27")
CO2_SOURCE_ISOTOPE_MODES = ("smow", "printed_table3")
R5_MODES = (
    "variant_default",
    "paper",
    "whole_stratosphere",
    "table3_balanced",
    "table3-balanced",
    "young_behavior",
    "dynamic_air",
    "dynamic_air_calibrated",
    "dynamic_o2",
)


@dataclass(frozen=True)
class ModelConfig:
    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.4
    gpp_scale: float = 1.0
    rp_o2: float | None = None
    o2_budget_mode: str = "young_fixed_gpp"
    model_variant: str = "table3_diagnostic"
    solve_mode: str = "young_steady"
    staged: bool = True
    closure_mode: str = "modern"
    co2_photo_sink_mode: str = "o2_source_water"
    co2_photo_sink_factor: float = 1.0
    co2_closure_isotope_mode: str = "printed_table3"
    co2_source_isotope_mode: str = "smow"
    young_steady_co2_source_isotope_mode: str = "printed_table3"
    r8_biosphere_bookkeeping: bool = False
    r5_mode: str = "variant_default"
    r5_effective_moles: float | None = None
    r5_co2_partner_factor: float = 0.0
    r7_throughput_factor: float = DEFAULT_R7_THROUGHPUT_FACTOR
    r7c_factor: float = 1.0
    r7g_factor: float = 1.0
    r8c_factor: float = DEFAULT_R8C_FACTOR
    r8_rate_factor: float = 1.0
    a_mif: float | None = None
    co2_sink_factor: float = 1.0
    co2_ocean_infusion_factor: float = 1.0
    r7_transfer_damping_amplitude: float = 0.0
    r7_transfer_damping_exponent: float = 1.0
    r7_transfer_damping_reference_ppm: float = 30000.0
    r7_transfer_damping_gpp_power: float = 0.0
    r7_transfer_damping_min_efficiency: float = 0.02
    r7_transfer_damping_shape: str = "linear_clamp"
    r7_vertical_gate_half_ppm: float = 10000.0
    r7_vertical_gate_hill: float = 2.0
    r7_two_box_lower_half_ppm: float = 3000.0
    r7_two_box_upper_half_ppm: float = 100000.0
    r7_two_box_upper_source_fraction: float = 0.60
    r7_two_box_upper_export_efficiency: float = 0.75
    r7_two_box_normalize_modern: bool = True
    r7_transfer_enhancement_gain: float = 0.0
    r7_transfer_enhancement_statistical_fraction: float | None = None
    r7_transfer_enhancement_exchange_time_yr: float | None = None
    r7_transfer_enhancement_exposure_time_yr: float | None = None
    r7_transfer_enhancement_half_ppm: float = 300.0
    r7_transfer_enhancement_exposure_mode: str = "none"
    r7_transfer_enhancement_exposure_power: float = 1.0
    r7_transfer_enhancement_branch_group: str = "incoming_heavy"
    r7_transfer_enhancement_full_atmosphere_only: bool = False
    r7_finite_exposure_exchange_time_modern_yr: float | None = None
    r7_finite_exposure_residence_time_yr: float | None = None
    r7_finite_exposure_max_excess: float = 1.0 / 3.0
    r7_finite_exposure_branch_group: str = "incoming_heavy"
    r7_finite_exposure_full_atmosphere_only: bool = False
    r7_full_atmosphere_enhancement_statistical_fraction: float | None = None
    photo_o17_source_law: str = "source_water"
    processed_column_source_d17o_permil: float | None = None
    processed_column_transition_tau_yr: float = 0.005
    processed_column_recovery_tau_yr: float | None = None
    processed_column_gate_mode: str = "hill"
    processed_column_gate_half_ppm: float = 30000.0
    processed_column_gate_hill: float = 8.0
    processed_column_gpp_power: float = 2.0
    processed_column_low_gpp_signal_mode: str = "current_unbounded"
    processed_column_transition_activation: float = 0.0
    processed_column_shared_tail_activation: float = 0.0
    processed_column_shared_tail_po2_fraction: float = 0.0
    processed_column_shared_tail_po2_power: float = 1.0
    processed_column_low_gpp_tail_activation: float = 0.0
    processed_column_low_gpp_tail_po2_fraction: float = 0.0
    processed_column_low_gpp_tail_po2_power: float = 1.0
    processed_column_low_gpp_transition_activation: float = 0.0
    processed_column_low_gpp_low_po2_transition_activation: float = 0.0
    processed_column_low_gpp_recovery_activation: float = 0.0
    processed_column_low_gpp_recovery_po2_mode: str = "power"
    processed_column_low_gpp_recovery_po2_power: float = 0.0
    processed_column_low_gpp_recovery_po2_half_pal: float = 0.75
    processed_column_low_gpp_recovery_po2_hill: float = 8.0
    processed_access_reservoir_mode: str = "none"
    processed_access_reservoir_tau_yr: float = 30.0
    processed_access_reservoir_initial: float = 1.0
    processed_access_reservoir_half_pal: float = 0.85
    processed_access_reservoir_hill: float = 8.0
    co2_export_signature_mode: str = "bulk_stratosphere"
    co2_export_upper_survival_fraction: float = 1.0
    co2_export_coupled_transport: bool = False
    explicit_lower_box_mode: str = "none"
    explicit_lower_box_upper_mode: str = "static_1p00"
    explicit_lower_box_lower_to_trop_rate_per_year: float = 3.0
    explicit_lower_box_net_export_rate_per_year: float | None = None
    explicit_lower_box_lower_major_scale: float = 1.0
    explicit_lower_box_source_weight_mode: str = "none"
    explicit_lower_box_source_weight_boost: float = 0.0
    explicit_lower_box_source_weight_half_ppm: float = 1000.0
    explicit_lower_box_source_weight_hill: float = 4.0
    alpha_respiration_18: float | None = None
    beta_respiration_17: float | None = None
    evapotranspiration_alpha_18: float | None = None
    evapotranspiration_beta_17: float | None = None


@dataclass(frozen=True)
class ModelRun:
    config: ModelConfig
    y: np.ndarray
    full_result: SolveResult | FixedReservoirSolveResult
    staged_result: object | None
    summaries: dict[str, IsotopeSummary]
    max_atmosphere_residual_per_year: float


@dataclass(frozen=True)
class CO2DependentTransferReaction:
    """Wrap an R7 isotope-transfer reaction with state-dependent damping.

    This is diagnostic: it tests whether effective O(1D)->CO2 isotope transfer
    should saturate at high pCO2, especially when GPP is low. The factor is
    based on the live tropospheric CO2 reservoir so transient runs are not tied
    to a static initial pCO2.
    """

    base: Reaction
    amplitude: float
    exponent: float
    reference_ppm: float
    gpp_scale: float
    gpp_power: float
    min_efficiency: float
    shape: str
    two_box_lower_half_ppm: float = 3000.0
    two_box_upper_half_ppm: float = 100000.0
    two_box_upper_source_fraction: float = 0.60
    two_box_upper_export_efficiency: float = 0.75
    two_box_normalize_modern: bool = True
    vertical_gate_half_ppm: float = 10000.0
    vertical_gate_hill: float = 2.0

    @property
    def key(self) -> str:
        return self.base.key

    @property
    def reactants(self) -> Mapping[str, float]:
        return self.base.reactants

    @property
    def products(self) -> Mapping[str, float]:
        return self.base.products

    @property
    def rate_constant(self) -> float:
        return self.base.rate_constant

    @property
    def units(self) -> str:
        return self.base.units

    @property
    def note(self) -> str:
        return (
            self.base.note
            + f"; diagnostic CO2-dependent R7 transfer damping amp={self.amplitude:g}, "
            + f"exp={self.exponent:g}, ref={self.reference_ppm:g} ppm, gpp_power={self.gpp_power:g}, "
            + f"min_eff={self.min_efficiency:g}, shape={self.shape}"
        )

    def efficiency(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        pco2_ppm = 294.4 * float(y[index["CO2_trop"]]) / TABLE3_CO2_TROP
        reference_ppm = max(self.reference_ppm, 1.0e-12)
        gpp_term = (1.0 / max(self.gpp_scale, 1.0e-12)) ** self.gpp_power
        min_efficiency = max(min(self.min_efficiency, 1.0), 0.0)
        if self.shape == "two_box_export_proxy":
            def source_response(pco2: float, half_ppm: float) -> float:
                return max(pco2, 0.0) / (max(pco2, 0.0) + max(half_ppm, 1.0e-30))

            def raw_proxy(pco2: float, gpp_scale: float) -> float:
                f_upper = min(max(self.two_box_upper_source_fraction, 0.0), 1.0)
                lower_source = (1.0 - f_upper) * source_response(pco2, self.two_box_lower_half_ppm)
                upper_source = f_upper * source_response(pco2, self.two_box_upper_half_ppm)
                export_efficiency = self.two_box_upper_export_efficiency * max(gpp_scale, 1.0e-12) ** self.gpp_power
                total = lower_source + upper_source
                if total <= 0.0:
                    return 1.0
                return (lower_source + export_efficiency * upper_source) / total

            efficiency = raw_proxy(pco2_ppm, self.gpp_scale)
            if self.two_box_normalize_modern:
                efficiency /= max(raw_proxy(294.4, 1.0), 1.0e-30)
            return max(min_efficiency, efficiency)
        if self.shape == "o1d_competition":
            # Smooth proxy for O(1D) loss competition. The reference value is
            # used as the half-saturation pCO2 implied by the explicit O(1D)
            # loss audit, avoiding the hard high-pCO2 floor of linear_clamp.
            co2_fraction = max(pco2_ppm, 0.0) / (reference_ppm + max(pco2_ppm, 0.0))
            modern_fraction = 294.4 / (reference_ppm + 294.4)
            co2_term = max((co2_fraction - modern_fraction) / max(1.0 - modern_fraction, 1.0e-12), 0.0)
            damping = max(self.amplitude * co2_term * gpp_term, 0.0)
            return min_efficiency + (1.0 - min_efficiency) / (1.0 + damping)
        if self.shape == "exchange_saturation":
            # Finite-exposure proxy for the vertical transport problem noted by
            # Yung/Liang: CO2 approaches photochemical isotope equilibrium over
            # an exchange time, so effective transfer saturates when pCO2 makes
            # exchange fast compared with the air-parcel exposure time.
            exposure = max(self.amplitude, 0.0) * (max(pco2_ppm, 0.0) / reference_ppm) ** self.exponent
            exposure *= gpp_term
            if exposure < 1.0e-8:
                efficiency = 1.0 - 0.5 * exposure
            else:
                efficiency = -float(np.expm1(-exposure)) / exposure
            return max(min_efficiency, efficiency)
        if self.shape == "vertical_gate_exchange_saturation":
            # A process-oriented alternative to a naked high CO2 exponent:
            # keep the collision/exchange term near linear in pCO2, but gate it
            # by the fraction of air/export that reaches the high-altitude
            # photochemical exposure region. The gate is diagnostic until a
            # resolved vertical ODE module replaces it.
            pco2 = max(pco2_ppm, 0.0)
            half = max(self.vertical_gate_half_ppm, 1.0e-30)
            hill = max(self.vertical_gate_hill, 1.0e-12)
            gate = pco2**hill / (pco2**hill + half**hill)
            exposure = max(self.amplitude, 0.0) * (pco2 / reference_ppm) ** self.exponent
            exposure *= gate * gpp_term
            if exposure < 1.0e-8:
                efficiency = 1.0 - 0.5 * exposure
            else:
                efficiency = -float(np.expm1(-exposure)) / exposure
            return max(min_efficiency, efficiency)
        if self.shape == "threshold_vertical_access":
            # Source/export access inferred from the inverse Fig. 8 lower-box
            # audit: R7 transfer remains near full strength through mid-pCO2,
            # then loses an export-fed high-altitude component at the very
            # high-pCO2 tail, more strongly at low GPP.
            pco2 = max(pco2_ppm, 0.0)
            half = max(self.vertical_gate_half_ppm, 1.0e-30)
            hill = max(self.vertical_gate_hill, 1.0e-12)
            gate = pco2**hill / (pco2**hill + half**hill)
            suppression = max(self.amplitude, 0.0) * gate * gpp_term
            return max(min_efficiency, 1.0 - suppression)
        co2_term = (max(pco2_ppm, 0.0) / reference_ppm) ** self.exponent
        damping = max(self.amplitude * co2_term * gpp_term, 0.0)
        if self.shape == "smooth_saturating":
            return min_efficiency + (1.0 - min_efficiency) / (1.0 + damping)
        return max(min_efficiency, 1.0 - damping)

    def rate(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        return self.base.rate(y, index) * self.efficiency(y, index)

    def apply(self, dydt: np.ndarray, rate: float, index: Mapping[str, int]) -> None:
        self.base.apply(dydt, rate, index)


@dataclass(frozen=True)
class CO2O1DEnhancedTransferReaction:
    """Wrap selected R7 isotope-transfer reactions with a CO2-state enhancement.

    This is a diagnostic proxy for transient O(1D)->CO2 isotope-transfer
    branching/exposure effects. A zero gain leaves the original Young-rate
    reconstruction unchanged.
    """

    base: Reaction
    gain: float
    half_ppm: float
    exposure_mode: str
    exposure_power: float
    branch_group: str

    @property
    def key(self) -> str:
        return self.base.key

    @property
    def reactants(self) -> Mapping[str, float]:
        return self.base.reactants

    @property
    def products(self) -> Mapping[str, float]:
        return self.base.products

    @property
    def rate_constant(self) -> float:
        return self.base.rate_constant

    @property
    def units(self) -> str:
        return self.base.units

    @property
    def note(self) -> str:
        return (
            self.base.note
            + f"; diagnostic CO2/O1D R7 transfer enhancement gain={self.gain:g}, "
            + f"half={self.half_ppm:g} ppm, exposure={self.exposure_mode}, group={self.branch_group}"
        )

    def factor(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        pco2_ppm = 294.4 * float(y[index["CO2_trop"]]) / TABLE3_CO2_TROP
        anomaly = max(pco2_ppm - 294.4, 0.0)
        co2_signal = anomaly / (self.half_ppm + anomaly)
        if self.exposure_mode == "none":
            exposure_signal = 1.0
        elif self.exposure_mode == "o1d":
            exposure_signal = max(float(y[index["O1D_strat"]]) / TABLE3_MOLES["O1D_strat"], 0.0)
        elif self.exposure_mode == "co2_o1d":
            live = float(y[index["CO2_strat"]]) * float(y[index["O1D_strat"]])
            modern = TABLE3_MOLES["CO2_strat"] * TABLE3_MOLES["O1D_strat"]
            exposure_signal = max(live / modern, 0.0)
        else:
            raise ValueError(f"unknown R7 transfer enhancement exposure mode: {self.exposure_mode}")
        return 1.0 + self.gain * co2_signal * exposure_signal**self.exposure_power

    def rate(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        return self.base.rate(y, index) * self.factor(y, index)

    def apply(self, dydt: np.ndarray, rate: float, index: Mapping[str, int]) -> None:
        self.base.apply(dydt, rate, index)


@dataclass(frozen=True)
class FiniteExposureR7TransferReaction:
    """Wrap selected R7 branches with finite O(1D)-CO2 exposure activation.

    Young's Table 2 R7 isotope-transfer branches use 1/2 factors. Yung/Liang
    CO3* statistical transfer gives an incoming-heavy limit of 2/3, i.e. a
    maximum relative excess of 1/3. This wrapper normalizes the finite-exposure
    activation so modern pCO2 leaves the branch unchanged.
    """

    base: Reaction
    exchange_time_modern_yr: float
    residence_time_yr: float
    max_excess: float
    branch_group: str

    @property
    def key(self) -> str:
        return self.base.key

    @property
    def reactants(self) -> Mapping[str, float]:
        return self.base.reactants

    @property
    def products(self) -> Mapping[str, float]:
        return self.base.products

    @property
    def rate_constant(self) -> float:
        return self.base.rate_constant

    @property
    def units(self) -> str:
        return self.base.units

    @property
    def note(self) -> str:
        return (
            self.base.note
            + f"; finite-exposure R7 transfer exchange={self.exchange_time_modern_yr:g} yr, "
            + f"residence={self.residence_time_yr:g} yr, max_excess={self.max_excess:g}, "
            + f"group={self.branch_group}"
        )

    def activation(self, pco2_ppm: float) -> float:
        co2_ratio = max(pco2_ppm / 294.4, 1.0e-30)
        exchange_time = max(self.exchange_time_modern_yr, 1.0e-30) / co2_ratio
        return 1.0 - float(np.exp(-max(self.residence_time_yr, 0.0) / exchange_time))

    def factor(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        pco2_ppm = 294.4 * float(y[index["CO2_trop"]]) / TABLE3_CO2_TROP
        modern_activation = self.activation(294.4)
        live_activation = self.activation(pco2_ppm)
        denominator = max(1.0 - modern_activation, 1.0e-12)
        signal = max((live_activation - modern_activation) / denominator, 0.0)
        return 1.0 + self.max_excess * min(signal, 1.0)

    def rate(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        return self.base.rate(y, index) * self.factor(y, index)

    def apply(self, dydt: np.ndarray, rate: float, index: Mapping[str, int]) -> None:
        self.base.apply(dydt, rate, index)


@dataclass(frozen=True)
class ProcessedColumnPhotoO17Source:
    """Photosynthetic O17O source with a processed-column source shift.

    This promotes the current three-term Fig. 8 source/export law from an audit
    replacement into a normal model component. The shift is applied to the
    effective source-water delta-prime 17O signature while leaving the major O2
    and O18O photosynthetic sources untouched.
    """

    major_rate: float
    source_d18_prime_permil: float
    source_beta17: float
    processed_d17o_permil: float
    transition_tau_yr: float
    recovery_tau_yr: float
    gate_mode: str
    gate_half_ppm: float
    gate_hill: float
    gpp_power: float
    low_gpp_signal_mode: str
    transition_activation: float
    shared_tail_activation: float
    shared_tail_po2_fraction: float
    shared_tail_po2_power: float
    low_gpp_tail_activation: float
    low_gpp_tail_po2_fraction: float
    low_gpp_tail_po2_power: float
    low_gpp_transition_activation: float
    low_gpp_low_po2_transition_activation: float
    low_gpp_recovery_activation: float
    low_gpp_recovery_po2_mode: str
    low_gpp_recovery_po2_power: float
    low_gpp_recovery_po2_half_pal: float
    low_gpp_recovery_po2_hill: float

    @property
    def key(self) -> str:
        return "photo_O17O_three_term_processed_column"

    @property
    def reactants(self) -> Mapping[str, float]:
        return {}

    @property
    def products(self) -> Mapping[str, float]:
        return {"O17O_trop": 1.0}

    @property
    def rate_constant(self) -> float:
        return self.major_rate

    @property
    def units(self) -> str:
        return "mol yr^-1"

    @property
    def note(self) -> str:
        return (
            "three-term processed-column photosynthetic O17O source law "
            f"processed_D17O={self.processed_d17o_permil:g} per mil"
        )

    def finite_activation(self, pco2_ppm: float) -> float:
        co2_ratio = max(pco2_ppm / 294.4, 1.0e-30)
        exchange_time = LIANG_EXCHANGE_TIME_MODERN_YR / co2_ratio
        return 1.0 - float(np.exp(-max(self.transition_tau_yr, 0.0) / max(exchange_time, 1.0e-30)))

    def finite_activation_for_tau(self, pco2_ppm: float, tau_yr: float) -> float:
        co2_ratio = max(pco2_ppm / 294.4, 1.0e-30)
        exchange_time = LIANG_EXCHANGE_TIME_MODERN_YR / co2_ratio
        return 1.0 - float(np.exp(-max(tau_yr, 0.0) / max(exchange_time, 1.0e-30)))

    def transition_shape(self, pco2_ppm: float) -> float:
        activation = self.finite_activation(pco2_ppm)
        return 4.0 * activation * (1.0 - activation)

    def recovery_shape(self, pco2_ppm: float) -> float:
        activation = self.finite_activation_for_tau(pco2_ppm, self.recovery_tau_yr)
        return 4.0 * activation * (1.0 - activation)

    def normalized_gate(self, pco2_ppm: float) -> float:
        pco2 = max(pco2_ppm, 0.0)
        half = max(self.gate_half_ppm, 1.0e-30)
        if self.gate_mode == "hill":
            hill = max(self.gate_hill, 1.0e-12)
            modern = 294.4**hill / (294.4**hill + half**hill)
            live = pco2**hill / (pco2**hill + half**hill)
        elif self.gate_mode == "finite_exposure":
            # Smooth exposure probability with 50% activation at gate_half_ppm.
            # This keeps the same interpretable half-point as the Hill gate, but
            # removes the sharp high-pCO2 step that can create contour waviness.
            coefficient = np.log(2.0) / half
            modern = 1.0 - float(np.exp(-coefficient * 294.4))
            live = 1.0 - float(np.exp(-coefficient * pco2))
        else:
            raise ValueError(f"unknown processed-column gate mode: {self.gate_mode}")
        return max((live - modern) / max(1.0 - modern, 1.0e-12), 0.0)

    def low_gpp_signal(self) -> float:
        gpp_scale = max(self.major_rate / MODERN_RP_O2, 1.0e-12)
        raw = max((1.0 / gpp_scale) ** self.gpp_power - 1.0, 0.0)
        if self.low_gpp_signal_mode == "current_unbounded":
            return raw
        if self.low_gpp_signal_mode == "log1p":
            return float(np.log1p(raw))
        if self.low_gpp_signal_mode == "sqrt":
            return float(np.sqrt(raw))
        if self.low_gpp_signal_mode == "bounded_ratio":
            return raw / (1.0 + raw)
        if self.low_gpp_signal_mode == "linear_deficit":
            return max(1.0 - gpp_scale, 0.0)
        raise ValueError(f"unknown low-GPP source signal mode: {self.low_gpp_signal_mode}")

    def po2_recovery_gate(self, po2_pal: float) -> float:
        if self.low_gpp_recovery_po2_mode == "power":
            return po2_pal**self.low_gpp_recovery_po2_power
        if self.low_gpp_recovery_po2_mode == "threshold_access":
            half = max(self.low_gpp_recovery_po2_half_pal, 1.0e-30)
            hill = max(self.low_gpp_recovery_po2_hill, 1.0e-30)

            def sigmoid(value: float) -> float:
                x = max(value, 1.0e-30) / half
                return x**hill / (1.0 + x**hill)

            modern = max(sigmoid(1.0), 1.0e-30)
            return min(sigmoid(po2_pal) / modern, 1.0)
        raise ValueError(f"unknown low-GPP recovery pO2 gate: {self.low_gpp_recovery_po2_mode}")

    def source_shift(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        pco2_ppm = 294.4 * float(y[index["CO2_trop"]]) / TABLE3_CO2_TROP
        po2_pal = max(float(y[index["O2_trop"]]) / TABLE3_MOLES["O2_trop"], 1.0e-30)
        source_d17o = (self.source_beta17 - 0.528) * self.source_d18_prime_permil
        contrast = self.processed_d17o_permil - source_d17o
        transition = -contrast * YOUNG_UPPER_COLUMN_FRACTION * self.transition_shape(pco2_ppm)
        shared_tail = contrast * YOUNG_UPPER_COLUMN_FRACTION * self.normalized_gate(pco2_ppm)
        low_transition = transition * self.low_gpp_signal()
        low_tail = shared_tail * self.low_gpp_signal()
        if self.low_gpp_recovery_po2_mode == "access_state":
            po2_gate = min(max(float(y[index["processed_access_state"]]), 0.0), 1.0)
        else:
            po2_gate = self.po2_recovery_gate(po2_pal)
        shared_gate = (1.0 - self.shared_tail_po2_fraction) + self.shared_tail_po2_fraction * (
            po2_gate ** self.shared_tail_po2_power
        )
        tail_gate = (1.0 - self.low_gpp_tail_po2_fraction) + self.low_gpp_tail_po2_fraction * (
            po2_gate ** self.low_gpp_tail_po2_power
        )
        low_recovery = contrast * YOUNG_UPPER_COLUMN_FRACTION * self.recovery_shape(pco2_ppm) * self.low_gpp_signal() * po2_gate
        low_po2_transition = transition * self.low_gpp_signal() * (1.0 - po2_gate)
        return (
            self.transition_activation * transition
            + self.shared_tail_activation * shared_tail * shared_gate
            + self.low_gpp_transition_activation * low_transition
            + self.low_gpp_low_po2_transition_activation * low_po2_transition
            + self.low_gpp_tail_activation * low_tail * tail_gate
            + self.low_gpp_recovery_activation * low_recovery
        )

    def apply_state(self, dydt: np.ndarray, y: np.ndarray, index: Mapping[str, int]) -> None:
        source_d17_prime = self.source_beta17 * self.source_d18_prime_permil + self.source_shift(y, index)
        source_r17 = R17_VSMOW * np.exp(source_d17_prime / 1000.0)
        dydt[index["O17O_trop"]] += self.major_rate * 2.0 * source_r17


@dataclass(frozen=True)
class CO2LowerBoxExportTransportReaction:
    """Hidden-lower-box approximation for CO2 stratosphere-to-troposphere export.

    The Young one-box model exports the bulk stratospheric CO2 isotope
    composition directly. This diagnostic wrapper removes CO2 from the
    stratospheric isotopologue reservoirs with the normal first-order transport
    rate, but adds it to the troposphere with a diluted lower-box isotope
    composition. It conserves total CO2 carbon in the explicit reservoirs, but
    isotope conservation is delegated to the hidden lower box. Use only as a
    coupled-shape experiment until a full extra-box ODE state is implemented.
    """

    k_st: float
    upper_survival_fraction: float

    @property
    def key(self) -> str:
        return "k_ST_CO2_lower_box_export"

    @property
    def reactants(self) -> Mapping[str, float]:
        return {"CO2_strat": 1.0}

    @property
    def products(self) -> Mapping[str, float]:
        return {"CO2_trop": 1.0}

    @property
    def rate_constant(self) -> float:
        return self.k_st

    @property
    def units(self) -> str:
        return "yr^-1"

    @property
    def note(self) -> str:
        return f"diagnostic lower-box CO2 export transport f_upper={self.upper_survival_fraction:g}"

    def rate(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        total = (
            float(y[index["CO2_strat"]])
            + float(y[index["CO18O_strat"]])
            + float(y[index["CO17O_strat"]])
        )
        return self.k_st * total

    @staticmethod
    def _prime_delta(heavy: float, major: float, reference_ratio: float) -> float:
        ratio = max(heavy, 1.0e-300) / max(major, 1.0e-300)
        return 1000.0 * float(np.log(max(ratio / reference_ratio, 1.0e-300)))

    def apply(self, dydt: np.ndarray, rate: float, index: Mapping[str, int]) -> None:
        raise RuntimeError("CO2LowerBoxExportTransportReaction requires derivative(... ) apply_state support")

    def apply_state(self, dydt: np.ndarray, y: np.ndarray, index: Mapping[str, int]) -> None:
        strat_major = float(y[index["CO2_strat"]])
        strat_18 = float(y[index["CO18O_strat"]])
        strat_17 = float(y[index["CO17O_strat"]])
        trop_major = float(y[index["CO2_trop"]])
        trop_18 = float(y[index["CO18O_trop"]])
        trop_17 = float(y[index["CO17O_trop"]])

        removed_major = self.k_st * strat_major
        removed_18 = self.k_st * strat_18
        removed_17 = self.k_st * strat_17
        total_export = removed_major + removed_18 + removed_17

        f_upper = self.upper_survival_fraction
        strat_d18p = self._prime_delta(strat_18, strat_major, R18_VSMOW)
        strat_d17p = self._prime_delta(strat_17, strat_major, R17_VSMOW)
        trop_d18p = self._prime_delta(trop_18, trop_major, R18_VSMOW)
        trop_d17p = self._prime_delta(trop_17, trop_major, R17_VSMOW)
        export_d18p = trop_d18p + f_upper * (strat_d18p - trop_d18p)
        export_d17p = trop_d17p + f_upper * (strat_d17p - trop_d17p)
        export_r18 = R18_VSMOW * float(np.exp(export_d18p / 1000.0))
        export_r17 = R17_VSMOW * float(np.exp(export_d17p / 1000.0))
        export_major = total_export / (1.0 + export_r18 + export_r17)
        export_18 = export_major * export_r18
        export_17 = export_major * export_r17

        dydt[index["CO2_strat"]] -= removed_major
        dydt[index["CO18O_strat"]] -= removed_18
        dydt[index["CO17O_strat"]] -= removed_17
        dydt[index["CO2_trop"]] += export_major
        dydt[index["CO18O_trop"]] += export_18
        dydt[index["CO17O_trop"]] += export_17


def apply_model_variant(
    reactions: list[Reaction],
    model_variant: str,
    r7_throughput_factor: float = DEFAULT_R7_THROUGHPUT_FACTOR,
    r8c_factor: float = DEFAULT_R8C_FACTOR,
) -> list[Reaction]:
    if model_variant == "paper":
        return reactions
    if model_variant not in (
        "r7_balance_diagnostic",
        "r7_balance_throughput_diagnostic",
        "r7_throughput_diagnostic",
        "r7_yung_branching_diagnostic",
        "table3_diagnostic",
        "young_behavior_diagnostic",
    ):
        raise ValueError(f"unknown model variant: {model_variant}")

    adjusted = []
    yung_branching_factors = {
        # Yung et al. 1991 and Liang et al. 2007 use statistical CO3*
        # branching: incoming heavy O(1D) transfers into CO2 with 2/3
        # probability and a heavy O already in CO2 transfers back out with
        # 1/3 probability. The printed Young table as transcribed uses 1/2
        # for all R7 isotope branches, so these are diagnostic factors.
        "R7b": 2.0 / 3.0,
        "R7c": 4.0 / 3.0,
        "R7d": 2.0 / 3.0,
        "R7e": 4.0 / 3.0,
        "R7f": 2.0 / 3.0,
        "R7g": 4.0 / 3.0,
        "R7h": 2.0 / 3.0,
        "R7i": 4.0 / 3.0,
    }
    for reaction in reactions:
        if model_variant == "r7_throughput_diagnostic":
            factor = r7_throughput_factor if reaction.key.startswith("R7") else 1.0
        elif model_variant == "r7_yung_branching_diagnostic":
            factor = 1.0
            if reaction.key.startswith("R7"):
                factor *= r7_throughput_factor
            factor *= yung_branching_factors.get(reaction.key, 1.0)
        elif model_variant == "r7_balance_diagnostic":
            factor = R7_SOURCE_BALANCE_FACTORS.get(reaction.key, 1.0)
        elif model_variant == "r7_balance_throughput_diagnostic":
            factor = 1.0
            if reaction.key.startswith("R7"):
                factor *= r7_throughput_factor
            factor *= R7_SOURCE_BALANCE_FACTORS.get(reaction.key, 1.0)
        else:
            factor = CALIBRATION_FACTORS.get(reaction.key, 1.0)
        if reaction.key == "R8c":
            factor *= r8c_factor
        if factor == 1.0:
            adjusted.append(reaction)
        else:
            adjusted.append(
                replace(
                    reaction,
                    rate_constant=reaction.rate_constant * factor,
                    note=reaction.note + f"; diagnostic {model_variant} factor={factor:.8g}",
                )
            )
    return adjusted


def apply_r7_transfer_damping(
    reactions: list[Reaction],
    amplitude: float,
    exponent: float,
    reference_ppm: float,
    gpp_scale: float,
    gpp_power: float,
    min_efficiency: float,
    shape: str,
    vertical_gate_half_ppm: float = 10000.0,
    vertical_gate_hill: float = 2.0,
    two_box_lower_half_ppm: float = 3000.0,
    two_box_upper_half_ppm: float = 100000.0,
    two_box_upper_source_fraction: float = 0.60,
    two_box_upper_export_efficiency: float = 0.75,
    two_box_normalize_modern: bool = True,
) -> list[Reaction]:
    if amplitude == 0.0 and shape != "two_box_export_proxy":
        return reactions
    adjusted = []
    for reaction in reactions:
        if reaction.key in R7_TRANSFER_BRANCHES:
            adjusted.append(
                CO2DependentTransferReaction(
                    reaction,
                    amplitude=amplitude,
                    exponent=exponent,
                    reference_ppm=reference_ppm,
                    gpp_scale=gpp_scale,
                    gpp_power=gpp_power,
                    min_efficiency=min_efficiency,
                    shape=shape,
                    vertical_gate_half_ppm=vertical_gate_half_ppm,
                    vertical_gate_hill=vertical_gate_hill,
                    two_box_lower_half_ppm=two_box_lower_half_ppm,
                    two_box_upper_half_ppm=two_box_upper_half_ppm,
                    two_box_upper_source_fraction=two_box_upper_source_fraction,
                    two_box_upper_export_efficiency=two_box_upper_export_efficiency,
                    two_box_normalize_modern=two_box_normalize_modern,
                )
            )
        else:
            adjusted.append(reaction)
    return adjusted


def apply_r7_transfer_enhancement(
    reactions: list[Reaction],
    gain: float,
    half_ppm: float,
    exposure_mode: str,
    exposure_power: float,
    branch_group: str,
) -> list[Reaction]:
    if gain == 0.0:
        return reactions
    if branch_group not in R7_TRANSFER_ENHANCEMENT_GROUPS:
        raise ValueError(f"unknown R7 transfer enhancement branch group: {branch_group}")
    selected = R7_TRANSFER_ENHANCEMENT_GROUPS[branch_group]
    adjusted = []
    for reaction in reactions:
        if reaction.key in selected:
            adjusted.append(
                CO2O1DEnhancedTransferReaction(
                    reaction,
                    gain=gain,
                    half_ppm=half_ppm,
                    exposure_mode=exposure_mode,
                    exposure_power=exposure_power,
                    branch_group=branch_group,
                )
            )
        else:
            adjusted.append(reaction)
    return adjusted


def apply_r7_finite_exposure_transfer(
    reactions: list[Reaction],
    exchange_time_modern_yr: float | None,
    residence_time_yr: float | None,
    max_excess: float,
    branch_group: str,
) -> list[Reaction]:
    if exchange_time_modern_yr is None or residence_time_yr is None:
        return reactions
    if branch_group not in R7_TRANSFER_ENHANCEMENT_GROUPS:
        raise ValueError(f"unknown R7 finite exposure branch group: {branch_group}")
    selected = R7_TRANSFER_ENHANCEMENT_GROUPS[branch_group]
    adjusted = []
    for reaction in reactions:
        if reaction.key in selected:
            adjusted.append(
                FiniteExposureR7TransferReaction(
                    reaction,
                    exchange_time_modern_yr=exchange_time_modern_yr,
                    residence_time_yr=residence_time_yr,
                    max_excess=max_excess,
                    branch_group=branch_group,
                )
            )
        else:
            adjusted.append(reaction)
    return adjusted


def apply_r7_branch_factors(reactions: list[Reaction], r7c_factor: float, r7g_factor: float) -> list[Reaction]:
    factors = {"R7c": r7c_factor, "R7g": r7g_factor}
    if all(abs(value - 1.0) <= 1.0e-12 for value in factors.values()):
        return reactions
    adjusted = []
    for reaction in reactions:
        factor = factors.get(reaction.key, 1.0)
        if abs(factor - 1.0) <= 1.0e-12:
            adjusted.append(reaction)
        else:
            adjusted.append(
                replace(
                    reaction,
                    rate_constant=reaction.rate_constant * factor,
                    note=reaction.note + f"; diagnostic branch factor={factor:.8g}",
                )
            )
    return adjusted


def source_d18_prime_permil(config: ModelConfig) -> float:
    alpha18 = PARAMETERS["evapotranspiration_alpha_18"] if config.evapotranspiration_alpha_18 is None else config.evapotranspiration_alpha_18
    return 1000.0 * float(np.log(alpha18))


def source_beta17(config: ModelConfig) -> float:
    return PARAMETERS["evapotranspiration_beta_17"] if config.evapotranspiration_beta_17 is None else config.evapotranspiration_beta_17


def apply_photo_o17_source_law(reactions: list[Reaction], config: ModelConfig) -> list[Reaction]:
    if config.photo_o17_source_law in ("source_water", "none"):
        return reactions
    if config.photo_o17_source_law != "three_term_processed_column":
        raise ValueError(f"unknown photo O17O source law: {config.photo_o17_source_law}")
    photo_major = next((reaction.rate_constant for reaction in reactions if reaction.key == "photo_O2"), None)
    if photo_major is None:
        raise ValueError("three-term processed-column source law requires a photo_O2 reaction")
    processed_d17o = (
        TABLE3_TARGETS["D17_O1D_permil"]
        if config.processed_column_source_d17o_permil is None
        else config.processed_column_source_d17o_permil
    )
    source = ProcessedColumnPhotoO17Source(
        major_rate=photo_major,
        source_d18_prime_permil=source_d18_prime_permil(config),
        source_beta17=source_beta17(config),
        processed_d17o_permil=processed_d17o,
        transition_tau_yr=config.processed_column_transition_tau_yr,
        recovery_tau_yr=(
            config.processed_column_transition_tau_yr
            if config.processed_column_recovery_tau_yr is None
            else config.processed_column_recovery_tau_yr
        ),
        gate_mode=config.processed_column_gate_mode,
        gate_half_ppm=config.processed_column_gate_half_ppm,
        gate_hill=config.processed_column_gate_hill,
        gpp_power=config.processed_column_gpp_power,
        low_gpp_signal_mode=config.processed_column_low_gpp_signal_mode,
        transition_activation=config.processed_column_transition_activation,
        shared_tail_activation=config.processed_column_shared_tail_activation,
        shared_tail_po2_fraction=config.processed_column_shared_tail_po2_fraction,
        shared_tail_po2_power=config.processed_column_shared_tail_po2_power,
        low_gpp_tail_activation=config.processed_column_low_gpp_tail_activation,
        low_gpp_tail_po2_fraction=config.processed_column_low_gpp_tail_po2_fraction,
        low_gpp_tail_po2_power=config.processed_column_low_gpp_tail_po2_power,
        low_gpp_transition_activation=config.processed_column_low_gpp_transition_activation,
        low_gpp_low_po2_transition_activation=config.processed_column_low_gpp_low_po2_transition_activation,
        low_gpp_recovery_activation=config.processed_column_low_gpp_recovery_activation,
        low_gpp_recovery_po2_mode=config.processed_column_low_gpp_recovery_po2_mode,
        low_gpp_recovery_po2_power=config.processed_column_low_gpp_recovery_po2_power,
        low_gpp_recovery_po2_half_pal=config.processed_column_low_gpp_recovery_po2_half_pal,
        low_gpp_recovery_po2_hill=config.processed_column_low_gpp_recovery_po2_hill,
    )
    return [*[reaction for reaction in reactions if reaction.key != "photo_O17O"], source]


def r7_transfer_enhancement_gain(config: ModelConfig) -> float:
    statistical_fraction = config.r7_transfer_enhancement_statistical_fraction
    if statistical_fraction is None and (
        config.r7_transfer_enhancement_exchange_time_yr is not None
        and config.r7_transfer_enhancement_exposure_time_yr is not None
    ):
        exchange_time = max(config.r7_transfer_enhancement_exchange_time_yr, 1.0e-30)
        exposure_time = max(config.r7_transfer_enhancement_exposure_time_yr, 0.0)
        statistical_fraction = 1.0 - float(np.exp(-exposure_time / exchange_time))
    if statistical_fraction is None:
        return config.r7_transfer_enhancement_gain
    # Young's tabulated R7 isotope-transfer branches use 1/2 factors. The
    # statistical CO3* branching in Yung/Liang corresponds to 2/3 for incoming
    # heavy-O transfer, i.e. a relative factor (2/3)/(1/2) = 4/3. The excess
    # above Young's 1/2 bookkeeping is therefore 1/3.
    return statistical_fraction * (1.0 / 3.0)


def apply_co2_lower_box_export_transport(
    reactions: list[Reaction],
    upper_survival_fraction: float,
) -> list[Reaction]:
    """Replace one-box CO2 ST transport with a hidden-lower-box export wrapper."""

    removed = {"k_ST_CO2_strat", "k_ST_CO18O_strat", "k_ST_CO17O_strat"}
    adjusted = [reaction for reaction in reactions if reaction.key not in removed]
    adjusted.append(
        CO2LowerBoxExportTransportReaction(
            PARAMETERS["k_ST_per_year"],
            upper_survival_fraction,
        )
    )
    return adjusted


def build_reactions(config: ModelConfig) -> list[Reaction]:
    r5_collision_partner_mode = "fixed"
    if config.r5_effective_moles is not None:
        r5_m = config.r5_effective_moles
    else:
        r5_mode = config.r5_mode.replace("-", "_")
        if r5_mode == "variant_default":
            if config.model_variant == "paper":
                r5_m = None
            elif config.model_variant == "young_behavior_diagnostic":
                r5_m = R5_TABLE3_BALANCED_MOLES * 1.25
            else:
                r5_m = R5_TABLE3_BALANCED_MOLES
        elif r5_mode in ("paper", "whole_stratosphere"):
            r5_m = None
        elif r5_mode == "table3_balanced":
            r5_m = R5_TABLE3_BALANCED_MOLES
        elif r5_mode == "young_behavior":
            r5_m = R5_TABLE3_BALANCED_MOLES * 1.25
        elif r5_mode in ("dynamic_air", "dynamic_air_calibrated", "dynamic_o2"):
            r5_m = None
            r5_collision_partner_mode = r5_mode
        else:
            raise ValueError(f"unknown R5 mode: {config.r5_mode}")
    co2_source_override = None
    closure_mode = config.closure_mode
    if config.solve_mode == "young_steady":
        # Young describes varying pCO2 by changing the balance between volcanic
        # delivery and weathering uptake to obtain new steady-state [CO2].
        # With O2 fixed and rp/kr fixed, respiration/photosynthesis and
        # strat/trop transport cancel in the major C16O2 budget, so the
        # required source is the first-order weathering + ocean-infusion sink.
        co2_trop_mol = 5.29e16 * (config.p_co2_ppm * 1.0e-6) / 2.944e-4
        co2_source_override = (
            PARAMETERS["k_CO2_weathering_per_year"]
            + PARAMETERS["k_ocean_CO2_infusion_per_year"] * config.co2_ocean_infusion_factor
        ) * config.co2_sink_factor * co2_trop_mol
        if config.closure_mode in ("finite_geosphere_burial", "modern_finite_geosphere"):
            closure_mode = "finite_geosphere_burial"
        elif config.closure_mode in ("lasaga_burial", "modern_lasaga_burial"):
            closure_mode = "lasaga_burial"
        else:
            closure_mode = "organic_burial"
        co2_source_isotope_mode = config.young_steady_co2_source_isotope_mode
    else:
        co2_source_isotope_mode = config.co2_source_isotope_mode
    rp_o2 = config.rp_o2
    if rp_o2 is None:
        if config.o2_budget_mode == "young_fixed_gpp":
            rp_o2 = None
        elif config.o2_budget_mode == "target_p_o2_balance":
            rp_o2 = consistent_rp_o2(
                config.p_o2_pal,
                gpp_scale=config.gpp_scale,
                closure_mode=closure_mode,
            ).required_rp_o2
        else:
            raise ValueError(f"unknown o2_budget_mode: {config.o2_budget_mode}")
    reactions = executable_reactions(
        r5_collision_partner_moles=r5_m,
        r5_collision_partner_mode=r5_collision_partner_mode,
        r5_co2_partner_factor=config.r5_co2_partner_factor,
        gpp_scale=config.gpp_scale,
        rp_o2=rp_o2,
        closure_mode=closure_mode,
        co2_photo_sink_factor=config.co2_photo_sink_factor,
        co2_photo_sink_mode=config.co2_photo_sink_mode,
        co2_closure_isotope_mode=config.co2_closure_isotope_mode,
        co2_source_override_mol_per_year=co2_source_override,
        co2_source_isotope_mode=co2_source_isotope_mode,
        co2_sink_factor=config.co2_sink_factor,
        co2_ocean_infusion_factor=config.co2_ocean_infusion_factor,
        r8_biosphere_bookkeeping=config.r8_biosphere_bookkeeping,
        r8_rate_factor=config.r8_rate_factor,
        a_mif=config.a_mif,
        alpha_respiration_18=config.alpha_respiration_18,
        beta_respiration_17=config.beta_respiration_17,
        evapotranspiration_alpha_18=config.evapotranspiration_alpha_18,
        evapotranspiration_beta_17=config.evapotranspiration_beta_17,
    )
    reactions = apply_model_variant(
        reactions,
        config.model_variant,
        config.r7_throughput_factor,
        config.r8c_factor,
    )
    reactions = apply_photo_o17_source_law(reactions, config)
    reactions = apply_r7_branch_factors(reactions, config.r7c_factor, config.r7g_factor)
    reactions = apply_r7_transfer_damping(
        reactions,
        config.r7_transfer_damping_amplitude,
        config.r7_transfer_damping_exponent,
        config.r7_transfer_damping_reference_ppm,
        config.gpp_scale,
        config.r7_transfer_damping_gpp_power,
        config.r7_transfer_damping_min_efficiency,
        config.r7_transfer_damping_shape,
        config.r7_vertical_gate_half_ppm,
        config.r7_vertical_gate_hill,
        config.r7_two_box_lower_half_ppm,
        config.r7_two_box_upper_half_ppm,
        config.r7_two_box_upper_source_fraction,
        config.r7_two_box_upper_export_efficiency,
        config.r7_two_box_normalize_modern,
    )
    enhancement_gain = r7_transfer_enhancement_gain(config)
    if (not config.r7_transfer_enhancement_full_atmosphere_only) or config.solve_mode == "full_atmosphere":
        reactions = apply_r7_transfer_enhancement(
            reactions,
            enhancement_gain,
            config.r7_transfer_enhancement_half_ppm,
            config.r7_transfer_enhancement_exposure_mode,
            config.r7_transfer_enhancement_exposure_power,
            config.r7_transfer_enhancement_branch_group,
        )
    if (not config.r7_finite_exposure_full_atmosphere_only) or config.solve_mode == "full_atmosphere":
        reactions = apply_r7_finite_exposure_transfer(
            reactions,
            config.r7_finite_exposure_exchange_time_modern_yr,
            config.r7_finite_exposure_residence_time_yr,
            config.r7_finite_exposure_max_excess,
            config.r7_finite_exposure_branch_group,
        )
    if config.solve_mode == "full_atmosphere" and config.r7_full_atmosphere_enhancement_statistical_fraction is not None:
        reactions = apply_r7_transfer_enhancement(
            reactions,
            config.r7_full_atmosphere_enhancement_statistical_fraction / 3.0,
            config.r7_transfer_enhancement_half_ppm,
            config.r7_transfer_enhancement_exposure_mode,
            config.r7_transfer_enhancement_exposure_power,
            config.r7_transfer_enhancement_branch_group,
        )
    if config.co2_export_coupled_transport:
        reactions = apply_co2_lower_box_export_transport(
            reactions,
            config.co2_export_upper_survival_fraction,
        )
    return reactions


def max_atmosphere_residual_per_year(y: np.ndarray, reactions: list[Reaction]) -> float:
    dydt = derivative(y, reactions, SPECIES_ORDER)
    excluded = {"O_bio", "O18_bio", "O17_bio", "O_geo", "O18_geo", "O17_geo"}
    values = []
    for i, name in enumerate(SPECIES_ORDER):
        if name in excluded:
            continue
        values.append(abs(float(dydt[i])) / max(abs(float(y[i])), 1.0))
    return max(values)


def run_model(config: ModelConfig) -> ModelRun:
    y0 = scaled_table3_state(config.p_o2_pal, config.p_co2_ppm, isotope_mode="printed")
    if config.closure_mode in ("finite_geosphere_burial", "modern_finite_geosphere"):
        y0 = initialize_finite_geosphere_burial_state(y0)
    reactions = build_reactions(config)
    staged_result = None
    if config.solve_mode in ("fixed_reservoir", "young_steady"):
        full_result = solve_fixed_reservoir_isotopes(y0, reactions)
    elif config.solve_mode == "full_atmosphere":
        if config.staged:
            staged_result = solve_subsystem(y0, reactions)
            y0 = staged_result.y
        full_result = solve_full_atmosphere(y0, reactions)
    elif config.solve_mode == "full_27":
        if config.staged:
            staged_result = solve_subsystem(y0, reactions)
            y0 = staged_result.y
        full_result = solve_full_27(y0, reactions)
    else:
        raise ValueError(f"unknown solve mode: {config.solve_mode}")
    summaries = {summary.label: summary for summary in isotope_summaries(full_result.y)}
    return ModelRun(
        config=config,
        y=full_result.y,
        full_result=full_result,
        staged_result=staged_result,
        summaries=summaries,
        max_atmosphere_residual_per_year=max_atmosphere_residual_per_year(full_result.y, reactions),
    )


def run_summary_row(run: ModelRun) -> dict[str, float | str | bool]:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    return {
        "pO2_pal": run.config.p_o2_pal,
        "pCO2_ppm": run.config.p_co2_ppm,
        "gpp_scale": run.config.gpp_scale,
        "rp_o2": "" if run.config.rp_o2 is None else run.config.rp_o2,
        "o2_budget_mode": run.config.o2_budget_mode,
        "model_variant": run.config.model_variant,
        "solve_mode": run.config.solve_mode,
        "r5_mode": run.config.r5_mode,
        "r5_co2_partner_factor": run.config.r5_co2_partner_factor,
        "r7_throughput_factor": run.config.r7_throughput_factor,
        "r8c_factor": run.config.r8c_factor,
        "r8_rate_factor": run.config.r8_rate_factor,
        "a_mif": "" if run.config.a_mif is None else run.config.a_mif,
        "r8_biosphere_bookkeeping": run.config.r8_biosphere_bookkeeping,
        "co2_photo_sink_factor": run.config.co2_photo_sink_factor,
        "co2_ocean_infusion_factor": run.config.co2_ocean_infusion_factor,
        "converged": run.full_result.converged,
        "staged_converged": bool(run.staged_result.converged) if run.staged_result is not None else "",
        "residual_norm": run.full_result.residual_norm,
        "max_atm_rel_resid_per_yr": run.max_atmosphere_residual_per_year,
        "O2_trop_mol": run.y[idx["O2_trop"]],
        "CO2_trop_mol": run.y[idx["CO2_trop"]],
        "O2_trop_D17O_permil": run.summaries["O2_trop"].cap_delta17,
        "O2_trop_d18p_permil": run.summaries["O2_trop"].delta18_prime,
        "O2_trop_d17p_permil": run.summaries["O2_trop"].delta17_prime,
        "CO2_trop_D17O_permil": run.summaries["CO2_trop"].cap_delta17,
        "CO2_strat_D17O_permil": run.summaries["CO2_strat"].cap_delta17,
        "O3_strat_D17O_permil": run.summaries["O3_strat"].cap_delta17,
    }
