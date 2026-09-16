"""Scenario interface for publication-oriented Young-model runs.

This module is the stable layer that scripts, notebooks, and a future UI should
call. The lower-level modules still contain diagnostics and reconstruction
experiments; this file turns the useful combinations into named, documented
presets with explicit metadata and warnings.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

from calibrated_model import ModelConfig, run_model
from consistent_o2_budget import O2BudgetTerms, consistent_rp_o2
from earth_history_envelopes import (
    PCO2_RECOMMENDED_MAX_PPM,
    pco2_envelope_status,
)
from gpp_normalization import (
    GPP_NORMALIZATIONS,
    modern_gpp_pgC_per_year,
    internal_young_gpp_scale,
    normalization_role,
    requested_gpp_pgC_per_year,
)
from phanerozoic_o2 import (
    MODEL_PO2_WORKING_MAX_PAL,
    MODEL_PO2_WORKING_MIN_PAL,
    in_working_po2_envelope,
    percent_equivalent,
)
from processed_altitude_reservoir import ColumnProcessedFraction
from r7_rate_sources import CURRENT_R7_THROUGHPUT_FACTOR
from r8_reference_sources import R8C_REFERENCE_RATIO_FACTOR
from table3_state import SPECIES_ORDER
from young_architecture_candidate import (
    CURRENT_ARCHITECTURE_NAME,
    STEADY_SOURCE_LAW,
    TRANSIENT_LAW,
    V0_STEADY_SOURCE_LAW,
    V1_STEADY_SOURCE_LAW,
    V2_STEADY_SOURCE_LAW,
    V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW,
    V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_STEADY_SOURCE_LAW,
    V2_LOG_LOW_GPP_STEADY_SOURCE_LAW,
    V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW,
    V2_TRANSIENT_LAW,
    YOUNG_LIKE_V0_NAME,
    YOUNG_LIKE_V1_NAME,
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME,
    YOUNG_LIKE_V2_PARALLEL_CO2_NAME,
    YOUNG_LIKE_V2_NAME,
)
from young_model_inventory import PARAMETERS, TABLE3_TARGETS
from conventions import (
    DEFAULT_CONVENTIONS,
    R7_DEFAULT,
    R7_O1D_CO2_RATE,
    conventions_metadata_for_parameters,
)


SCHEMA_VERSION = "scenario-v0.1"
CURRENT_YOUNG_PRINTED_PRESET = "young_printed_inputs"
CURRENT_YOUNG_REPRODUCTION_PRESET = "young_reproduction"
CURRENT_UPDATED_PHYSICAL_PRESET = "updated_physical"
PHYSICAL_EXTRAPOLATION_PRESET = "physical_extrapolation"
# Public default: the resolved-chemistry extrapolation branch. It reproduces
# Young Fig. 7/Fig. 8 in-domain and extrapolates physically across the
# literature pCO2/pO2/GPP envelope. Modern O2 Delta'17O is the raw chemistry
# value (~ -0.427 permil), within uncertainty of the Pack 2021 anchor
# (-0.432 permil); no modern output offset is applied. Use updated_physical
# when an exact modern Pack anchor or Young's exact 50% GPP / 30000 ppm Fig. 8
# endpoint is the priority.
CURRENT_PUBLICATION_PRESET = PHYSICAL_EXTRAPOLATION_PRESET

# --- Public model surface ------------------------------------------------
# The two presets exposed in the web UI and described in the manuscript. The
# default is the updated model (PHYSICAL_EXTRAPOLATION_PRESET); the second is the
# validated Young et al. (2014) reconstruction, offered only within the input
# domain Young actually explored (digitized Fig. 7/8 extent). Every other entry in
# PRESET_CONFIGS is a development/diagnostic scenario and is NOT part of the public
# surface; keep this pair in sync with the UI selector and the manuscript.
YOUNG_REPRODUCTION_PRESET = "young_reproduction"
PUBLIC_PRESETS = (PHYSICAL_EXTRAPOLATION_PRESET, YOUNG_REPRODUCTION_PRESET)
PUBLIC_PRESET_LABELS = {
    PHYSICAL_EXTRAPOLATION_PRESET: "Updated model (this study)",
    YOUNG_REPRODUCTION_PRESET: "Young et al. (2014) reconstruction",
}
# Plain-language, one-line descriptions for the public UI caption. These are the
# user-facing counterparts of the developer-oriented PRESET_ROLES entries; the UI
# prefers these and falls back to preset_role() for anything else.
PUBLIC_PRESET_ROLES = {
    PHYSICAL_EXTRAPOLATION_PRESET: (
        "Updated model (default): the Young et al. (2014) framework with recent literature "
        "constraints, applicable across the full pCO2-pO2-productivity domain. Modern O2 "
        "Delta'17O is within uncertainty of the Pack (2021) anchor; no output offset is applied."
    ),
    YOUNG_REPRODUCTION_PRESET: (
        "Young et al. (2014) reconstruction: reproduces the original published behavior "
        "(Table 3, Fig. 7/8) and is restricted to the input range Young validated. A benchmark "
        "for comparison, not for extrapolation."
    ),
}
# Validated input domain for the Young reconstruction (digitized Fig. 7/8 extent).
# Outside this range the reconstruction is not benchmarked, so the UI clamps to it.
YOUNG_REPRODUCTION_BOUNDS = {
    "pco2_ppm": (94.0, 29775.0),
    "gpp_percent": (52.0, 148.0),
    "po2_pal": (1.0, 1.0),
}

UPDATED_VALIDATED_YOUNG_LOCAL_PACK_PRESET = "updated_physical_from_validated_young_local_pack_anchor"
UPDATED_VALIDATED_YOUNG_BETA_PACK_PRESET = "updated_physical_from_validated_young_beta_respiration_pack_candidate"
UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET = "updated_physical_from_validated_young_water_beta_pack_candidate"
PACK2021_O2_D17O_TARGET_PERMIL = -0.432
PACK2021_BETA_RESPIRATION_17_CANDIDATE = 0.5140
PACK2021_EVAPOTRANSPIRATION_BETA_17_CANDIDATE = 0.5210
PACK2021_ANCHORED_A_MIF = 1.0664813672006126
PACK2021_YOUNG_REPRO_ANCHORED_A_MIF = 1.0638489508455442
ADNEW2025_YOUNG_REPRO_EXPORT_SURVIVAL_FRACTION = 0.4769201290476315
PACK2021_EXPLICIT_EXPORT_A_MIF = 1.0633162045828066
PACK2021_VALIDATED_YOUNG_EXPLICIT_EXPORT_A_MIF = 1.072548429398871
ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR = 0.9596991848253845
YOUNG_COLUMN_PROCESSED_UPPER_ACTIVATION = 0.30
O2_BUDGET_NEAR_BOUNDARY_RP_RELATIVE_TO_MODERN = 0.005


@dataclass(frozen=True)
class ScenarioInput:
    """High-level model request.

    `preset` controls the default model branch. Individual fields can override
    that preset when they are not None.
    """

    preset: str = CURRENT_PUBLICATION_PRESET
    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.4
    gpp_scale: float = 1.0
    gpp_normalization: str | None = None
    gpp_modern_pgC_per_year: float | None = None
    solve_mode: str | None = None
    closure_mode: str | None = None
    o2_budget_mode: str | None = None
    model_variant: str | None = None
    r7_throughput_factor: float | None = None
    r7c_factor: float | None = None
    r7g_factor: float | None = None
    r8c_factor: float | None = None
    r5_mode: str | None = None
    r5_co2_partner_factor: float | None = None
    co2_photo_sink_factor: float | None = None
    r8_rate_factor: float | None = None
    a_mif: float | None = None
    co2_source_isotope_mode: str | None = None
    young_steady_co2_source_isotope_mode: str | None = None
    co2_sink_factor: float | None = None
    co2_ocean_infusion_factor: float | None = None
    r7_transfer_damping_amplitude: float | None = None
    r7_transfer_damping_exponent: float | None = None
    r7_transfer_damping_reference_ppm: float | None = None
    r7_transfer_damping_gpp_power: float | None = None
    r7_transfer_damping_min_efficiency: float | None = None
    r7_transfer_damping_shape: str | None = None
    r7_vertical_gate_half_ppm: float | None = None
    r7_vertical_gate_hill: float | None = None
    r7_two_box_lower_half_ppm: float | None = None
    r7_two_box_upper_half_ppm: float | None = None
    r7_two_box_upper_source_fraction: float | None = None
    r7_two_box_upper_export_efficiency: float | None = None
    r7_two_box_normalize_modern: bool | None = None
    r7_transfer_enhancement_gain: float | None = None
    r7_transfer_enhancement_statistical_fraction: float | None = None
    r7_transfer_enhancement_exchange_time_yr: float | None = None
    r7_transfer_enhancement_exposure_time_yr: float | None = None
    r7_transfer_enhancement_half_ppm: float | None = None
    r7_transfer_enhancement_exposure_mode: str | None = None
    r7_transfer_enhancement_exposure_power: float | None = None
    r7_transfer_enhancement_branch_group: str | None = None
    r7_transfer_enhancement_full_atmosphere_only: bool | None = None
    r7_finite_exposure_exchange_time_modern_yr: float | None = None
    r7_finite_exposure_residence_time_yr: float | None = None
    r7_finite_exposure_max_excess: float | None = None
    r7_finite_exposure_branch_group: str | None = None
    r7_finite_exposure_full_atmosphere_only: bool | None = None
    r7_full_atmosphere_enhancement_statistical_fraction: float | None = None
    photo_o17_source_law: str | None = None
    processed_column_source_d17o_permil: float | None = None
    processed_column_transition_tau_yr: float | None = None
    processed_column_recovery_tau_yr: float | None = None
    processed_column_gate_mode: str | None = None
    processed_column_gate_half_ppm: float | None = None
    processed_column_gate_hill: float | None = None
    processed_column_gpp_power: float | None = None
    processed_column_low_gpp_signal_mode: str | None = None
    processed_column_transition_activation: float | None = None
    processed_column_shared_tail_activation: float | None = None
    processed_column_shared_tail_po2_fraction: float | None = None
    processed_column_shared_tail_po2_power: float | None = None
    processed_column_low_gpp_tail_activation: float | None = None
    processed_column_low_gpp_tail_po2_fraction: float | None = None
    processed_column_low_gpp_tail_po2_power: float | None = None
    processed_column_low_gpp_transition_activation: float | None = None
    processed_column_low_gpp_low_po2_transition_activation: float | None = None
    processed_column_low_gpp_recovery_activation: float | None = None
    processed_column_low_gpp_recovery_po2_mode: str | None = None
    processed_column_low_gpp_recovery_po2_power: float | None = None
    processed_column_low_gpp_recovery_po2_half_pal: float | None = None
    processed_column_low_gpp_recovery_po2_hill: float | None = None
    processed_access_reservoir_mode: str | None = None
    processed_access_reservoir_tau_yr: float | None = None
    processed_access_reservoir_initial: float | None = None
    processed_access_reservoir_half_pal: float | None = None
    processed_access_reservoir_hill: float | None = None
    co2_export_signature_mode: str | None = None
    co2_export_upper_survival_fraction: float | None = None
    co2_export_coupled_transport: bool | None = None
    explicit_lower_box_mode: str | None = None
    explicit_lower_box_upper_mode: str | None = None
    explicit_lower_box_lower_to_trop_rate_per_year: float | None = None
    explicit_lower_box_net_export_rate_per_year: float | None = None
    explicit_lower_box_lower_major_scale: float | None = None
    explicit_lower_box_source_weight_mode: str | None = None
    explicit_lower_box_source_weight_boost: float | None = None
    explicit_lower_box_source_weight_half_ppm: float | None = None
    explicit_lower_box_source_weight_hill: float | None = None
    alpha_respiration_18: float | None = None
    beta_respiration_17: float | None = None
    evapotranspiration_alpha_18: float | None = None
    evapotranspiration_beta_17: float | None = None
    o2_d17o_calibration_mode: str | None = None
    # Opt-in: also report the companion O2 Delta'17O estimate (gate off vs on)
    # so out-of-domain values carry an explicit physical/Young-tuned bound pair.
    report_extrapolation_bounds: bool = False


@dataclass(frozen=True)
class ScenarioResult:
    schema_version: str
    created_utc: str
    input: dict[str, Any]
    config: dict[str, Any]
    outputs: dict[str, Any]
    warnings: list[str]


PRESET_CONFIGS: dict[str, ModelConfig] = {
    "paper_reconstruction": ModelConfig(
        model_variant="paper",
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="young_fixed_gpp",
        young_steady_co2_source_isotope_mode="printed_table3",
    ),
    "young_best_fit": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="young_fixed_gpp",
        young_steady_co2_source_isotope_mode="smow",
    ),
    "young_best_fit_consistent_o2": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        young_steady_co2_source_isotope_mode="smow",
    ),
    "physical_o2_budget": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
    ),
    "physical_young_transient_co2": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_ocean_infusion_factor=0.5,
    ),
    "physical_slow_co2_sink_050_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.5,
    ),
    "physical_slow_co2_sink_070_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.7,
    ),
    "physical_o1d_transfer_efficiency_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        r7_transfer_damping_amplitude=0.2,
        r7_transfer_damping_exponent=3.0,
        r7_transfer_damping_gpp_power=1.0,
    ),
    "physical_combined_shape_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.7,
        r7_transfer_damping_amplitude=0.2,
        r7_transfer_damping_exponent=3.0,
        r7_transfer_damping_gpp_power=1.0,
    ),
    "physical_combined_dole_r8_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.85,
        r8_rate_factor=5.0,
        r7_transfer_damping_amplitude=0.2,
        r7_transfer_damping_exponent=3.0,
        r7_transfer_damping_gpp_power=1.0,
        alpha_respiration_18=1.0 / 1.016,
        beta_respiration_17=0.514,
        evapotranspiration_alpha_18=1.007304,
    ),
    "physical_combined_dole_shape_r8_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.000126,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.82,
        r8_rate_factor=12.0,
        r7_transfer_damping_amplitude=0.18,
        r7_transfer_damping_exponent=12.0,
        r7_transfer_damping_gpp_power=1.25,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_directional_r7_transfer_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.18,
        r7_transfer_damping_exponent=12.0,
        r7_transfer_damping_gpp_power=1.25,
        r7_transfer_enhancement_exchange_time_yr=1.0e8 / (365.25 * 24 * 60 * 60),
        r7_transfer_enhancement_exposure_time_yr=1.10,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=True,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_directional_r7_no_damping_candidate": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.0,
        r7_transfer_damping_exponent=12.0,
        r7_transfer_damping_gpp_power=1.25,
        r7_transfer_enhancement_exchange_time_yr=1.0e8 / (365.25 * 24 * 60 * 60),
        r7_transfer_enhancement_exposure_time_yr=1.10,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=True,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_directional_r7_o1d_competition_candidate": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.05,
        r7_transfer_damping_exponent=1.0,
        r7_transfer_damping_reference_ppm=1000000.0,
        r7_transfer_damping_gpp_power=0.25,
        r7_transfer_damping_min_efficiency=0.0,
        r7_transfer_damping_shape="o1d_competition",
        r7_transfer_enhancement_exchange_time_yr=1.0e8 / (365.25 * 24 * 60 * 60),
        r7_transfer_enhancement_exposure_time_yr=1.10,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=True,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_two_process_fig8_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.19,
        r7_transfer_damping_exponent=3.5,
        r7_transfer_damping_gpp_power=1.25,
        r7_transfer_enhancement_statistical_fraction=0.005,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=False,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_two_process_dynamic_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.19,
        r7_transfer_damping_exponent=3.5,
        r7_transfer_damping_gpp_power=1.25,
        r7_transfer_enhancement_statistical_fraction=0.005,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=False,
        r7_full_atmosphere_enhancement_statistical_fraction=0.29,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_exchange_saturation_dynamic_candidate": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.42,
        r7_transfer_damping_exponent=4.0,
        r7_transfer_damping_reference_ppm=30000.0,
        r7_transfer_damping_gpp_power=1.75,
        r7_transfer_damping_min_efficiency=0.0,
        r7_transfer_damping_shape="exchange_saturation",
        r7_transfer_enhancement_statistical_fraction=0.0065,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=False,
        r7_full_atmosphere_enhancement_statistical_fraction=0.29,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_two_box_export_proxy_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=1.0,
        r7_transfer_damping_gpp_power=0.50,
        r7_transfer_damping_min_efficiency=0.0,
        r7_transfer_damping_shape="two_box_export_proxy",
        r7_two_box_lower_half_ppm=3000.0,
        r7_two_box_upper_half_ppm=100000.0,
        r7_two_box_upper_source_fraction=0.60,
        r7_two_box_upper_export_efficiency=0.75,
        r7_two_box_normalize_modern=True,
        r7_transfer_enhancement_statistical_fraction=0.0065,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=False,
        r7_full_atmosphere_enhancement_statistical_fraction=0.29,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "physical_fast_r8_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        r8_rate_factor=3.0,
    ),
    "physical_fast_r8_ocean085_diagnostic": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        r8_rate_factor=3.0,
        co2_ocean_infusion_factor=0.85,
    ),
    "physical_lasaga_low_po2": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern_lasaga_burial",
        o2_budget_mode="target_p_o2_balance",
        young_steady_co2_source_isotope_mode="smow",
    ),
    "r7_balance_shape_test": ModelConfig(
        model_variant="r7_balance_diagnostic",
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="young_fixed_gpp",
        young_steady_co2_source_isotope_mode="smow",
    ),
    "physical_amif_1109": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        a_mif=1.109,
    ),
    "updated_physical_two_box_export_experimental": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.0,
        r7_transfer_damping_exponent=12.0,
        r7_transfer_damping_gpp_power=1.25,
        r7_transfer_enhancement_exchange_time_yr=1.0e8 / (365.25 * 24 * 60 * 60),
        r7_transfer_enhancement_exposure_time_yr=1.10,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=True,
        co2_export_signature_mode="two_box_survival",
        co2_export_upper_survival_fraction=0.42,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "updated_physical_two_box_transport_experimental": ModelConfig(
        model_variant="r7_throughput_diagnostic",
        r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
        r8c_factor=R8C_REFERENCE_RATIO_FACTOR,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        co2_sink_factor=0.94,
        r8_rate_factor=1.0,
        r7_transfer_damping_amplitude=0.0,
        r7_transfer_damping_exponent=12.0,
        r7_transfer_damping_gpp_power=1.25,
        r7_transfer_enhancement_exchange_time_yr=1.0e8 / (365.25 * 24 * 60 * 60),
        r7_transfer_enhancement_exposure_time_yr=1.10,
        r7_transfer_enhancement_half_ppm=300.0,
        r7_transfer_enhancement_exposure_mode="none",
        r7_transfer_enhancement_exposure_power=1.0,
        r7_transfer_enhancement_branch_group="incoming_heavy",
        r7_transfer_enhancement_full_atmosphere_only=True,
        co2_export_signature_mode="two_box_survival",
        co2_export_upper_survival_fraction=0.42,
        co2_export_coupled_transport=True,
        alpha_respiration_18=1.0 / 1.017,
        beta_respiration_17=0.5150,
        evapotranspiration_alpha_18=1.0063077,
    ),
    "r7_balance_amif_1109": ModelConfig(
        model_variant="r7_balance_diagnostic",
        r8c_factor=1.0001807352,
        solve_mode="young_steady",
        closure_mode="modern",
        o2_budget_mode="target_p_o2_balance",
        r5_mode="dynamic_air_calibrated",
        young_steady_co2_source_isotope_mode="smow",
        a_mif=1.109,
    ),
}

# Stable public aliases. The older long diagnostic names remain available so
# old notebooks, metadata files, and plots continue to resolve.
PRESET_CONFIGS[CURRENT_YOUNG_PRINTED_PRESET] = PRESET_CONFIGS["paper_reconstruction"]
PRESET_CONFIGS[CURRENT_UPDATED_PHYSICAL_PRESET] = PRESET_CONFIGS["physical_directional_r7_no_damping_candidate"]
PRESET_CONFIGS["physical_threshold_vertical_access_candidate"] = replace(
    PRESET_CONFIGS[CURRENT_UPDATED_PHYSICAL_PRESET],
    r7_transfer_damping_amplitude=0.30,
    r7_transfer_damping_exponent=0.0,
    r7_transfer_damping_reference_ppm=294.4,
    r7_transfer_damping_gpp_power=1.50,
    r7_transfer_damping_min_efficiency=0.0,
    r7_transfer_damping_shape="threshold_vertical_access",
    r7_vertical_gate_half_ppm=30000.0,
    r7_vertical_gate_hill=6.0,
)
PRESET_CONFIGS["young_reproduction_explicit_lower_threshold_candidate"] = replace(
    PRESET_CONFIGS["physical_threshold_vertical_access_candidate"],
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="gate_decreasing_1p00_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=3.0,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS["young_reproduction_explicit_lower_split_candidate"] = replace(
    PRESET_CONFIGS["physical_threshold_vertical_access_candidate"],
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="gate_decreasing_1p00_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=1.0,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[YOUNG_LIKE_V0_NAME] = ModelConfig(
    model_variant="r7_throughput_diagnostic",
    r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
    r8c_factor=1.0001807352,
    solve_mode="young_steady",
    closure_mode="modern",
    o2_budget_mode="target_p_o2_balance",
    r5_mode="dynamic_air_calibrated",
    young_steady_co2_source_isotope_mode="smow",
    co2_sink_factor=0.94,
    r8_rate_factor=1.0,
    photo_o17_source_law="three_term_processed_column",
    processed_column_source_d17o_permil=TABLE3_TARGETS["D17_O1D_permil"],
    processed_column_transition_tau_yr=V0_STEADY_SOURCE_LAW.transition_tau_yr,
    processed_column_recovery_tau_yr=V0_STEADY_SOURCE_LAW.recovery_tau_yr,
    processed_column_gate_half_ppm=V0_STEADY_SOURCE_LAW.gate_half_ppm,
    processed_column_gate_hill=V0_STEADY_SOURCE_LAW.gate_hill,
    processed_column_gpp_power=V0_STEADY_SOURCE_LAW.gpp_power,
    processed_column_transition_activation=V0_STEADY_SOURCE_LAW.transition_activation,
    processed_column_shared_tail_activation=V0_STEADY_SOURCE_LAW.shared_tail_activation,
    processed_column_low_gpp_tail_activation=V0_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    processed_column_low_gpp_recovery_activation=V0_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    processed_column_low_gpp_recovery_po2_mode=V0_STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    processed_column_low_gpp_recovery_po2_power=V0_STEADY_SOURCE_LAW.low_gpp_recovery_po2_power,
    processed_column_low_gpp_recovery_po2_half_pal=V0_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    processed_column_low_gpp_recovery_po2_hill=V0_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    r7_finite_exposure_exchange_time_modern_yr=TRANSIENT_LAW.exchange_time_modern_yr,
    r7_finite_exposure_residence_time_yr=TRANSIENT_LAW.residence_time_yr,
    r7_finite_exposure_max_excess=TRANSIENT_LAW.max_excess,
    r7_finite_exposure_branch_group=TRANSIENT_LAW.branch_group,
    r7_finite_exposure_full_atmosphere_only=TRANSIENT_LAW.full_atmosphere_only,
    alpha_respiration_18=1.0 / 1.017,
    beta_respiration_17=0.5150,
    evapotranspiration_alpha_18=1.0063077,
)
PRESET_CONFIGS[CURRENT_ARCHITECTURE_NAME] = ModelConfig(
    model_variant="r7_throughput_diagnostic",
    r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
    r8c_factor=1.0001807352,
    solve_mode="young_steady",
    closure_mode="modern",
    o2_budget_mode="target_p_o2_balance",
    r5_mode="dynamic_air_calibrated",
    young_steady_co2_source_isotope_mode="smow",
    co2_sink_factor=0.94,
    r8_rate_factor=1.0,
    photo_o17_source_law="three_term_processed_column",
    processed_column_source_d17o_permil=TABLE3_TARGETS["D17_O1D_permil"],
    processed_column_transition_tau_yr=STEADY_SOURCE_LAW.transition_tau_yr,
    processed_column_recovery_tau_yr=STEADY_SOURCE_LAW.recovery_tau_yr,
    processed_column_gate_half_ppm=STEADY_SOURCE_LAW.gate_half_ppm,
    processed_column_gate_hill=STEADY_SOURCE_LAW.gate_hill,
    processed_column_gpp_power=STEADY_SOURCE_LAW.gpp_power,
    processed_column_transition_activation=STEADY_SOURCE_LAW.transition_activation,
    processed_column_shared_tail_activation=STEADY_SOURCE_LAW.shared_tail_activation,
    processed_column_low_gpp_tail_activation=STEADY_SOURCE_LAW.low_gpp_tail_activation,
    processed_column_low_gpp_recovery_activation=STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    processed_column_low_gpp_recovery_po2_mode=STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    processed_column_low_gpp_recovery_po2_power=STEADY_SOURCE_LAW.low_gpp_recovery_po2_power,
    processed_column_low_gpp_recovery_po2_half_pal=STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    processed_column_low_gpp_recovery_po2_hill=STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    r7_finite_exposure_exchange_time_modern_yr=TRANSIENT_LAW.exchange_time_modern_yr,
    r7_finite_exposure_residence_time_yr=TRANSIENT_LAW.residence_time_yr,
    r7_finite_exposure_max_excess=TRANSIENT_LAW.max_excess,
    r7_finite_exposure_branch_group=TRANSIENT_LAW.branch_group,
    r7_finite_exposure_full_atmosphere_only=TRANSIENT_LAW.full_atmosphere_only,
    alpha_respiration_18=1.0 / 1.017,
    beta_respiration_17=0.5150,
    evapotranspiration_alpha_18=1.0063077,
)
PRESET_CONFIGS[YOUNG_LIKE_V1_NAME] = ModelConfig(
    model_variant="r7_throughput_diagnostic",
    r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
    r8c_factor=1.0001807352,
    solve_mode="young_steady",
    closure_mode="modern",
    o2_budget_mode="target_p_o2_balance",
    r5_mode="dynamic_air_calibrated",
    young_steady_co2_source_isotope_mode="smow",
    co2_sink_factor=0.94,
    r8_rate_factor=1.0,
    photo_o17_source_law="three_term_processed_column",
    processed_column_source_d17o_permil=TABLE3_TARGETS["D17_O1D_permil"],
    processed_column_transition_tau_yr=V1_STEADY_SOURCE_LAW.transition_tau_yr,
    processed_column_recovery_tau_yr=V1_STEADY_SOURCE_LAW.recovery_tau_yr,
    processed_column_gate_half_ppm=V1_STEADY_SOURCE_LAW.gate_half_ppm,
    processed_column_gate_hill=V1_STEADY_SOURCE_LAW.gate_hill,
    processed_column_gpp_power=V1_STEADY_SOURCE_LAW.gpp_power,
    processed_column_transition_activation=V1_STEADY_SOURCE_LAW.transition_activation,
    processed_column_shared_tail_activation=V1_STEADY_SOURCE_LAW.shared_tail_activation,
    processed_column_low_gpp_tail_activation=V1_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    processed_column_low_gpp_recovery_activation=V1_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    processed_column_low_gpp_recovery_po2_mode=V1_STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    processed_column_low_gpp_recovery_po2_power=V1_STEADY_SOURCE_LAW.low_gpp_recovery_po2_power,
    processed_column_low_gpp_recovery_po2_half_pal=V1_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    processed_column_low_gpp_recovery_po2_hill=V1_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    r7_finite_exposure_exchange_time_modern_yr=TRANSIENT_LAW.exchange_time_modern_yr,
    r7_finite_exposure_residence_time_yr=TRANSIENT_LAW.residence_time_yr,
    r7_finite_exposure_max_excess=TRANSIENT_LAW.max_excess,
    r7_finite_exposure_branch_group=TRANSIENT_LAW.branch_group,
    r7_finite_exposure_full_atmosphere_only=TRANSIENT_LAW.full_atmosphere_only,
    alpha_respiration_18=1.0 / 1.017,
    beta_respiration_17=0.5150,
    evapotranspiration_alpha_18=1.0063077,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_NAME] = ModelConfig(
    model_variant="r7_throughput_diagnostic",
    r7_throughput_factor=CURRENT_R7_THROUGHPUT_FACTOR,
    r8c_factor=1.0001807352,
    solve_mode="young_steady",
    closure_mode="modern",
    o2_budget_mode="target_p_o2_balance",
    r5_mode="dynamic_air_calibrated",
    young_steady_co2_source_isotope_mode="smow",
    co2_sink_factor=0.94,
    r8_rate_factor=1.0,
    photo_o17_source_law="three_term_processed_column",
    processed_column_source_d17o_permil=TABLE3_TARGETS["D17_O1D_permil"],
    processed_column_transition_tau_yr=V2_STEADY_SOURCE_LAW.transition_tau_yr,
    processed_column_recovery_tau_yr=V2_STEADY_SOURCE_LAW.recovery_tau_yr,
    processed_column_gate_half_ppm=V2_STEADY_SOURCE_LAW.gate_half_ppm,
    processed_column_gate_hill=V2_STEADY_SOURCE_LAW.gate_hill,
    processed_column_gpp_power=V2_STEADY_SOURCE_LAW.gpp_power,
    processed_column_transition_activation=V2_STEADY_SOURCE_LAW.transition_activation,
    processed_column_shared_tail_activation=V2_STEADY_SOURCE_LAW.shared_tail_activation,
    processed_column_low_gpp_tail_activation=V2_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    processed_column_low_gpp_recovery_activation=V2_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    processed_column_low_gpp_recovery_po2_mode=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    processed_column_low_gpp_recovery_po2_power=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_power,
    processed_column_low_gpp_recovery_po2_half_pal=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    processed_column_low_gpp_recovery_po2_hill=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    r7_finite_exposure_exchange_time_modern_yr=V2_TRANSIENT_LAW.exchange_time_modern_yr,
    r7_finite_exposure_residence_time_yr=V2_TRANSIENT_LAW.residence_time_yr,
    r7_finite_exposure_max_excess=V2_TRANSIENT_LAW.max_excess,
    r7_finite_exposure_branch_group=V2_TRANSIENT_LAW.branch_group,
    r7_finite_exposure_full_atmosphere_only=V2_TRANSIENT_LAW.full_atmosphere_only,
    alpha_respiration_18=1.0 / 1.017,
    beta_respiration_17=0.5150,
    evapotranspiration_alpha_18=1.0063077,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_PARALLEL_CO2_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_NAME],
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="gate_decreasing_1p00_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=1.0,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_NAME],
    processed_access_reservoir_mode="lagged_state",
    processed_access_reservoir_tau_yr=30.0,
    processed_access_reservoir_initial=1.0,
    processed_access_reservoir_half_pal=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    processed_access_reservoir_hill=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_NAME],
    processed_access_reservoir_mode="lagged_state",
    processed_access_reservoir_tau_yr=1.0,
    processed_access_reservoir_initial=1.0,
    processed_access_reservoir_half_pal=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    processed_access_reservoir_hill=V2_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME],
    processed_column_gpp_power=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.gpp_power,
    processed_column_low_gpp_signal_mode=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_signal_mode,
    processed_column_low_gpp_tail_activation=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    processed_column_low_gpp_recovery_activation=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME],
    processed_column_low_gpp_tail_po2_fraction=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_po2_fraction,
    processed_column_low_gpp_tail_po2_power=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_po2_power,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME],
    processed_column_gate_half_ppm=V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_STEADY_SOURCE_LAW.gate_half_ppm,
    processed_column_gate_hill=V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_STEADY_SOURCE_LAW.gate_hill,
    processed_column_shared_tail_activation=V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_STEADY_SOURCE_LAW.shared_tail_activation,
    processed_column_low_gpp_tail_activation=V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_activation,
)
PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME] = replace(
    PRESET_CONFIGS[YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME],
    processed_column_gate_half_ppm=V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW.gate_half_ppm,
    processed_column_gate_hill=V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW.gate_hill,
    processed_column_shared_tail_activation=V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW.shared_tail_activation,
    processed_column_shared_tail_po2_fraction=V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW.shared_tail_po2_fraction,
    processed_column_shared_tail_po2_power=V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW.shared_tail_po2_power,
    processed_column_low_gpp_tail_activation=V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_activation,
)
PRESET_CONFIGS[CURRENT_YOUNG_REPRODUCTION_PRESET] = PRESET_CONFIGS[
    YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME
]
PRESET_CONFIGS["updated_physical_from_young_modern_export"] = replace(
    PRESET_CONFIGS[CURRENT_YOUNG_REPRODUCTION_PRESET],
    a_mif=PACK2021_YOUNG_REPRO_ANCHORED_A_MIF,
    co2_export_signature_mode="two_box_survival",
    co2_export_upper_survival_fraction=ADNEW2025_YOUNG_REPRO_EXPORT_SURVIVAL_FRACTION,
)
PRESET_CONFIGS["updated_physical_from_young_explicit_export"] = replace(
    PRESET_CONFIGS[CURRENT_YOUNG_REPRODUCTION_PRESET],
    a_mif=PACK2021_EXPLICIT_EXPORT_A_MIF,
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="static_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[CURRENT_UPDATED_PHYSICAL_PRESET] = PRESET_CONFIGS["updated_physical_from_young_explicit_export"]
PRESET_CONFIGS["updated_physical_modern_export"] = replace(
    PRESET_CONFIGS["updated_physical_two_box_export_experimental"],
    a_mif=PACK2021_ANCHORED_A_MIF,
)
PRESET_CONFIGS["young_reproduction_biosphere_anchor_candidate"] = replace(
    PRESET_CONFIGS[CURRENT_YOUNG_REPRODUCTION_PRESET],
    alpha_respiration_18=1.0 / 1.0175,
    beta_respiration_17=0.5150,
    evapotranspiration_alpha_18=1.0058,
    evapotranspiration_beta_17=0.524,
    processed_column_low_gpp_recovery_activation=(
        PRESET_CONFIGS[CURRENT_YOUNG_REPRODUCTION_PRESET].processed_column_low_gpp_recovery_activation * 0.10
    ),
)
PRESET_CONFIGS["young_reproduction_po2_feedback_candidate"] = replace(
    PRESET_CONFIGS["young_reproduction_biosphere_anchor_candidate"],
    processed_column_low_gpp_low_po2_transition_activation=1.20,
    processed_access_reservoir_half_pal=0.65,
    processed_access_reservoir_hill=12.0,
)
PRESET_CONFIGS["updated_physical_from_validated_young_explicit_export"] = replace(
    PRESET_CONFIGS["young_reproduction_po2_feedback_candidate"],
    a_mif=PACK2021_VALIDATED_YOUNG_EXPLICIT_EXPORT_A_MIF,
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="static_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[UPDATED_VALIDATED_YOUNG_LOCAL_PACK_PRESET] = replace(
    PRESET_CONFIGS["young_reproduction_po2_feedback_candidate"],
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="static_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[UPDATED_VALIDATED_YOUNG_BETA_PACK_PRESET] = replace(
    PRESET_CONFIGS["young_reproduction_po2_feedback_candidate"],
    beta_respiration_17=PACK2021_BETA_RESPIRATION_17_CANDIDATE,
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="static_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET] = replace(
    PRESET_CONFIGS["young_reproduction_po2_feedback_candidate"],
    evapotranspiration_beta_17=PACK2021_EVAPOTRANSPIRATION_BETA_17_CANDIDATE,
    explicit_lower_box_mode="enabled",
    explicit_lower_box_upper_mode="static_0p60",
    explicit_lower_box_lower_to_trop_rate_per_year=2.0,
    explicit_lower_box_net_export_rate_per_year=ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
    explicit_lower_box_lower_major_scale=1.0,
)
PRESET_CONFIGS[CURRENT_UPDATED_PHYSICAL_PRESET] = PRESET_CONFIGS[UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET]

# Physics-extrapolation branch: identical to updated_physical but with the
# empirical three-term processed-column O2 source law switched OFF, so the
# tropospheric O2 Delta'17O is set by the resolved R7/stratosphere chemistry
# with a physical source-water photosynthetic source. This reproduces Young
# Fig. 7/Fig. 8 to ~0.17 per mil mean and, unlike the gated branch, extrapolates
# monotonically and physically far outside Young's calibration box (high pCO2,
# low/high pO2), which is required for application-regime values down to about
# -20 per mil. Use this branch for out-of-domain "serious" values; use
# updated_physical when tight in-domain agreement with Young's exact Fig. 8
# curvature (the 50% GPP / 30000 ppm endpoint) is the priority.
# Legacy reduced-model default: gate-off coupled solve. Its R5 normalization is
# a labelled Table 3-balanced reconstruction diagnostic, not a printed Young term.
# R7 throughput is taken from the Yung et al. (1991) O(1D)+CO2 rate (see
# conventions.py / r7_rate_sources.py) instead of an anonymous tuned factor; it
# reproduces Young Fig. 7/8 in-domain essentially as well and anchors modern O2
# Delta'17O slightly closer to Pack (2021). The R5 convention is the unresolved
# Table 3-balanced diagnostic. O(1D) Delta'17O = Young 27 permil, R8 = 1 yr,
# and Young burial are the other inherited defaults.
PRESET_CONFIGS[PHYSICAL_EXTRAPOLATION_PRESET] = replace(
    PRESET_CONFIGS[CURRENT_UPDATED_PHYSICAL_PRESET],
    photo_o17_source_law="source_water",
    # Explicitly retire the empirical scaffolding for the public default. These
    # are verified inert for the gate-off branch (neutralizing them changes O2
    # Delta'17O by 0.000000 per mil across the pCO2/pO2/GPP grid); setting them
    # here makes the public config self-documenting as scaffolding-free.
    processed_access_reservoir_mode="none",
    r7_transfer_damping_amplitude=0.0,
    r7_transfer_enhancement_gain=0.0,
    **R7_O1D_CO2_RATE[R7_DEFAULT].parameters,
)

PRESET_O2_D17O_CALIBRATION_MODES: dict[str, str] = {
    UPDATED_VALIDATED_YOUNG_LOCAL_PACK_PRESET: "pack2021_validated_young_local_offset",
}

PRESET_ROLES: dict[str, str] = {
    CURRENT_YOUNG_PRINTED_PRESET: (
        "Closest literal reconstruction of Young printed inputs/equations currently implemented; "
        "use this to expose unresolved reconstruction gaps."
    ),
    CURRENT_YOUNG_REPRODUCTION_PRESET: (
        "Best current reduced reproduction of Young figures and text. It points to the balanced low-GPP tail-access "
        "branch that passes the full Young acceptance gate and is the preferred Young-style validation anchor."
    ),
    CURRENT_UPDATED_PHYSICAL_PRESET: (
        "Young-tuned updated-model branch (former public default). It builds from the validated "
        "Young-reproduction pO2-feedback branch, keeps the Young a_MIF value, applies the Pack, 2021 "
        "modern O2 Delta'17O anchor through a source-water/transpiration beta update, and uses an "
        "explicit lower-stratospheric CO2 export box. Use it when an exact modern Pack anchor or Young's "
        "exact 50% GPP / 30000 ppm Fig. 8 endpoint is the priority; otherwise prefer physical_extrapolation."
    ),
    PHYSICAL_EXTRAPOLATION_PRESET: (
        "Public default. Resolved-chemistry branch: updated_physical with the empirical processed-column "
        "O2 source law disabled, so O2 Delta'17O comes from the resolved R7/stratosphere chemistry with a "
        "physical source-water photosynthetic source. Reproduces Young Fig. 7/Fig. 8 to ~0.17 per mil and "
        "extrapolates monotonically across the literature pO2 (0.1-2.1 PAL), high-pCO2 (to ~100000 ppm), "
        "and GPP envelopes. Modern O2 Delta'17O is the raw chemistry value (~ -0.427 permil), within "
        "uncertainty of the Pack 2021 anchor (-0.432 permil); no modern output offset is applied. "
        "Preferred for application-regime values outside Young's calibration box."
    ),
    "paper_reconstruction": "Legacy name for young_printed_inputs.",
    "physical_directional_r7_transfer_diagnostic": "Legacy name for young_reproduction.",
    "physical_directional_r7_no_damping_candidate": "Legacy no-damping candidate retained for comparison; not the current updated_physical alias.",
    "updated_physical_two_box_export_experimental": (
        "Experimental updated-physical branch with an export-aware lower stratospheric CO2 signature; "
        "currently affects exported isoflux metadata, not the core ODE solution."
    ),
    "updated_physical_modern_export": (
        "Updated-physical candidate with the no-damping O2 solution plus a modern STE-consistent "
        "two-box CO2 export signature and an explicit Pack, 2021 modern O2 Delta'17O anchor "
        f"implemented through a_MIF={PACK2021_ANCHORED_A_MIF:.12g}. This is the current preferred "
        "direction for the publication model, but it remains separate until the explicit vertical ODE "
        "extension is finalized."
    ),
    "young_reproduction_biosphere_anchor_candidate": (
        "Diagnostic Young-reproduction candidate that improves the modern Fig. 7/Table 3 O2 Delta'17O anchor "
        "using only respiration/source-water isotope-closure parameters, while reducing the low-GPP recovery "
        "term to preserve the Young Fig. 9 transient minimum. It is not the accepted Young-reproduction alias yet."
    ),
    "young_reproduction_po2_feedback_candidate": (
        "Diagnostic Young-reproduction candidate built from the biosphere-anchor candidate with an added low-GPP, "
        "low-pO2 processed-column transition response. This tests whether Young's halved-photosynthesis final state "
        "requires a pO2-sensitive source response that is inactive in fixed-pO2 Fig. 7/Fig. 8 experiments."
    ),
    "updated_physical_from_young_modern_export": (
        "Bridge candidate built directly from the accepted Young-reproduction branch. It keeps the validated "
        "Young-like O2 response, applies the Pack, 2021 modern O2 Delta'17O anchor through "
        f"a_MIF={PACK2021_YOUNG_REPRO_ANCHORED_A_MIF:.12g}, and reports the modern two-box CO2 export "
        "signature with an Adnew-2025-matching upper survival fraction "
        f"{ADNEW2025_YOUNG_REPRO_EXPORT_SURVIVAL_FRACTION:.12g}. This is the preferred bridge for developing "
        "the final updated model, but it is not yet the public default until GPP normalization and vertical "
        "CO2 mass balance are settled."
    ),
    "updated_physical_from_young_explicit_export": (
        "Bridge candidate built directly from the accepted Young-reproduction branch with an explicit lower "
        "stratospheric CO2 export box. It keeps the validated Young-like O2 response, applies the Pack, 2021 "
        f"modern O2 Delta'17O anchor through a_MIF={PACK2021_EXPLICIT_EXPORT_A_MIF:.12g}, uses a 60% "
        "upper-source lower-box mixture, and sets the net lower-box export rate to "
        f"{ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR:.12g} yr^-1 so the modern CO2 Delta'17O isoflux "
        "matches Adnew et al., 2025. Treat as the current explicit-export candidate for the updated model, not as a "
        "Young printed-equation reconstruction."
    ),
    "updated_physical_from_validated_young_explicit_export": (
        "Updated-model branch built directly from the validated Young pO2-feedback candidate with an explicit lower "
        "stratospheric CO2 export box. It applies the Pack, 2021 modern O2 Delta'17O anchor through "
        f"a_MIF={PACK2021_VALIDATED_YOUNG_EXPLICIT_EXPORT_A_MIF:.12g}, uses a 60% upper-source lower-box mixture, "
        "and sets the net lower-box export rate to "
        f"{ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR:.12g} yr^-1 as the Adnew et al., 2025 modern CO2 "
        "isoflux calibration."
    ),
    UPDATED_VALIDATED_YOUNG_LOCAL_PACK_PRESET: (
        "Updated-model policy candidate built from the validated Young pO2-feedback branch with an explicit lower "
        "stratospheric CO2 export box. It keeps the Young-like ODE and a_MIF behavior, then applies the Pack, 2021 "
        "modern O2 Delta'17O anchor as a transparent reported-output calibration rather than a global ozone-MIF "
        "rate change."
    ),
    UPDATED_VALIDATED_YOUNG_BETA_PACK_PRESET: (
        "Updated-model mechanism candidate built from the validated Young pO2-feedback branch with an explicit lower "
        "stratospheric CO2 export box. It keeps a_MIF at the Young value and moves modern O2 Delta'17O toward the "
        f"Pack, 2021 anchor by setting beta_respiration_17={PACK2021_BETA_RESPIRATION_17_CANDIDATE:g}. This is a "
        "physically traceable biosphere-isotope sensitivity candidate, not yet the accepted public model."
    ),
    UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET: (
        "Updated-model mechanism candidate built from the validated Young pO2-feedback branch with an explicit lower "
        "stratospheric CO2 export box. It keeps a_MIF and respiration beta at the Young-like values and moves modern "
        "O2 Delta'17O toward the Pack, 2021 anchor by setting evapotranspiration_beta_17="
        f"{PACK2021_EVAPOTRANSPIRATION_BETA_17_CANDIDATE:g}. This is a source-water/transpiration sensitivity "
        "candidate within the Young-discussed beta range, not yet the accepted public model."
    ),
    "updated_physical_two_box_transport_experimental": (
        "Experimental updated-physical branch with lower-box CO2 export coupled into stratosphere-to-troposphere "
        "transport; uses a hidden lower box and is not yet the final explicit ODE extension."
    ),
    "physical_two_process_fig8_diagnostic": (
        "Diagnostic branch testing a two-process Fig. 8 shape mechanism: very small incoming-heavy R7 enhancement "
        "plus low-GPP/high-pCO2 transfer damping. Do not use as the final updated model without literature support."
    ),
    "physical_two_process_dynamic_diagnostic": (
        "Diagnostic branch combining the improved Fig. 8 two-process steady shape with an additional full-atmosphere-only "
        "R7 enhancement for Young Fig. 9 transient behavior. This explicitly separates steady and transient bookkeeping."
    ),
    "physical_exchange_saturation_dynamic_candidate": (
        "Candidate branch replacing the empirical high-pCO2 clamp with finite-exposure R7 exchange saturation. "
        "Promising for Young Fig. 8, but still requires literature-derived exposure/transport constraints."
    ),
    "physical_threshold_vertical_access_candidate": (
        "Candidate branch replacing the empirical high-pCO2 clamp with a threshold-like vertical/source access term "
        "inferred from the lower-box inverse Fig. 8 audit. It is a Young-reconstruction candidate, not the final "
        "updated/public model until the threshold is independently constrained."
    ),
    "young_reproduction_explicit_lower_threshold_candidate": (
        "Best current Fig. 8 Young-reconstruction candidate with an explicit lower-stratospheric CO2 export box "
        "coupled to the threshold vertical/source R7 access law. This is a stable scenario wrapper around the "
        "audit mechanism; dynamic Fig. 9/Fig. 10 tests still need an explicit full-atmosphere lower-box extension."
    ),
    "young_reproduction_explicit_lower_split_candidate": (
        "Transport-bounded explicit lower-box Young-reconstruction candidate. It uses faster internal lower-box "
        "isotope exchange than net one-way CO2 export, with net export near Young kST. Low-pCO2 source weighting "
        "is kept out of this preset because the full Fig. 7 benchmark rejects that diagnostic requirement shape."
    ),
    YOUNG_LIKE_V0_NAME: (
        "Pinned v0 young-like architecture: three-term processed-column photosynthetic O17O source law for "
        "steady Fig. 7/Fig. 8 plus transient-only finite-exposure R7 incoming-heavy transfer for Fig. 9/Fig. 10."
    ),
    CURRENT_ARCHITECTURE_NAME: (
        "Pinned current young-like architecture: three-term processed-column photosynthetic O17O source law for "
        "steady Fig. 7/Fig. 8 plus transient-only finite-exposure R7 incoming-heavy transfer for Fig. 9/Fig. 10. "
        "Use this as the main guardrailed reproduction candidate while remaining source-law details are formalized."
    ),
    YOUNG_LIKE_V1_NAME: (
        "Experimental steady-surface improvement candidate: residence-split processed-column source law improves "
        "Fig. 7/Fig. 8 residuals but currently makes the Fig. 9 final O2 Delta'17O too high unless another transient "
        "mechanism is added. Keep separate from the current whole-architecture preset."
    ),
    YOUNG_LIKE_V2_NAME: (
        "Experimental v2 young-like candidate: v1 residence-split steady source law plus a threshold-style pO2 "
        "access gate for the low-GPP recovery contribution and a 0.135 finite-exposure ratio. Treat as a candidate "
        "until the threshold is tied to an explicit altitude/processed-reservoir mechanism."
    ),
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME: (
        "Literature-grounded explicit transient-state candidate for v2: the threshold-access term is represented "
        "by a scalar processed-access reservoir relaxing with tau=1 yr, close to Young kST and published "
        "stratosphere-troposphere transport/exposure anchors. Steady Fig. 7/Fig. 8 behavior remains v2-like."
    ),
    YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME: (
        "Candidate descendant of the preferred tau=1 v2 branch. It replaces the unbounded low-GPP source signal "
        "with a gentler log1p signal selected by the bounded-low-GPP audit, while retaining the Young O(1D) "
        "signature, Young upper-column scale, pO2 access reservoir, and Fig. 8 steady fit."
    ),
    YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME: (
        "Candidate descendant of the log-low-GPP tau=1 branch. It routes half of the low-GPP processed-column "
        "tail through the same pO2/access gate used by the recovery term, preserving the Young acceptance "
        "benchmarks while reducing low-pO2 extrapolation hooks in the validity-envelope audit."
    ),
    YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME: (
        "Candidate descendant of the low-GPP tail-access branch. It relaxes the high-pCO2 processed-column "
        "tail gate from Hill=8 at 30,000 ppm to Hill=4 at 35,000 ppm and modestly retunes the shared/low-GPP "
        "tail amplitudes, preserving the Young acceptance benchmarks while reducing high-pCO2 contour waviness "
        "in dense pO2/GPP scans."
    ),
    YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME: (
        "Candidate descendant of the low-GPP tail-access branch selected by the gate-candidate override audit. "
        "It uses a moderately relaxed high-pCO2 processed-column gate (Hill=6 at 30,000 ppm), routes the shared "
        "tail through pO2 access, and sets shared and low-GPP tail amplitudes at 90% of the tail-access branch, "
        "improving Young Fig. 8 residuals while keeping low-pO2 extrapolation hooks small."
    ),
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME: (
        "Explicit transient-state candidate for v2: the same threshold-access term is represented by a scalar "
        "processed-access reservoir relaxing toward pO2 access with tau=30 yr. Steady Fig. 7/Fig. 8 runs remain "
        "v2-like; the extra state is active in full-atmosphere transient validation. This is retained as an "
        "effective-memory sensitivity because tau=30 yr is model-preserving but weakly anchored by direct "
        "transport/photochemistry literature."
    ),
    YOUNG_LIKE_V2_PARALLEL_CO2_NAME: (
        "Best current scoreable Young-like candidate combining the v2 O2 architecture with an explicit lower-box "
        "CO2 export diagnostic and a source-derived parallel high-altitude processed column component. This uses "
        "the Young printed upper-column scale, not an exact target-fitted CO2 anomaly."
    ),
    "physical_two_box_export_proxy_diagnostic": (
        "Diagnostic branch applying the upper/lower stratospheric R7 export proxy directly to R7 isotope-transfer branches. "
        "This tests mechanism shape only; it is not a final explicit two-box ODE implementation."
    ),
}


def preset_names() -> tuple[str, ...]:
    return tuple(PRESET_CONFIGS)


def preset_role(preset: str) -> str:
    return PRESET_ROLES.get(preset, "Diagnostic or legacy scenario preset.")


def config_from_scenario(scenario: ScenarioInput) -> ModelConfig:
    if scenario.preset not in PRESET_CONFIGS:
        raise ValueError(f"unknown preset {scenario.preset!r}; choices: {', '.join(preset_names())}")
    config = PRESET_CONFIGS[scenario.preset]
    updates = {
        "p_o2_pal": scenario.p_o2_pal,
        "p_co2_ppm": scenario.p_co2_ppm,
        "gpp_scale": internal_young_gpp_scale(
            scenario.gpp_scale,
            scenario.gpp_normalization,
            scenario.gpp_modern_pgC_per_year,
        ),
    }
    for field in (
        "solve_mode",
        "closure_mode",
        "o2_budget_mode",
        "model_variant",
        "r7_throughput_factor",
        "r7c_factor",
        "r7g_factor",
        "r8c_factor",
        "r5_mode",
        "r5_co2_partner_factor",
        "co2_photo_sink_factor",
        "r8_rate_factor",
        "a_mif",
        "co2_source_isotope_mode",
        "young_steady_co2_source_isotope_mode",
        "co2_sink_factor",
        "co2_ocean_infusion_factor",
        "r7_transfer_damping_amplitude",
        "r7_transfer_damping_exponent",
        "r7_transfer_damping_reference_ppm",
        "r7_transfer_damping_gpp_power",
        "r7_transfer_damping_min_efficiency",
        "r7_transfer_damping_shape",
        "r7_vertical_gate_half_ppm",
        "r7_vertical_gate_hill",
        "r7_two_box_lower_half_ppm",
        "r7_two_box_upper_half_ppm",
        "r7_two_box_upper_source_fraction",
        "r7_two_box_upper_export_efficiency",
        "r7_two_box_normalize_modern",
        "r7_transfer_enhancement_gain",
        "r7_transfer_enhancement_statistical_fraction",
        "r7_transfer_enhancement_exchange_time_yr",
        "r7_transfer_enhancement_exposure_time_yr",
        "r7_transfer_enhancement_half_ppm",
        "r7_transfer_enhancement_exposure_mode",
        "r7_transfer_enhancement_exposure_power",
        "r7_transfer_enhancement_branch_group",
        "r7_transfer_enhancement_full_atmosphere_only",
        "r7_finite_exposure_exchange_time_modern_yr",
        "r7_finite_exposure_residence_time_yr",
        "r7_finite_exposure_max_excess",
        "r7_finite_exposure_branch_group",
        "r7_finite_exposure_full_atmosphere_only",
        "r7_full_atmosphere_enhancement_statistical_fraction",
        "photo_o17_source_law",
        "processed_column_source_d17o_permil",
        "processed_column_transition_tau_yr",
        "processed_column_recovery_tau_yr",
        "processed_column_gate_mode",
        "processed_column_gate_half_ppm",
        "processed_column_gate_hill",
        "processed_column_gpp_power",
        "processed_column_low_gpp_signal_mode",
        "processed_column_transition_activation",
        "processed_column_shared_tail_activation",
        "processed_column_shared_tail_po2_fraction",
        "processed_column_shared_tail_po2_power",
        "processed_column_low_gpp_tail_activation",
        "processed_column_low_gpp_tail_po2_fraction",
        "processed_column_low_gpp_tail_po2_power",
        "processed_column_low_gpp_transition_activation",
        "processed_column_low_gpp_low_po2_transition_activation",
        "processed_column_low_gpp_recovery_activation",
        "processed_column_low_gpp_recovery_po2_mode",
        "processed_column_low_gpp_recovery_po2_power",
        "processed_column_low_gpp_recovery_po2_half_pal",
        "processed_column_low_gpp_recovery_po2_hill",
        "processed_access_reservoir_mode",
        "processed_access_reservoir_tau_yr",
        "processed_access_reservoir_initial",
        "processed_access_reservoir_half_pal",
        "processed_access_reservoir_hill",
        "co2_export_signature_mode",
        "co2_export_upper_survival_fraction",
        "co2_export_coupled_transport",
        "explicit_lower_box_mode",
        "explicit_lower_box_upper_mode",
        "explicit_lower_box_lower_to_trop_rate_per_year",
        "explicit_lower_box_net_export_rate_per_year",
        "explicit_lower_box_lower_major_scale",
        "explicit_lower_box_source_weight_mode",
        "explicit_lower_box_source_weight_boost",
        "explicit_lower_box_source_weight_half_ppm",
        "explicit_lower_box_source_weight_hill",
        "alpha_respiration_18",
        "beta_respiration_17",
        "evapotranspiration_alpha_18",
        "evapotranspiration_beta_17",
    ):
        value = getattr(scenario, field)
        if value is not None:
            updates[field] = value
    return replace(config, **updates)


def gpp_metadata_from_scenario(scenario: ScenarioInput, config: ModelConfig, rp_o2: float) -> dict[str, Any]:
    gpp_reference = modern_gpp_pgC_per_year(scenario.gpp_normalization, scenario.gpp_modern_pgC_per_year)
    gpp_key = "young_2014" if scenario.gpp_normalization is None else scenario.gpp_normalization
    gpp_reference_info = GPP_NORMALIZATIONS.get(gpp_key)
    return {
        "gpp_user_scale": scenario.gpp_scale,
        "gpp_user_percent_modern": scenario.gpp_scale * 100.0,
        "gpp_normalization": gpp_key,
        "gpp_normalization_role": normalization_role(gpp_key),
        "gpp_normalization_label": gpp_reference_info.label if gpp_reference_info is not None else "Custom",
        "gpp_normalization_note": gpp_reference_info.note if gpp_reference_info is not None else "User-specified modern GPP normalization.",
        "gpp_modern_reference_pgC_per_year": gpp_reference,
        "gpp_requested_pgC_per_year": requested_gpp_pgC_per_year(
            scenario.gpp_scale,
            scenario.gpp_normalization,
            scenario.gpp_modern_pgC_per_year,
        ),
        "gpp_internal_young_scale": config.gpp_scale,
        "effective_gpp_pgC_per_year": rp_o2 * 12.0107 / 1.0e15,
    }


def effective_rp_o2(config: ModelConfig) -> float:
    terms = effective_o2_budget_terms(config)
    if terms is not None:
        return terms.required_rp_o2
    return PARAMETERS["k_respiration_per_year"] * config.gpp_scale * 3.80e19


def effective_o2_budget_terms(config: ModelConfig) -> O2BudgetTerms | None:
    if config.rp_o2 is not None:
        return None
    if config.o2_budget_mode == "target_p_o2_balance":
        closure_mode = config.closure_mode
        if config.solve_mode == "young_steady":
            if closure_mode in ("modern_lasaga_burial", "lasaga_burial"):
                closure_mode = "lasaga_burial"
            elif closure_mode in ("modern_finite_geosphere", "finite_geosphere_burial"):
                closure_mode = "finite_geosphere_burial"
            else:
                closure_mode = "organic_burial"
        return consistent_rp_o2(config.p_o2_pal, config.gpp_scale, closure_mode)
    return None


_PACK2021_LOCAL_OFFSET_CACHE: dict[str, float] = {}


def o2_d17o_calibration_mode(scenario: ScenarioInput) -> str:
    if scenario.o2_d17o_calibration_mode is not None:
        return scenario.o2_d17o_calibration_mode
    return PRESET_O2_D17O_CALIBRATION_MODES.get(scenario.preset, "none")


def pack2021_local_o2_offset_for_preset(preset: str) -> float:
    """Return Pack 2021 minus raw modern O2 Delta'17O for a preset."""

    if preset not in _PACK2021_LOCAL_OFFSET_CACHE:
        raw_reference = run_scenario(
            ScenarioInput(
                preset=preset,
                p_o2_pal=1.0,
                p_co2_ppm=294.4,
                gpp_scale=1.0,
                solve_mode="full_atmosphere",
                o2_d17o_calibration_mode="none",
            )
        )
        _PACK2021_LOCAL_OFFSET_CACHE[preset] = (
            PACK2021_O2_D17O_TARGET_PERMIL - float(raw_reference.outputs["O2_trop_D17O_raw_permil"])
        )
    return _PACK2021_LOCAL_OFFSET_CACHE[preset]


def apply_o2_d17o_output_calibration(scenario: ScenarioInput, outputs: dict[str, Any]) -> None:
    """Apply a reported-output O2 Delta'17O calibration without changing ODE results."""

    raw_d17o = float(outputs["O2_trop_D17O_permil"])
    raw_d17_prime = float(outputs["O2_trop_d17_prime_permil"])
    outputs["O2_trop_D17O_raw_permil"] = raw_d17o
    outputs["O2_trop_d17_prime_raw_permil"] = raw_d17_prime
    outputs["o2_d17o_calibration_mode"] = "none"
    outputs["o2_d17o_calibration_source"] = ""
    outputs["o2_d17o_calibration_target_permil"] = ""
    outputs["o2_d17o_calibration_offset_permil"] = 0.0
    outputs["O2_trop_D17O_reported_permil"] = raw_d17o
    outputs["O2_trop_d17_prime_reported_permil"] = raw_d17_prime

    mode = o2_d17o_calibration_mode(scenario)
    if mode in ("none", ""):
        return
    if mode != "pack2021_validated_young_local_offset":
        raise ValueError(f"unknown O2 Delta'17O calibration mode {mode!r}")

    offset = pack2021_local_o2_offset_for_preset(scenario.preset)
    reported_d17o = raw_d17o + offset
    reported_d17_prime = raw_d17_prime + offset
    outputs["O2_trop_D17O_permil"] = reported_d17o
    outputs["O2_trop_d17_prime_permil"] = reported_d17_prime
    outputs["O2_trop_D17O_reported_permil"] = reported_d17o
    outputs["O2_trop_d17_prime_reported_permil"] = reported_d17_prime
    outputs["o2_d17o_calibration_mode"] = mode
    outputs["o2_d17o_calibration_source"] = (
        "Pack, 2021 modern atmospheric O2 Delta'17O anchor applied as a preset-local reported-output offset"
    )
    outputs["o2_d17o_calibration_target_permil"] = PACK2021_O2_D17O_TARGET_PERMIL
    outputs["o2_d17o_calibration_offset_permil"] = offset


def scenario_domain(config: ModelConfig, run_converged: bool) -> dict[str, Any]:
    """Return a machine-readable validity classification for a scenario.

    This is intentionally stricter and more compact than the human warning
    list. It separates hard outside-domain cases from exploratory but still
    mathematically usable model runs.
    """

    status = "valid"
    reasons: list[str] = []

    if not run_converged:
        status = "fail"
        reasons.append("solver_not_converged")

    budget_terms = effective_o2_budget_terms(config)
    if budget_terms is not None and budget_terms.required_rp_o2 <= 0.0:
        status = "fail"
        reasons.append("o2_budget_requires_nonpositive_photosynthesis")
    elif (
        budget_terms is not None
        and budget_terms.rp_relative_to_modern <= O2_BUDGET_NEAR_BOUNDARY_RP_RELATIVE_TO_MODERN
    ):
        if status != "fail":
            status = "warning"
        reasons.append("o2_budget_near_photosynthesis_boundary")

    pco2_status = pco2_envelope_status(config.p_co2_ppm)
    if pco2_status.level != "recommended":
        if status != "fail":
            status = "warning"
        reasons.append(f"pco2_{pco2_status.level}")

    if not in_working_po2_envelope(config.p_o2_pal):
        if status != "fail":
            status = "warning"
        reasons.append("po2_outside_working_envelope")

    if config.o2_budget_mode == "young_fixed_gpp" and abs(config.p_o2_pal - 1.0) > 1.0e-9:
        if status != "fail":
            status = "warning"
        reasons.append("young_fixed_gpp_nonmodern_po2")

    if config.closure_mode in ("modern_lasaga_burial", "lasaga_burial") and config.p_o2_pal < 0.6:
        if status != "fail":
            status = "warning"
        reasons.append("lasaga_low_o2_extrapolation")

    return {
        "status": status,
        "reasons": reasons,
        "reason_text": " | ".join(reasons),
    }


def scenario_warnings(config: ModelConfig, run_converged: bool) -> list[str]:
    warnings: list[str] = []
    if not run_converged:
        warnings.append("Solver did not converge; treat numerical outputs as diagnostic only.")
    if config.o2_budget_mode == "young_fixed_gpp" and abs(config.p_o2_pal - 1.0) > 1.0e-9:
        warnings.append(
            "pO2 differs from modern while o2_budget_mode is young_fixed_gpp; "
            "this can impose inconsistent modern photosynthesis on a non-modern O2 reservoir."
        )
    budget_terms = effective_o2_budget_terms(config)
    if budget_terms is not None and budget_terms.required_rp_o2 <= 0.0:
        warnings.append(
            "Target-pO2 O2 budget requires non-positive photosynthetic O2 flux; "
            "this pO2/GPP/closure combination is physically infeasible in the reduced model."
        )
    elif (
        budget_terms is not None
        and budget_terms.rp_relative_to_modern <= O2_BUDGET_NEAR_BOUNDARY_RP_RELATIVE_TO_MODERN
    ):
        warnings.append(
            "Target-pO2 O2 budget is very close to the positive-photosynthesis boundary "
            f"(rp <= {100.0 * O2_BUDGET_NEAR_BOUNDARY_RP_RELATIVE_TO_MODERN:g}% of modern); "
            "treat this low-pO2/low-GPP cell as an extrapolation warning."
        )
    if config.closure_mode in ("modern_lasaga_burial", "lasaga_burial") and config.p_o2_pal < 0.6:
        warnings.append(
            "Lasaga-Ohmoto burial feedback is in its low-O2/anoxic regime; "
            "interpret as literature-extended model behavior, not strict Young reproduction."
        )
    if not in_working_po2_envelope(config.p_o2_pal):
        warnings.append(
            f"pO2 is outside the {MODEL_PO2_WORKING_MIN_PAL:g}-{MODEL_PO2_WORKING_MAX_PAL:g} PAL "
            "Phanerozoic working envelope currently tracked for this model."
        )
    pco2_status = pco2_envelope_status(config.p_co2_ppm)
    if pco2_status.level != "recommended":
        warnings.append(pco2_status.message)
    if config.p_co2_ppm > PCO2_RECOMMENDED_MAX_PPM:
        warnings.append("pCO2 is beyond the main Young Fig. 8 plotted range; compare trends rather than exact values.")
    if abs(config.co2_sink_factor - 1.0) > 1.0e-12:
        warnings.append(
            "CO2 sink factor differs from Young Table 2 constants; use as an explicit transient-timescale sensitivity."
        )
    if abs(config.co2_ocean_infusion_factor - 1.0) > 1.0e-12:
        warnings.append(
            "Ocean CO2 infusion factor differs from Young Table 2 constants; use as an explicit Young-transient CO2 turnover convention."
        )
    if abs(config.r7_transfer_damping_amplitude) > 1.0e-12:
        warnings.append(
            "R7 O(1D)->CO2 isotope-transfer damping is a diagnostic high-pCO2 shape test, not an equation printed by Young et al."
        )
    if config.r7_transfer_damping_shape == "two_box_export_proxy":
        warnings.append(
            "R7 transfer uses an upper/lower stratospheric export proxy; this is a diagnostic mechanism-shape test, not yet an explicit two-box ODE module."
        )
    if config.r7_transfer_damping_shape == "vertical_gate_exchange_saturation":
        warnings.append(
            "R7 transfer uses a vertical-gated finite-exposure proxy; this is a diagnostic replacement candidate for the high-pCO2 shape term."
        )
    if config.r7_transfer_damping_shape == "threshold_vertical_access":
        warnings.append(
            "R7 transfer uses a threshold-like vertical/source access term inferred from Fig. 8 inverse efficiency; this is a candidate mechanism shape, not a printed Young equation."
        )
        if config.p_co2_ppm > PCO2_RECOMMENDED_MAX_PPM:
            warnings.append(
                "Threshold-access Young reconstruction is outside the validated Fig. 8 domain; "
                "the low-GPP high-pCO2 tail can develop a nonphysical hook and should be treated as exploratory."
            )
    if config.explicit_lower_box_mode != "none":
        warnings.append(
            "Scenario uses an explicit lower-stratospheric CO2 export box; this is a Young-reconstruction candidate extension, not one of the printed 27 ODEs."
        )
        if config.solve_mode not in ("young_steady", "fixed_reservoir"):
            warnings.append(
                "Explicit lower-box mode is currently implemented for steady isotope solves only; full transient dynamics still use the one-box atmosphere."
            )
        if config.explicit_lower_box_net_export_rate_per_year is not None:
            warnings.append(
                "Explicit lower-box net export rate is split from internal isotope exchange; this is a residence-time/export diagnostic, not a printed Young term."
            )
        if config.explicit_lower_box_source_weight_mode != "none":
            warnings.append(
                "Explicit lower-box source weighting modifies incoming-heavy R7 branches with a pCO2-dependent requirement shape; this is not a printed Young term."
            )
    if abs(config.r7c_factor - 1.0) > 1.0e-12 or abs(config.r7g_factor - 1.0) > 1.0e-12:
        warnings.append(
            "Branch-specific R7c/R7g factors differ from the shared R7 throughput convention; use as explicit modern CO2 isotope calibration diagnostics."
        )
    if abs(config.r7_transfer_enhancement_gain) > 1.0e-12:
        warnings.append(
            "R7 O(1D)->CO2 isotope-transfer enhancement is a diagnostic transient-branching proxy, not an equation printed by Young et al."
        )
    if config.r7_transfer_enhancement_statistical_fraction is not None:
        warnings.append(
            "R7 O(1D)->CO2 enhancement is expressed as partial activation of Yung/Liang statistical CO3* branching; the activation fraction remains diagnostic."
        )
    if (
        config.r7_transfer_enhancement_exchange_time_yr is not None
        and config.r7_transfer_enhancement_exposure_time_yr is not None
    ):
        warnings.append(
            "R7 O(1D)->CO2 enhancement is expressed as exchange-time activation of Yung/Liang statistical CO3* branching; exposure time remains diagnostic."
        )
    if config.r7_full_atmosphere_enhancement_statistical_fraction is not None:
        warnings.append(
            "Additional R7 O(1D)->CO2 enhancement is active only in full-atmosphere transient solves; use as a diagnostic separation of steady and transient behavior."
        )
    if config.photo_o17_source_law == "three_term_processed_column":
        warnings.append(
            "Photosynthetic O17O source uses the three-term processed-column Young-like architecture law; "
            "this is validated against Young Fig. 7/Fig. 8 but is not printed as an equation by Young et al."
        )
    if (
        config.alpha_respiration_18 is not None
        or config.beta_respiration_17 is not None
        or config.evapotranspiration_alpha_18 is not None
        or config.evapotranspiration_beta_17 is not None
    ):
        warnings.append(
            "Respiration/source-water isotope parameters override Young defaults; use as an explicit Dole-effect sensitivity."
        )
    if abs(config.r8_rate_factor - 1.0) > 1.0e-12:
        warnings.append(
            "R8 CO2-H2O exchange factor differs from Young's 1 yr^-1 value; use as a diagnostic isotope-exchange sensitivity unless independently justified."
        )
    if abs(config.r8c_factor - 1.0) > 1.0e-12:
        warnings.append(
            "R8c 17O exchange factor differs from the printed Young Table 2 convention; use as an explicit isotope-exchange convention sensitivity."
        )
    if config.a_mif is not None and abs(config.a_mif - PARAMETERS["a_MIF"]) > 1.0e-12:
        warnings.append(
            "aMIF differs from Young's basic 1.065 value; this is a Young-discussed ozone-MIF sensitivity, not the baseline run."
        )
    if config.co2_export_signature_mode != "bulk_stratosphere":
        if config.co2_export_coupled_transport:
            warnings.append(
                "CO2 stratosphere-troposphere transport uses an explicit lower-box export signature instead of the one-box bulk stratospheric CO2 value."
            )
        else:
            warnings.append(
                "CO2 stratosphere-troposphere isoflux uses an explicit export signature instead of the one-box bulk stratospheric CO2 value; core ODE chemistry is unchanged in this experimental branch."
            )
        if not 0.0 <= config.co2_export_upper_survival_fraction <= 1.0:
            warnings.append(
                "CO2 export upper-box survival fraction is outside 0-1; treat this as an extrapolation diagnostic, not a physical two-box mixture."
            )
    if config.co2_export_coupled_transport:
        warnings.append(
            "CO2 lower-box export signature is coupled into ST transport using a hidden-box approximation; isotope conservation is not explicit until the full extra-box ODE is implemented."
        )
    if config.processed_access_reservoir_mode != "none":
        warnings.append(
            "Processed-access reservoir metadata is enabled; the extra access-state ODE is active only in "
            "full-atmosphere transient experiments that explicitly extend the state vector."
        )
    return warnings


def co2_export_signature(config: ModelConfig, co2_trop_d17: float, co2_strat_d17: float) -> tuple[float, float]:
    """Return exported CO2 Delta17O and upper-box survival fraction."""

    if config.co2_export_signature_mode == "bulk_stratosphere":
        return co2_strat_d17, 1.0
    if config.co2_export_signature_mode == "two_box_survival":
        fraction = config.co2_export_upper_survival_fraction
        return co2_trop_d17 + fraction * (co2_strat_d17 - co2_trop_d17), fraction
    raise ValueError(f"unknown CO2 export signature mode: {config.co2_export_signature_mode}")


def explicit_lower_box_upper_fraction(config: ModelConfig) -> float:
    """Return the upper-source fraction used by the explicit lower CO2 box."""

    mode = config.explicit_lower_box_upper_mode
    if mode == "static_1p00":
        return 1.0
    if mode == "static_0p75":
        return 0.75
    if mode == "static_0p60":
        return 0.60
    pco2 = max(config.p_co2_ppm, 0.0)
    half = max(config.r7_vertical_gate_half_ppm, 1.0e-30)
    hill = max(config.r7_vertical_gate_hill, 1.0e-12)
    gate = pco2**hill / (pco2**hill + half**hill)
    if mode == "gate_decreasing_1p00_0p60":
        return 1.0 - 0.40 * gate
    if mode == "gate_increasing_0p60_1p00":
        return 0.60 + 0.40 * gate
    raise ValueError(f"unknown explicit lower-box upper mode: {mode}")


def explicit_lower_box_source_weight(config: ModelConfig) -> float:
    """Return the pCO2-dependent incoming-heavy R7 source weight."""

    mode = config.explicit_lower_box_source_weight_mode
    if mode in ("none", None):
        return 1.0
    if mode == "low_pco2_gate":
        pco2 = max(config.p_co2_ppm, 0.0)
        boost = config.explicit_lower_box_source_weight_boost
        half = max(config.explicit_lower_box_source_weight_half_ppm, 1.0e-30)
        hill = max(config.explicit_lower_box_source_weight_hill, 1.0e-12)
        return 1.0 + boost / (1.0 + (pco2 / half) ** hill)
    raise ValueError(f"unknown explicit lower-box source weight mode: {mode}")


def run_explicit_lower_box_scenario(scenario: ScenarioInput, config: ModelConfig) -> ScenarioResult:
    """Run the explicit lower-box steady isotope candidate through ScenarioResult."""

    from explicit_lower_co2_box_model import (
        EXTENDED_SPECIES_ORDER,
        LowerBoxConfig,
        solve_explicit_lower_box_fixed_reservoir,
        summarize_extended_state,
    )

    lower_box = LowerBoxConfig(
        upper_survival_fraction=explicit_lower_box_upper_fraction(config),
        lower_to_trop_rate_per_year=config.explicit_lower_box_lower_to_trop_rate_per_year,
        net_export_rate_per_year=config.explicit_lower_box_net_export_rate_per_year,
        lower_major_scale=config.explicit_lower_box_lower_major_scale,
    )
    source_weight = explicit_lower_box_source_weight(config)
    effective_scenario = replace(
        scenario,
        r7c_factor=config.r7c_factor * source_weight,
        r7g_factor=config.r7g_factor * source_weight,
    )
    result, _, _ = solve_explicit_lower_box_fixed_reservoir(effective_scenario, lower_box)
    summaries = summarize_extended_state(result.y)
    idx = {name: i for i, name in enumerate(EXTENDED_SPECIES_ORDER)}
    rp = effective_rp_o2(config)
    o2_budget = effective_o2_budget_terms(config)
    domain = scenario_domain(config, result.converged)
    o2_percent = percent_equivalent(config.p_o2_pal)
    co2_trop_d17 = summaries["CO2_trop"].cap_delta17
    co2_strat_d17 = summaries["CO2_strat"].cap_delta17
    co2_export_d17 = summaries["CO2_lower"].cap_delta17
    co2_flux_one_box = (
        PARAMETERS["k_ST_per_year"]
        * result.y[idx["CO2_strat"]]
        * co2_strat_d17
    )
    co2_flux = (
        lower_box.effective_net_export_rate_per_year
        * result.y[idx["CO2_lower"]]
        * co2_export_d17
    )
    co2_lower_major_export_flux = lower_box.effective_net_export_rate_per_year * result.y[idx["CO2_lower"]]
    processed_parallel_d17 = TABLE3_TARGETS["D17_O1D_permil"]
    processed_parallel_flux_for_young_anomaly = (
        (TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"] - co2_flux) / processed_parallel_d17
    )
    processed_parallel_flux_for_young_fixed_d17 = (
        (TABLE3_TARGETS["D17_CO2_strat_permil"] * co2_lower_major_export_flux - co2_flux)
        / processed_parallel_d17
    )
    processed_parallel_flux_for_young_anomaly = max(processed_parallel_flux_for_young_anomaly, 0.0)
    processed_parallel_flux_for_young_fixed_d17 = max(processed_parallel_flux_for_young_fixed_d17, 0.0)
    processed_parallel_total_anomaly_flux = (
        co2_flux + processed_parallel_flux_for_young_anomaly * processed_parallel_d17
    )
    processed_parallel_total_major_flux = co2_lower_major_export_flux + processed_parallel_flux_for_young_anomaly
    column_processed = ColumnProcessedFraction(
        upper_activation=YOUNG_COLUMN_PROCESSED_UPPER_ACTIVATION,
    )
    column_processed_fraction = column_processed.processed_fraction
    column_processed_anomaly_flux = column_processed.additional_major_anomaly_flux(
        co2_lower_major_export_flux,
        co2_export_d17,
        processed_parallel_d17,
    )
    column_processed_extra_major_d17 = column_processed.additional_major_mixed_d17o(
        co2_export_d17,
        processed_parallel_d17,
    )
    column_processed_fixed_d17 = column_processed.fixed_reported_mixed_d17o(
        co2_export_d17,
        processed_parallel_d17,
    )
    outputs = {
        "converged": result.converged,
        "domain_status": domain["status"],
        "domain_reasons": domain["reason_text"],
        "domain_reason_list": domain["reasons"],
        "residual_norm": result.residual_norm,
        "max_atmosphere_residual_per_year": result.residual_norm,
        "effective_rp_o2_mol_per_year": rp,
        **gpp_metadata_from_scenario(scenario, config, rp),
        "effective_rp_relative_to_modern": rp / (PARAMETERS["k_respiration_per_year"] * 3.80e19),
        "effective_o2_budget_required_rp_o2_mol_per_year": rp,
        "effective_o2_budget_net_primary_o2_mol_per_year": "" if o2_budget is None else o2_budget.net_primary_o2,
        "effective_o2_budget_npp_relative_to_modern_gpp": "" if o2_budget is None else o2_budget.npp_relative_to_modern_gpp,
        "effective_o2_budget_photosynthesis_feasible": bool(rp > 0.0),
        "effective_o2_budget_closure_mode": "" if o2_budget is None else o2_budget.closure_mode,
        "pO2_percent_young_table3_modern": o2_percent.young_percent_o2,
        "pO2_percent_mills_modern": o2_percent.mills_percent_o2,
        "O2_trop_mol": result.y[idx["O2_trop"]],
        "CO2_trop_mol": result.y[idx["CO2_trop"]],
        "O2_trop_D17O_permil": summaries["O2_trop"].cap_delta17,
        "O2_trop_d18_prime_permil": summaries["O2_trop"].delta18_prime,
        "O2_trop_d17_prime_permil": summaries["O2_trop"].delta17_prime,
        "CO2_trop_D17O_permil": co2_trop_d17,
        "CO2_strat_D17O_permil": co2_strat_d17,
        "CO2_lower_D17O_permil": co2_export_d17,
        "CO2_export_D17O_permil": co2_export_d17,
        "CO2_export_upper_survival_fraction": lower_box.upper_survival_fraction,
        "CO2_export_signature_mode": "explicit_lower_box",
        "CO2_export_coupled_transport": True,
        "O3_strat_D17O_permil": summaries["O3_strat"].cap_delta17,
        "CO2_strat_D17O_flux_one_box_permil_mol_per_year": co2_flux_one_box,
        "CO2_strat_D17O_flux_permil_mol_per_year": co2_flux,
        "CO2_lower_major_export_flux_mol_per_year": co2_lower_major_export_flux,
        "processed_parallel_CO2_D17O_permil": processed_parallel_d17,
        "processed_parallel_flux_for_young_anomaly_mol_per_year": processed_parallel_flux_for_young_anomaly,
        "processed_parallel_fraction_for_young_anomaly": (
            processed_parallel_flux_for_young_anomaly / co2_lower_major_export_flux
            if co2_lower_major_export_flux
            else math.nan
        ),
        "processed_parallel_mixed_D17O_with_extra_major_flux_permil": (
            processed_parallel_total_anomaly_flux / processed_parallel_total_major_flux
            if processed_parallel_total_major_flux
            else math.nan
        ),
        "processed_parallel_flux_for_young_fixed_reported_D17O_mol_per_year": processed_parallel_flux_for_young_fixed_d17,
        "processed_parallel_fraction_for_young_fixed_reported_D17O": (
            processed_parallel_flux_for_young_fixed_d17 / co2_lower_major_export_flux
            if co2_lower_major_export_flux
            else math.nan
        ),
        "column_processed_signature_mode": "young_column_25_40km_activation",
        "column_processed_upper_activation": column_processed.upper_activation,
        "column_processed_upper_column_fraction": column_processed.upper_column_fraction,
        "column_processed_fraction": column_processed_fraction,
        "column_processed_CO2_D17O_permil": processed_parallel_d17,
        "column_processed_flux_mol_per_year": co2_lower_major_export_flux * column_processed_fraction,
        "column_processed_anomaly_flux_permil_mol_per_year": column_processed_anomaly_flux,
        "column_processed_flux_ratio_to_young": (
            column_processed_anomaly_flux / TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"]
        ),
        "column_processed_mixed_D17O_fixed_reported_permil": column_processed_fixed_d17,
        "column_processed_mixed_D17O_extra_major_permil": column_processed_extra_major_d17,
        "a_mif": PARAMETERS["a_MIF"] if config.a_mif is None else config.a_mif,
        "co2_sink_factor": config.co2_sink_factor,
        "r8c_factor": config.r8c_factor,
        "r7c_factor": config.r7c_factor,
        "r7g_factor": config.r7g_factor,
        "explicit_lower_box_effective_r7c_factor": config.r7c_factor * source_weight,
        "explicit_lower_box_effective_r7g_factor": config.r7g_factor * source_weight,
        "r8_rate_factor": config.r8_rate_factor,
        "co2_ocean_infusion_factor": config.co2_ocean_infusion_factor,
        "r7_transfer_damping_amplitude": config.r7_transfer_damping_amplitude,
        "r7_transfer_damping_exponent": config.r7_transfer_damping_exponent,
        "r7_transfer_damping_reference_ppm": config.r7_transfer_damping_reference_ppm,
        "r7_transfer_damping_gpp_power": config.r7_transfer_damping_gpp_power,
        "r7_transfer_damping_min_efficiency": config.r7_transfer_damping_min_efficiency,
        "r7_transfer_damping_shape": config.r7_transfer_damping_shape,
        "r7_vertical_gate_half_ppm": config.r7_vertical_gate_half_ppm,
        "r7_vertical_gate_hill": config.r7_vertical_gate_hill,
        "r7_two_box_lower_half_ppm": config.r7_two_box_lower_half_ppm,
        "r7_two_box_upper_half_ppm": config.r7_two_box_upper_half_ppm,
        "r7_two_box_upper_source_fraction": config.r7_two_box_upper_source_fraction,
        "r7_two_box_upper_export_efficiency": config.r7_two_box_upper_export_efficiency,
        "r7_two_box_normalize_modern": config.r7_two_box_normalize_modern,
        "r7_transfer_enhancement_gain": config.r7_transfer_enhancement_gain,
        "r7_transfer_enhancement_statistical_fraction": config.r7_transfer_enhancement_statistical_fraction,
        "r7_transfer_enhancement_exchange_time_yr": config.r7_transfer_enhancement_exchange_time_yr,
        "r7_transfer_enhancement_exposure_time_yr": config.r7_transfer_enhancement_exposure_time_yr,
        "r7_transfer_enhancement_half_ppm": config.r7_transfer_enhancement_half_ppm,
        "r7_transfer_enhancement_exposure_mode": config.r7_transfer_enhancement_exposure_mode,
        "r7_transfer_enhancement_exposure_power": config.r7_transfer_enhancement_exposure_power,
        "r7_transfer_enhancement_branch_group": config.r7_transfer_enhancement_branch_group,
        "r7_transfer_enhancement_full_atmosphere_only": config.r7_transfer_enhancement_full_atmosphere_only,
        "r7_full_atmosphere_enhancement_statistical_fraction": config.r7_full_atmosphere_enhancement_statistical_fraction,
        "photo_o17_source_law": config.photo_o17_source_law,
        "processed_column_source_d17o_permil": config.processed_column_source_d17o_permil,
        "processed_column_transition_tau_yr": config.processed_column_transition_tau_yr,
        "processed_column_recovery_tau_yr": config.processed_column_recovery_tau_yr,
        "processed_column_gate_mode": config.processed_column_gate_mode,
        "processed_column_gate_half_ppm": config.processed_column_gate_half_ppm,
        "processed_column_gate_hill": config.processed_column_gate_hill,
        "processed_column_gpp_power": config.processed_column_gpp_power,
        "processed_column_transition_activation": config.processed_column_transition_activation,
        "processed_column_shared_tail_activation": config.processed_column_shared_tail_activation,
        "processed_column_shared_tail_po2_fraction": config.processed_column_shared_tail_po2_fraction,
        "processed_column_shared_tail_po2_power": config.processed_column_shared_tail_po2_power,
        "processed_column_low_gpp_tail_activation": config.processed_column_low_gpp_tail_activation,
        "processed_column_low_gpp_tail_po2_fraction": config.processed_column_low_gpp_tail_po2_fraction,
        "processed_column_low_gpp_tail_po2_power": config.processed_column_low_gpp_tail_po2_power,
        "processed_column_low_gpp_transition_activation": config.processed_column_low_gpp_transition_activation,
        "processed_column_low_gpp_low_po2_transition_activation": (
            config.processed_column_low_gpp_low_po2_transition_activation
        ),
        "processed_column_low_gpp_recovery_activation": config.processed_column_low_gpp_recovery_activation,
        "processed_column_low_gpp_recovery_po2_mode": config.processed_column_low_gpp_recovery_po2_mode,
        "processed_column_low_gpp_recovery_po2_power": config.processed_column_low_gpp_recovery_po2_power,
        "processed_column_low_gpp_recovery_po2_half_pal": config.processed_column_low_gpp_recovery_po2_half_pal,
        "processed_column_low_gpp_recovery_po2_hill": config.processed_column_low_gpp_recovery_po2_hill,
        "processed_access_reservoir_mode": config.processed_access_reservoir_mode,
        "processed_access_reservoir_tau_yr": config.processed_access_reservoir_tau_yr,
        "processed_access_reservoir_initial": config.processed_access_reservoir_initial,
        "processed_access_reservoir_half_pal": config.processed_access_reservoir_half_pal,
        "processed_access_reservoir_hill": config.processed_access_reservoir_hill,
        "explicit_lower_box_mode": config.explicit_lower_box_mode,
        "explicit_lower_box_upper_mode": config.explicit_lower_box_upper_mode,
        "explicit_lower_box_lower_to_trop_rate_per_year": config.explicit_lower_box_lower_to_trop_rate_per_year,
        "explicit_lower_box_net_export_rate_per_year": lower_box.effective_net_export_rate_per_year,
        "explicit_lower_box_net_export_fraction_of_internal_exchange": (
            lower_box.effective_net_export_rate_per_year / lower_box.lower_to_trop_rate_per_year
            if lower_box.lower_to_trop_rate_per_year
            else math.nan
        ),
        "explicit_lower_box_lower_major_scale": config.explicit_lower_box_lower_major_scale,
        "explicit_lower_box_source_weight_mode": config.explicit_lower_box_source_weight_mode,
        "explicit_lower_box_source_weight": source_weight,
        "explicit_lower_box_source_weight_boost": config.explicit_lower_box_source_weight_boost,
        "explicit_lower_box_source_weight_half_ppm": config.explicit_lower_box_source_weight_half_ppm,
        "explicit_lower_box_source_weight_hill": config.explicit_lower_box_source_weight_hill,
        "explicit_lower_box_CO2_lower_mol": result.y[idx["CO2_lower"]],
        "explicit_lower_box_CO18O_lower_mol": result.y[idx["CO18O_lower"]],
        "explicit_lower_box_CO17O_lower_mol": result.y[idx["CO17O_lower"]],
        "explicit_lower_box_base_species_count": len(SPECIES_ORDER),
        "explicit_lower_box_species_count": len(EXTENDED_SPECIES_ORDER),
    }
    apply_o2_d17o_output_calibration(scenario, outputs)
    return ScenarioResult(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),
        input=asdict(scenario),
        config=asdict(config),
        outputs=outputs,
        warnings=scenario_warnings(config, result.converged),
    )


# Literature anchors for the O(1D) Delta'17O, which sets the O2 Delta'17O
# saturation cap (Young et al. 2014, Section 6). aMIF is the ozone mass-
# independent fractionation amplitude that produces each O(1D) anomaly.
A_MIF_YOUNG = 1.065  # O(1D) Delta'17O ~ 27 permil (Young default)
A_MIF_YOUNG_BARKAN_LUZ_TARGET = 1.109  # Young sensitivity adjusted to the Barkan & Luz air-O2 target
A_MIF_BARKAN_LUZ = A_MIF_YOUNG_BARKAN_LUZ_TARGET  # Backward-compatible alias


# Physical saturation cap = -(O(1D) Delta'17O): the most negative O2 Delta'17O
# reachable at any GPP/pCO2 for each anchor (Young et al. 2014, Section 6).
O2_CAP_YOUNG_PERMIL = -27.6
O2_CAP_BARKAN_LUZ_PERMIL = -46.6


def _o2_d17o_at(scenario: ScenarioInput, *, a_mif: float) -> float:
    """Resolved-chemistry O2 Delta'17O at a given aMIF, or NaN on failure."""
    try:
        run = _run_scenario_core(
            replace(scenario, a_mif=a_mif, report_extrapolation_bounds=False)
        )
        return float(run.outputs["O2_trop_D17O_permil"])
    except Exception:  # a bound is a diagnostic; never break the primary run
        return float("nan")


def _attach_extrapolation_bounds(
    scenario: ScenarioInput, primary: ScenarioResult
) -> ScenarioResult:
    """Add a literature-bounded O2 Delta'17O interval.

    The estimate (``O2_trop_D17O_permil``) is unchanged. We add one thing: the
    same state re-evaluated for Young's basic O(1D) Delta'17O value (27 permil)
    and Young's 45.9 permil sensitivity case adjusted to the Barkan & Luz air-O2
    target. Young Section 6 identifies
    the O(1D) composition as "the ultimate driver" and most uncertain input, so
    this is a structural sensitivity envelope, not a statistical confidence
    interval. The envelope is tight near modern
    and widens toward extreme states (high pCO2 / low GPP / low pO2) - exactly
    where a single number should not be trusted. ``bound_low`` is the more
    negative (Barkan-Luz) end; ``bound_high`` is the less negative (Young) end.
    """
    high = _o2_d17o_at(scenario, a_mif=A_MIF_YOUNG)
    low = _o2_d17o_at(scenario, a_mif=A_MIF_BARKAN_LUZ)

    finite = [v for v in (low, high) if v == v]
    band_low = min(finite) if finite else float("nan")  # more negative (Barkan-Luz)
    band_high = max(finite) if finite else float("nan")  # less negative (Young, = estimate)

    extra = {
        "O2_trop_D17O_bound_low_permil": band_low,
        "O2_trop_D17O_bound_high_permil": band_high,
        "O2_trop_D17O_bound_spread_permil": (band_high - band_low) if finite else float("nan"),
        "O2_trop_D17O_bound_basis": (
            "Young basic O(1D) Delta'17O 27 permil to Young 45.9 permil sensitivity case "
            "adjusted to the Barkan & Luz air-O2 target"
        ),
        "O2_trop_D17O_bounds_note": (
            "Estimate (= bound_high) uses Young's preferred O(1D) Delta'17O (27 permil). bound_low is "
            "the same state in Young's 45.9 permil sensitivity case adjusted to match the Barkan & Luz "
            "air-O2 target. This is a structural sensitivity envelope, not a statistical confidence interval. "
            "bound_low is a conservative scenario, not an equal alternative: "
            "Young notes the 45.9 permil anchor makes the modern CO2 isoflux ~4x the observed value. "
            f"O2 Delta'17O cannot pass the stratospheric saturation cap (~{O2_CAP_YOUNG_PERMIL:.0f} permil "
            f"Young / ~{O2_CAP_BARKAN_LUZ_PERMIL:.0f} permil Barkan-Luz), approached only at very low GPP "
            "and/or high pCO2."
        ),
    }
    merged = dict(primary.outputs)
    merged.update(extra)
    return replace(primary, outputs=merged)


def _attach_conventions(result: ScenarioResult) -> ScenarioResult:
    """Record the active literature conventions (and their citations) in outputs."""
    resolved_parameters = dict(result.config)
    for field, value in tuple(resolved_parameters.items()):
        if value is None and field in result.outputs:
            resolved_parameters[field] = result.outputs[field]
    rows = conventions_metadata_for_parameters(resolved_parameters)
    merged = dict(result.outputs)
    merged["active_conventions"] = {r["group"]: r["convention"] for r in rows}
    merged["active_conventions_metadata"] = rows
    assumption_fields = (
        "co2_sink_factor",
        "r8c_factor",
        "photo_o17_source_law",
        "processed_access_reservoir_mode",
        "explicit_lower_box_mode",
        "explicit_lower_box_upper_mode",
        "explicit_lower_box_lower_to_trop_rate_per_year",
        "explicit_lower_box_net_export_rate_per_year",
        "alpha_respiration_18",
        "beta_respiration_17",
        "evapotranspiration_alpha_18",
        "evapotranspiration_beta_17",
    )
    merged["active_model_assumptions"] = {
        field: resolved_parameters.get(field) for field in assumption_fields
    }
    merged["active_conventions_citations"] = "; ".join(
        f"{r['group']}={r['convention']} ({r['citation']})" for r in rows
    )
    return replace(result, outputs=merged)


def run_scenario(scenario: ScenarioInput) -> ScenarioResult:
    result = _run_scenario_core(scenario)
    result = _attach_conventions(result)
    if scenario.report_extrapolation_bounds:
        result = _attach_extrapolation_bounds(scenario, result)
    return result


def _run_scenario_core(scenario: ScenarioInput) -> ScenarioResult:
    config = config_from_scenario(scenario)
    if config.explicit_lower_box_mode != "none":
        return run_explicit_lower_box_scenario(scenario, config)
    run = run_model(config)
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    rp = effective_rp_o2(config)
    o2_budget = effective_o2_budget_terms(config)
    domain = scenario_domain(config, run.full_result.converged)
    o2_percent = percent_equivalent(config.p_o2_pal)
    co2_trop_d17 = run.summaries["CO2_trop"].cap_delta17
    co2_strat_d17 = run.summaries["CO2_strat"].cap_delta17
    co2_export_d17, co2_export_fraction = co2_export_signature(config, co2_trop_d17, co2_strat_d17)
    co2_flux_one_box = (
        PARAMETERS["k_ST_per_year"]
        * run.y[idx["CO2_strat"]]
        * co2_strat_d17
    )
    co2_flux = (
        PARAMETERS["k_ST_per_year"]
        * run.y[idx["CO2_strat"]]
        * co2_export_d17
    )
    if config.r7_transfer_enhancement_statistical_fraction is not None:
        r7_statistical_fraction = config.r7_transfer_enhancement_statistical_fraction
    elif (
        config.r7_transfer_enhancement_exchange_time_yr is not None
        and config.r7_transfer_enhancement_exposure_time_yr is not None
    ):
        r7_statistical_fraction = 1.0 - math.exp(
            -config.r7_transfer_enhancement_exposure_time_yr
            / max(config.r7_transfer_enhancement_exchange_time_yr, 1.0e-30)
        )
    else:
        r7_statistical_fraction = None
    r7_effective_gain = (
        config.r7_transfer_enhancement_gain
        if r7_statistical_fraction is None
        else r7_statistical_fraction / 3.0
    )
    outputs = {
        "converged": run.full_result.converged,
        "domain_status": domain["status"],
        "domain_reasons": domain["reason_text"],
        "domain_reason_list": domain["reasons"],
        "residual_norm": run.full_result.residual_norm,
        "max_atmosphere_residual_per_year": run.max_atmosphere_residual_per_year,
        "effective_rp_o2_mol_per_year": rp,
        **gpp_metadata_from_scenario(scenario, config, rp),
        "effective_rp_relative_to_modern": rp / (PARAMETERS["k_respiration_per_year"] * 3.80e19),
        "effective_o2_budget_required_rp_o2_mol_per_year": rp,
        "effective_o2_budget_net_primary_o2_mol_per_year": "" if o2_budget is None else o2_budget.net_primary_o2,
        "effective_o2_budget_npp_relative_to_modern_gpp": "" if o2_budget is None else o2_budget.npp_relative_to_modern_gpp,
        "effective_o2_budget_photosynthesis_feasible": bool(rp > 0.0),
        "effective_o2_budget_closure_mode": "" if o2_budget is None else o2_budget.closure_mode,
        "pO2_percent_young_table3_modern": o2_percent.young_percent_o2,
        "pO2_percent_mills_modern": o2_percent.mills_percent_o2,
        "O2_trop_mol": run.y[idx["O2_trop"]],
        "CO2_trop_mol": run.y[idx["CO2_trop"]],
        "O2_trop_D17O_permil": run.summaries["O2_trop"].cap_delta17,
        "O2_trop_d18_prime_permil": run.summaries["O2_trop"].delta18_prime,
        "O2_trop_d17_prime_permil": run.summaries["O2_trop"].delta17_prime,
        "CO2_trop_D17O_permil": co2_trop_d17,
        "CO2_strat_D17O_permil": co2_strat_d17,
        "CO2_export_D17O_permil": co2_export_d17,
        "CO2_export_upper_survival_fraction": co2_export_fraction,
        "CO2_export_signature_mode": config.co2_export_signature_mode,
        "CO2_export_coupled_transport": config.co2_export_coupled_transport,
        "O3_strat_D17O_permil": run.summaries["O3_strat"].cap_delta17,
        "CO2_strat_D17O_flux_one_box_permil_mol_per_year": co2_flux_one_box,
        "CO2_strat_D17O_flux_permil_mol_per_year": co2_flux,
        "a_mif": PARAMETERS["a_MIF"] if config.a_mif is None else config.a_mif,
        "co2_sink_factor": config.co2_sink_factor,
        "r8c_factor": config.r8c_factor,
        "r7c_factor": config.r7c_factor,
        "r7g_factor": config.r7g_factor,
        "r8_rate_factor": config.r8_rate_factor,
        "co2_ocean_infusion_factor": config.co2_ocean_infusion_factor,
        "r7_transfer_damping_amplitude": config.r7_transfer_damping_amplitude,
        "r7_transfer_damping_exponent": config.r7_transfer_damping_exponent,
        "r7_transfer_damping_reference_ppm": config.r7_transfer_damping_reference_ppm,
        "r7_transfer_damping_gpp_power": config.r7_transfer_damping_gpp_power,
        "r7_transfer_damping_min_efficiency": config.r7_transfer_damping_min_efficiency,
        "r7_transfer_damping_shape": config.r7_transfer_damping_shape,
        "r7_vertical_gate_half_ppm": config.r7_vertical_gate_half_ppm,
        "r7_vertical_gate_hill": config.r7_vertical_gate_hill,
        "r7_two_box_lower_half_ppm": config.r7_two_box_lower_half_ppm,
        "r7_two_box_upper_half_ppm": config.r7_two_box_upper_half_ppm,
        "r7_two_box_upper_source_fraction": config.r7_two_box_upper_source_fraction,
        "r7_two_box_upper_export_efficiency": config.r7_two_box_upper_export_efficiency,
        "r7_two_box_normalize_modern": config.r7_two_box_normalize_modern,
        "r7_transfer_enhancement_gain": r7_effective_gain,
        "r7_transfer_enhancement_statistical_fraction": r7_statistical_fraction,
        "r7_transfer_enhancement_exchange_time_yr": config.r7_transfer_enhancement_exchange_time_yr,
        "r7_transfer_enhancement_exposure_time_yr": config.r7_transfer_enhancement_exposure_time_yr,
        "r7_transfer_enhancement_half_ppm": config.r7_transfer_enhancement_half_ppm,
        "r7_transfer_enhancement_exposure_mode": config.r7_transfer_enhancement_exposure_mode,
        "r7_transfer_enhancement_exposure_power": config.r7_transfer_enhancement_exposure_power,
        "r7_transfer_enhancement_branch_group": config.r7_transfer_enhancement_branch_group,
        "r7_transfer_enhancement_full_atmosphere_only": config.r7_transfer_enhancement_full_atmosphere_only,
        "r7_full_atmosphere_enhancement_statistical_fraction": config.r7_full_atmosphere_enhancement_statistical_fraction,
        "photo_o17_source_law": config.photo_o17_source_law,
        "processed_column_source_d17o_permil": config.processed_column_source_d17o_permil,
        "processed_column_transition_tau_yr": config.processed_column_transition_tau_yr,
        "processed_column_recovery_tau_yr": config.processed_column_recovery_tau_yr,
        "processed_column_gate_mode": config.processed_column_gate_mode,
        "processed_column_gate_half_ppm": config.processed_column_gate_half_ppm,
        "processed_column_gate_hill": config.processed_column_gate_hill,
        "processed_column_gpp_power": config.processed_column_gpp_power,
        "processed_column_transition_activation": config.processed_column_transition_activation,
        "processed_column_shared_tail_activation": config.processed_column_shared_tail_activation,
        "processed_column_shared_tail_po2_fraction": config.processed_column_shared_tail_po2_fraction,
        "processed_column_shared_tail_po2_power": config.processed_column_shared_tail_po2_power,
        "processed_column_low_gpp_tail_activation": config.processed_column_low_gpp_tail_activation,
        "processed_column_low_gpp_tail_po2_fraction": config.processed_column_low_gpp_tail_po2_fraction,
        "processed_column_low_gpp_tail_po2_power": config.processed_column_low_gpp_tail_po2_power,
        "processed_column_low_gpp_transition_activation": config.processed_column_low_gpp_transition_activation,
        "processed_column_low_gpp_low_po2_transition_activation": (
            config.processed_column_low_gpp_low_po2_transition_activation
        ),
        "processed_column_low_gpp_recovery_activation": config.processed_column_low_gpp_recovery_activation,
        "processed_column_low_gpp_recovery_po2_mode": config.processed_column_low_gpp_recovery_po2_mode,
        "processed_column_low_gpp_recovery_po2_power": config.processed_column_low_gpp_recovery_po2_power,
        "processed_column_low_gpp_recovery_po2_half_pal": config.processed_column_low_gpp_recovery_po2_half_pal,
        "processed_column_low_gpp_recovery_po2_hill": config.processed_column_low_gpp_recovery_po2_hill,
        "processed_access_reservoir_mode": config.processed_access_reservoir_mode,
        "processed_access_reservoir_tau_yr": config.processed_access_reservoir_tau_yr,
        "processed_access_reservoir_initial": config.processed_access_reservoir_initial,
        "processed_access_reservoir_half_pal": config.processed_access_reservoir_half_pal,
        "processed_access_reservoir_hill": config.processed_access_reservoir_hill,
        "explicit_lower_box_mode": config.explicit_lower_box_mode,
        "explicit_lower_box_upper_mode": config.explicit_lower_box_upper_mode,
        "explicit_lower_box_lower_to_trop_rate_per_year": config.explicit_lower_box_lower_to_trop_rate_per_year,
        "explicit_lower_box_net_export_rate_per_year": config.explicit_lower_box_net_export_rate_per_year,
        "explicit_lower_box_lower_major_scale": config.explicit_lower_box_lower_major_scale,
    }
    apply_o2_d17o_output_calibration(scenario, outputs)
    return ScenarioResult(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),
        input=asdict(scenario),
        config=asdict(config),
        outputs=outputs,
        warnings=scenario_warnings(config, run.full_result.converged),
    )
