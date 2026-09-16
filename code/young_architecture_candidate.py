"""Named current young-like architecture candidate.

The model reconstruction still contains many audit scripts. This module is the
small stable bridge between those audits and the publication-facing scenario
layer: it names the current best architecture, records the parameters that are
being validated, and defines the scorecard thresholds that future edits should
preserve unless a better physically justified mechanism replaces them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
OUTPUTS = _PROJECT_OUTPUTS

YOUNG_LIKE_V0_NAME = "young_like_three_term_column_finite_exposure_v0"
YOUNG_LIKE_V1_NAME = "young_like_residence_split_column_finite_exposure_v1"
YOUNG_LIKE_V2_NAME = "young_like_residence_split_threshold_access_v2"
YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME = "young_like_v2_access_reservoir_tau1"
YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME = "young_like_v2_log_low_gpp_access_tau1"
YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME = "young_like_v2_log_low_gpp_tail_access_tau1"
YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME = "young_like_v2_log_low_gpp_smooth_tail_access_tau1"
YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME = "young_like_v2_log_low_gpp_balanced_tail_access_tau1"
YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME = "young_like_v2_access_reservoir_tau30"
YOUNG_LIKE_V2_PARALLEL_CO2_NAME = "young_like_v2_parallel_column_co2"
PREFERRED_REDUCED_YOUNG_LIKE_NAME = YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME
CURRENT_ARCHITECTURE_NAME = YOUNG_LIKE_V0_NAME
CURRENT_ARCHITECTURE_LABEL = "three-term processed-column + transient finite-exposure R7"


@dataclass(frozen=True)
class ThreeTermColumnSourceLaw:
    """Steady Fig. 7/Fig. 8 source/export law selected by the current audits."""

    processed_signature: str = "young_o1d"
    transition_tau_yr: float = 0.005
    recovery_tau_yr: float | None = None
    gate_half_ppm: float = 30000.0
    gate_hill: float = 8.0
    gpp_power: float = 2.0
    low_gpp_signal_mode: str = "current_unbounded"
    transition_activation: float = 0.36505933163054555
    shared_tail_activation: float = 0.37862472267326996
    shared_tail_po2_fraction: float = 0.0
    shared_tail_po2_power: float = 1.0
    low_gpp_tail_activation: float = 0.450695424847644
    low_gpp_tail_po2_fraction: float = 0.0
    low_gpp_tail_po2_power: float = 1.0
    low_gpp_recovery_activation: float = 0.0
    low_gpp_recovery_po2_mode: str = "power"
    low_gpp_recovery_po2_power: float = 0.0
    low_gpp_recovery_po2_half_pal: float = 0.75
    low_gpp_recovery_po2_hill: float = 8.0
    source: str = "processed_column_three_term_law.md"


@dataclass(frozen=True)
class FiniteExposureTransientLaw:
    """Transient-only Fig. 9/Fig. 10 finite-exposure R7 transfer law."""

    branch_group: str = "incoming_heavy"
    exchange_time_modern_yr: float = 3.0
    residence_exchange_ratio: float = 0.12
    max_excess: float = 1.0 / 3.0
    full_atmosphere_only: bool = True
    source: str = "three_term_with_finite_exposure_transients.md"

    @property
    def residence_time_yr(self) -> float:
        return self.exchange_time_modern_yr * self.residence_exchange_ratio


@dataclass(frozen=True)
class ScoreThreshold:
    """Pass/warning/fail limits for one validation metric."""

    metric: str
    pass_limit: float
    warning_limit: float
    units: str
    note: str


V0_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw()
V1_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw(
    recovery_tau_yr=0.317,
    gpp_power=1.5,
    transition_activation=0.3760265685364437,
    shared_tail_activation=0.39346876678063736,
    low_gpp_tail_activation=0.7394024248543923,
    low_gpp_recovery_activation=0.031750399757004166,
    source="processed_column_residence_split_midgap.md",
)
V2_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw(
    recovery_tau_yr=V1_STEADY_SOURCE_LAW.recovery_tau_yr,
    gpp_power=V1_STEADY_SOURCE_LAW.gpp_power,
    transition_activation=V1_STEADY_SOURCE_LAW.transition_activation,
    shared_tail_activation=V1_STEADY_SOURCE_LAW.shared_tail_activation,
    low_gpp_tail_activation=V1_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    low_gpp_recovery_activation=V1_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    low_gpp_recovery_po2_mode="threshold_access",
    low_gpp_recovery_po2_half_pal=0.85,
    low_gpp_recovery_po2_hill=8.0,
    source="v1_threshold_access_recovery_scan.md",
)
V2_LOG_LOW_GPP_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw(
    recovery_tau_yr=V1_STEADY_SOURCE_LAW.recovery_tau_yr,
    gpp_power=2.0,
    low_gpp_signal_mode="log1p",
    transition_activation=V1_STEADY_SOURCE_LAW.transition_activation,
    shared_tail_activation=V1_STEADY_SOURCE_LAW.shared_tail_activation,
    low_gpp_tail_activation=0.9752210552272155,
    low_gpp_recovery_activation=0.04187659833683818,
    low_gpp_recovery_po2_mode="threshold_access",
    low_gpp_recovery_po2_half_pal=0.85,
    low_gpp_recovery_po2_hill=8.0,
    source="bounded_low_gpp_source_law.md",
)
V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw(
    recovery_tau_yr=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.recovery_tau_yr,
    gpp_power=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.gpp_power,
    low_gpp_signal_mode=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_signal_mode,
    transition_activation=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.transition_activation,
    shared_tail_activation=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.shared_tail_activation,
    low_gpp_tail_activation=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    low_gpp_tail_po2_fraction=0.50,
    low_gpp_tail_po2_power=1.0,
    low_gpp_recovery_activation=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    low_gpp_recovery_po2_mode=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    low_gpp_recovery_po2_half_pal=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    low_gpp_recovery_po2_hill=V2_LOG_LOW_GPP_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    source="po2_tail_access_partition.md",
)
V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw(
    recovery_tau_yr=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.recovery_tau_yr,
    gpp_power=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.gpp_power,
    low_gpp_signal_mode=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_signal_mode,
    transition_activation=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.transition_activation,
    shared_tail_activation=1.20 * V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.shared_tail_activation,
    low_gpp_tail_activation=1.10 * V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    low_gpp_tail_po2_fraction=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_po2_fraction,
    low_gpp_tail_po2_power=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_po2_power,
    low_gpp_recovery_activation=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    low_gpp_recovery_po2_mode=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    low_gpp_recovery_po2_half_pal=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    low_gpp_recovery_po2_hill=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    gate_half_ppm=35000.0,
    gate_hill=4.0,
    source="high_pco2_gate_retune.md",
)
V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_STEADY_SOURCE_LAW = ThreeTermColumnSourceLaw(
    recovery_tau_yr=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.recovery_tau_yr,
    gpp_power=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.gpp_power,
    low_gpp_signal_mode=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_signal_mode,
    transition_activation=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.transition_activation,
    shared_tail_activation=0.90 * V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.shared_tail_activation,
    shared_tail_po2_fraction=1.0,
    shared_tail_po2_power=1.0,
    low_gpp_tail_activation=0.90 * V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_activation,
    low_gpp_tail_po2_fraction=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_po2_fraction,
    low_gpp_tail_po2_power=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_tail_po2_power,
    low_gpp_recovery_activation=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_activation,
    low_gpp_recovery_po2_mode=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_po2_mode,
    low_gpp_recovery_po2_half_pal=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_po2_half_pal,
    low_gpp_recovery_po2_hill=V2_LOG_LOW_GPP_TAIL_ACCESS_STEADY_SOURCE_LAW.low_gpp_recovery_po2_hill,
    gate_half_ppm=30000.0,
    gate_hill=6.0,
    source="gate_candidate_override_audit.md",
)
STEADY_SOURCE_LAW = V0_STEADY_SOURCE_LAW
TRANSIENT_LAW = FiniteExposureTransientLaw()
V2_TRANSIENT_LAW = FiniteExposureTransientLaw(residence_exchange_ratio=0.135, source="v1_threshold_access_recovery_scan.md")

SCORE_THRESHOLDS: tuple[ScoreThreshold, ...] = (
    ScoreThreshold("Fig. 7 digitized mean abs residual", 0.03, 0.05, "permil", "Dense Fig. 7 contour shape"),
    ScoreThreshold("Fig. 7 digitized max abs residual", 0.08, 0.12, "permil", "Worst Fig. 7 contour residual"),
    ScoreThreshold("Fig. 8 digitized mean abs residual", 0.05, 0.10, "permil", "Main steady high-pCO2 benchmark"),
    ScoreThreshold("Fig. 8 digitized max abs residual", 0.15, 0.25, "permil", "Worst Fig. 8 benchmark point"),
    ScoreThreshold("Fig. 8 50% GPP worst residual", 0.10, 0.15, "permil", "Remaining low-GPP warning"),
    ScoreThreshold("Fig. 9 visual minimum residual", 0.03, 0.08, "permil", "Half-photosynthesis transient dip"),
    ScoreThreshold("Fig. 9 final residual", 0.02, 0.05, "permil", "Half-photosynthesis final state"),
    ScoreThreshold("Fig. 9 peak pCO2 residual", 150.0, 300.0, "ppm", "CO2 response to half photosynthesis"),
    ScoreThreshold("Fig. 10 final shift residual", 0.01, 0.03, "permil", "CO2-step final O2 isotope shift"),
    ScoreThreshold("Finite-exposure ratio vs Liang local", 0.03, 0.08, "ratio", "Independent exposure-time check"),
)

PREREQUISITE_OUTPUTS: tuple[Path, ...] = (
    OUTPUTS / "processed_column_three_term_steady_validation_rows.csv",
    OUTPUTS / "processed_column_three_term_steady_validation_summary.csv",
    OUTPUTS / "three_term_with_finite_exposure_transients_summary.csv",
    OUTPUTS / "three_term_with_finite_exposure_transients_timeseries.csv",
    OUTPUTS / "finite_exposure_literature_constraint.csv",
)


def missing_prerequisite_outputs() -> tuple[Path, ...]:
    return tuple(path for path in PREREQUISITE_OUTPUTS if not path.exists())


def architecture_metadata() -> dict[str, object]:
    return {
        "name": CURRENT_ARCHITECTURE_NAME,
        "label": CURRENT_ARCHITECTURE_LABEL,
        "preferred_reduced_young_like_name": PREFERRED_REDUCED_YOUNG_LIKE_NAME,
        "steady_source_law": STEADY_SOURCE_LAW.__dict__,
        "transient_law": {
            **TRANSIENT_LAW.__dict__,
            "residence_time_yr": TRANSIENT_LAW.residence_time_yr,
        },
        "score_thresholds": [threshold.__dict__ for threshold in SCORE_THRESHOLDS],
        "prerequisite_outputs": [str(path) for path in PREREQUISITE_OUTPUTS],
    }
