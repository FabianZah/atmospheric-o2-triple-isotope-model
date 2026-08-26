"""Lightweight checks for core model-rebuild utilities."""

from __future__ import annotations

# --- path bootstrap (direct execution) ---
import sys as _sys
from pathlib import Path as _Path
_root = next((p for p in _Path(__file__).resolve().parents if (p / ".project-root").exists()), None)
if _root is not None:
    for _sub in ("code", "validation"):
        _p = str(_root / _sub)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
# --- end path bootstrap ---
import math
from types import SimpleNamespace

import numpy as np

from integrate_fig9_fig10_transients import run_experiments

from conservative_column_transport import (
    AtmosphericLayer,
    ConservativeColumn,
    ExchangeInterface,
    young_two_layer_column,
)
from conservative_circulation_transport import (
    ConservativeCirculationNetwork,
    DirectedAirFlux,
    circulation_network_from_streamfunction,
    circulation_network_with_lower_reservoir_row,
    combined_transport_matrix_per_year,
    latitude_vertical_cell_name,
)
from convective_plume_transport import (
    ConvectivePlumeColumn,
    ConvectivePlumeGrid,
)
from vertical_column_profile import (
    LIANG_2006_DOMAIN,
    LIANG_2008_CORE_DOMAIN,
    LIANG_2008_EXTENDED_DOMAIN,
    YOUNG_BULK_STRATOSPHERE,
    ValidatedVerticalProfile,
    VerticalCell,
    published_vertical_references,
)
from photochem_profile import eddy_diffusion_column
from benchmark_photochem_transport import mean_first_passage_years
from era5_kyy_reference import (
    EARTH_ROTATION_PER_S,
    quasigeostrophic_pv_gradient_per_m_s,
)
from era5_tem_reference import Era5TemClimatology
from era5_convection_reference import (
    calendar_month_retrieval_jobs,
    ERA5_CONVECTION_PARAMETERS,
    global_cell_areas_from_centers_m2,
    l137_geopotential_and_thickness,
    l137_half_level_pressure_pa,
    l137_hybrid_coefficients,
    month_starts,
    forecast_monthly_mean_valid_time,
    monthly_retrieval_jobs,
    regular_latitude_longitude_edges,
    spherical_cell_areas_m2,
    zonal_detrainment_to_mol_per_year,
    zonal_mass_flux_to_mol_per_year,
)
from era5_convection_adapter import (
    Era5ZonalConvectionMonth,
    aggregate_plume_grid_to_two_reservoirs,
    mean_era5_zonal_convection_months,
)
from era5_convection_climatology import (
    Era5ConvectionClimatology,
    load_era5_convection_climatology,
)
from summers_vertical_mixing import (
    GAMMA1_S_PER_M2,
    PRANDTL_NUMBER,
    SOURCE_ALTITUDE_M,
    SUMMERS_TABLE1_WAVES,
    VERTICAL_SPACING_M,
    GravityWave,
    _wave_kzz_column,
    summers_kzz_on_native_grid,
    zero_phase_drag_m_per_s2,
)
from passive_age_of_air import (
    build_passive_age_transport,
    select_passive_transport_components,
    select_vertical_diffusion_altitude_domain,
    select_vertical_diffusion_below_local_tropopause,
    solve_passive_mean_age,
)
from finite_clock_tracer import (
    propagate_boundary_history,
    solve_finite_clock_age,
    solve_periodic_clock_age,
)
from e90_tracer import (
    E90_LIFETIME_YEARS,
    solve_e90_periodic,
    solve_e90_steady,
)
from gridded_isotope_transport import GriddedSpeciesSystem
from gridded_oxygen_chemistry import (
    ATMOSPHERIC_ATOM_COUNTS,
    ATMOSPHERIC_OXYGEN_SPECIES,
    ElementaryGridReaction,
    local_reaction_tendency_mol_per_year,
    reaction_atom_residual,
    validate_atom_balanced_reactions,
)
from meridional_transport_reference import (
    JIANG_2004_400K_FLUX,
    MORGAN_2004_CONTINUITY_TERMS,
    MORGAN_2004_GRID,
    SHIA_1989_TROPOPAUSE_FLUX,
    hydrostatic_air_moles,
)
from merra2_transport_reference import MERRA2_DAILY_ARCHIVES
from meridional_diffusion_transport import meridional_eddy_diffusion_operator
from isotopes import cap_delta17_from_primes, collision_frequency_alpha, reduced_mass
from earth_history_envelopes import pco2_envelope_status
from gpp_normalization import (
    BEERLING_1999_MODERN_GPP_PGC_PER_YEAR,
    DEFAULT_GPP_NORMALIZATION,
    YOUNG_MODERN_GPP_PGC_PER_YEAR,
)
from phanerozoic_o2 import (
    MODEL_PO2_WORKING_MAX_PAL,
    MODEL_PO2_WORKING_MIN_PAL,
    pal_to_percent,
    percent_to_pal,
)
from model_runner import isotope_summaries, scaled_table3_state
from model_scenarios import (
    ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
    O2_CAP_BARKAN_LUZ_PERMIL,
    PACK2021_BETA_RESPIRATION_17_CANDIDATE,
    PACK2021_EVAPOTRANSPIRATION_BETA_17_CANDIDATE,
    PACK2021_EXPLICIT_EXPORT_A_MIF,
    PACK2021_O2_D17O_TARGET_PERMIL,
    PACK2021_VALIDATED_YOUNG_EXPLICIT_EXPORT_A_MIF,
    CURRENT_YOUNG_REPRODUCTION_PRESET,
    ScenarioInput,
    UPDATED_VALIDATED_YOUNG_BETA_PACK_PRESET,
    UPDATED_VALIDATED_YOUNG_LOCAL_PACK_PRESET,
    UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET,
    config_from_scenario,
    explicit_lower_box_upper_fraction,
    preset_names,
    preset_role,
    run_scenario,
)
from modern_validation import modern_validation_rows, permil_mol_per_year_to_permil_pgc_per_year
from mass_balance_model import solve_tropospheric_o2_mass_balance
from explicit_lower_co2_box_model import LowerBoxConfig
from explicit_processed_co2_box_model import (
    PROCESSED_SPECIES_ORDER,
    ProcessedBoxConfig,
    solve_processed_box_fixed_reservoir,
    summarize_processed_state,
)
from processed_altitude_reservoir import (
    CO2IsotopeSignature,
    ColumnProcessedFraction,
    O1DIsotopeTransferSource,
    ProcessedSignatureSource,
    apply_processed_reservoir_mix,
    finite_exposure_activation,
    joint_required_fraction,
    mix_co2_isotope_signatures,
    required_fraction_isotope_resolved,
)
from reactions import Reaction, derivative
from solve_fast_stratosphere import solve_fast_species
from solve_fixed_reservoir_isotopes import solve_fixed_reservoir_isotopes
from solve_tropospheric_co2_isotopes import solve_co2_isotopes
from solve_coupled_isotope_subsystem import solve_subsystem
from score_scenario_presets import dynamic_residual_rows, scalar_residual_rows, score_preset, summarize
from spherule_to_air_d17o import (
    analytical_air_d17o_sigma,
    air_d17o_from_spherule,
    independent_calibration_sensitivity_envelope,
)
from table3_state import TABLE3_MOLES
from young_model_inventory import (
    TABLE3_FLUX_TARGETS,
    TABLE3_ISOTOPE_TARGETS,
    TABLE3_MOLE_FRACTION_TARGETS,
    TABLE3_MOLE_TARGETS,
    TABLE3_TARGETS,
)
from young_validation_targets import TABLE3_SCALAR_TARGETS
from young_bulk_fig10 import integrate_young_bulk_fig10
from young_reactions import REACTION_RECORDS, executable_reactions, md_alpha, r8_co2_h2o_exchange_reactions
from audit_young_remaining_table2_rows import EXPECTED_ROWS as REMAINING_TABLE2_ROWS, audit_rows as audit_remaining_table2_rows
from audit_young_r4_r6_r7_rows import EXPECTED_ROWS as R4_R6_R7_ROWS
from audit_current_architecture_scorecard import threshold_status
from young_architecture_candidate import (
    CURRENT_ARCHITECTURE_NAME,
    PREFERRED_REDUCED_YOUNG_LIKE_NAME,
    SCORE_THRESHOLDS,
    STEADY_SOURCE_LAW,
    TRANSIENT_LAW,
    YOUNG_LIKE_V0_NAME,
    YOUNG_LIKE_V1_NAME,
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_PARALLEL_CO2_NAME,
    YOUNG_LIKE_V2_NAME,
    architecture_metadata,
)


def check_close(name: str, got: float, expected: float, tol: float) -> None:
    if not math.isfinite(got) or abs(got - expected) > tol:
        raise AssertionError(f"{name}: got {got}, expected {expected} ± {tol}")


def test_young_table3_delta() -> None:
    # Zero-intercept 0.528 line gives -0.369 per mil from the printed d-prime
    # values. Young Table 3 reports -0.410 per mil, implying a reference-frame
    # intercept of about -0.041 per mil for this tabulated D17O value.
    check_close("Table 3 O2 troposphere D17O zero-intercept", cap_delta17_from_primes(11.887, 23.212), -0.369, 0.002)
    check_close("Table 3 O2 troposphere D17O with inferred intercept", cap_delta17_from_primes(11.887, 23.212, intercept=-0.041), -0.410, 0.002)
    check_close("Table 3 O2 troposphere D17O with gamma", cap_delta17_from_primes(11.887, 23.212, gamma=0.041), -0.410, 0.002)


def test_exported_conventions_match_resolved_configuration() -> None:
    updated = run_scenario(ScenarioInput(preset="physical_extrapolation"))
    young = run_scenario(ScenarioInput(preset="young_reproduction"))

    updated_meta = {row["group"]: row for row in updated.outputs["active_conventions_metadata"]}
    young_meta = {row["group"]: row for row in young.outputs["active_conventions_metadata"]}

    if updated_meta["r7"]["convention"] != "yung_1991":
        raise AssertionError("physical extrapolation should report its actual named R7 convention")
    if updated_meta["o1d_d17o"]["convention"] != "young":
        raise AssertionError("physical extrapolation should report Young's basic O(1D) convention")
    if young_meta["r7"]["convention"] != "custom":
        raise AssertionError("Young reproduction must not mislabel its fitted R7 throughput as a registry default")
    check_close(
        "exported custom Young R7 throughput",
        float(young_meta["r7"]["parameters"]["r7_throughput_factor"]),
        float(young.config["r7_throughput_factor"]),
        1.0e-12,
    )


def test_zahnow_eq3_spherule_conversion_and_uncertainty_metadata() -> None:
    # Zahnow et al. (2025) Table 1 and Eq. 3: modern MA-9.
    check_close("MA-9 reconstructed air D17O", air_d17o_from_spherule(-1.03, 54.0), -0.496, 1.0e-12)
    check_close(
        "MA-9 analytical sigma",
        analytical_air_d17o_sigma(0.08, 0.6),
        math.sqrt(0.08**2 + (0.0285 * 0.6) ** 2),
        1.0e-12,
    )
    low, high = independent_calibration_sensitivity_envelope(-1.03, 54.0)
    if not low < -0.496 < high:
        raise AssertionError("central Zahnow Eq. 3 value should lie inside the calibration sensitivity envelope")


def test_nonpositive_photosynthesis_state_is_hard_domain_failure() -> None:
    result = run_scenario(
        ScenarioInput(
            preset="physical_extrapolation",
            p_o2_pal=0.1,
            gpp_scale=0.05,
            p_co2_ppm=294.4,
        )
    )
    if result.outputs["domain_status"] != "fail":
        raise AssertionError("non-positive required photosynthesis must be a hard domain failure")
    if result.outputs["effective_o2_budget_photosynthesis_feasible"]:
        raise AssertionError("failed oxygen-budget state must not be labelled photosynthetically feasible")


def test_conservative_column_transport_exactly_reduces_to_young_exchange() -> None:
    column = young_two_layer_column()
    rates = column.interface_rate_constants_per_year()[0]
    check_close("Young troposphere to stratosphere rate", float(rates["first_to_second_per_year"]), 0.1, 1.0e-15)
    check_close("Young stratosphere to troposphere rate", float(rates["second_to_first_per_year"]), 1.0, 1.0e-15)
    check_close(
        "Young gross exchange flux",
        float(rates["gross_air_flux_mol_per_year"]),
        1.8e19,
        1.0e6,
    )


def test_conservative_column_transport_conserves_multiple_tracers() -> None:
    column = ConservativeColumn(
        layers=(
            AtmosphericLayer("troposphere", 1.8e20),
            AtmosphericLayer("lower_stratosphere", 1.0e19),
            AtmosphericLayer("upper_stratosphere", 0.8e19),
        ),
        interfaces=(
            ExchangeInterface("troposphere", "lower_stratosphere", 1.8e19, "test interface"),
            ExchangeInterface("lower_stratosphere", "upper_stratosphere", 0.6e19, "test interface"),
        ),
    )
    tracers = np.asarray(
        [
            [3.8e19, 2.0e18, 1.0e18],
            [1.0e16, 2.0e15, 3.0e15],
            [4.0e15, 8.0e14, 2.0e14],
        ]
    )
    tendency = column.derivative(tracers)
    residual = column.conservation_residual(tracers)
    relative_residual = np.abs(residual) / np.maximum(np.sum(np.abs(tendency), axis=-1), 1.0)
    if not np.all(relative_residual < 1.0e-14):
        raise AssertionError(
            f"column transport must conserve every tracer to floating-point precision: {relative_residual}"
        )

    uniform_mixing_ratio = 2.5e-4 * column.air_moles
    uniform_tendency = column.derivative(uniform_mixing_ratio)
    if not np.allclose(uniform_tendency, 0.0, rtol=1.0e-15, atol=1.0):
        raise AssertionError("uniform mixing ratio must be a stationary transport state")


def test_convective_plume_column_matches_finite_volume_known_answer() -> None:
    column = ConvectivePlumeColumn(
        layers=tuple(
            AtmosphericLayer(f"layer_{index}", 1.0)
            for index in range(3)
        ),
        updraught_interface_flux_mol_per_year=np.asarray([1.0, 2.0, 1.0, 0.0]),
        downdraught_interface_flux_mol_per_year=np.asarray([0.0, 1.0, 2.0, 1.0]),
        updraught_detrainment_mol_per_year=np.asarray([0.0, 1.0, 1.0]),
        downdraught_detrainment_mol_per_year=np.asarray([1.0, 1.0, 0.0]),
        source="synthetic IFS plume known answer",
    )
    if not np.array_equal(
        column.updraught_entrainment_mol_per_year,
        np.asarray([1.0, 0.0, 0.0]),
    ):
        raise AssertionError("updraught entrainment must follow IFS mass continuity")
    if not np.array_equal(
        column.downdraught_entrainment_mol_per_year,
        np.asarray([0.0, 0.0, 1.0]),
    ):
        raise AssertionError("downdraught entrainment must follow IFS mass continuity")
    expected = np.asarray(
        [
            [-3.0, 2.0, 1.0],
            [2.0, -4.0, 2.0],
            [1.0, 2.0, -3.0],
        ]
    )
    if not np.array_equal(column.transport_matrix_per_year(), expected):
        raise AssertionError("convective plume finite-volume matrix changed")


def test_convective_plume_transport_conserves_isotopologues() -> None:
    layers = (
        AtmosphericLayer("lower", 8.0),
        AtmosphericLayer("middle", 3.0),
        AtmosphericLayer("upper", 1.0),
    )
    column = ConvectivePlumeColumn(
        layers=layers,
        updraught_interface_flux_mol_per_year=np.asarray([1.0, 2.0, 1.0, 0.0]),
        downdraught_interface_flux_mol_per_year=np.asarray([0.0, 1.0, 2.0, 1.0]),
        updraught_detrainment_mol_per_year=np.asarray([0.0, 1.0, 1.0]),
        downdraught_detrainment_mol_per_year=np.asarray([1.0, 1.0, 0.0]),
        source="synthetic isotope-neutral IFS plume",
    )
    uniform = 4.0e-4 * column.air_moles
    if not np.allclose(column.derivative(uniform), 0.0, rtol=0.0, atol=1.0e-18):
        raise AssertionError("convection must preserve a uniform mixing ratio")
    isotopologues = np.asarray(
        [
            [2.0, 0.3, 0.1],
            [2.0e-3, 4.0e-4, 8.0e-5],
            [8.0e-3, 1.0e-3, 2.0e-4],
        ]
    )
    tendency = column.derivative(isotopologues)
    residual = column.conservation_residual(isotopologues)
    scale = np.maximum(np.sum(np.abs(tendency), axis=1), 1.0)
    if not np.all(np.abs(residual) / scale < 1.0e-14):
        raise AssertionError("convection must conserve each isotopologue")
    matrix = column.transport_matrix_per_year()
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.min(off_diagonal) < 0.0 or np.max(np.diag(matrix)) > 0.0:
        raise AssertionError("convective transport must be positivity preserving")


def test_convective_plume_grid_uses_latitude_major_block_order() -> None:
    def plume(prefix: str, scale: float) -> ConvectivePlumeColumn:
        return ConvectivePlumeColumn(
            layers=(
                AtmosphericLayer(f"{prefix}_lower", 2.0),
                AtmosphericLayer(f"{prefix}_upper", 1.0),
            ),
            updraught_interface_flux_mol_per_year=np.asarray(
                [scale, scale, 0.0]
            ),
            downdraught_interface_flux_mol_per_year=np.asarray(
                [0.0, scale, scale]
            ),
            updraught_detrainment_mol_per_year=np.asarray([0.0, scale]),
            downdraught_detrainment_mol_per_year=np.asarray([scale, 0.0]),
            source="synthetic latitude plume",
        )

    first = plume("south", 1.0)
    second = plume("north", 3.0)
    grid = ConvectivePlumeGrid((first, second))
    matrix = grid.transport_matrix_per_year()
    if not np.array_equal(matrix[:2, :2], first.transport_matrix_per_year()):
        raise AssertionError("first latitude plume block is misplaced")
    if not np.array_equal(matrix[2:, 2:], second.transport_matrix_per_year()):
        raise AssertionError("second latitude plume block is misplaced")
    if np.any(matrix[:2, 2:]) or np.any(matrix[2:, :2]):
        raise AssertionError("zonal plume columns must not create meridional transfer")


def test_convective_plume_rejects_negative_inferred_entrainment() -> None:
    try:
        ConvectivePlumeColumn(
            layers=(
                AtmosphericLayer("lower", 2.0),
                AtmosphericLayer("upper", 1.0),
            ),
            updraught_interface_flux_mol_per_year=np.asarray([2.0, 1.0, 0.0]),
            downdraught_interface_flux_mol_per_year=np.asarray([0.0, 0.0, 0.0]),
            updraught_detrainment_mol_per_year=np.asarray([0.0, 1.0]),
            downdraught_detrainment_mol_per_year=np.asarray([0.0, 0.0]),
            source="synthetic inconsistent plume",
        )
    except ValueError as exc:
        if "negative entrainment" not in str(exc):
            raise
    else:
        raise AssertionError("inconsistent ERA5 plume inputs must be rejected")


def test_convective_plume_preserves_small_positive_entrainment() -> None:
    positive_entrainment = 1.0e-10
    column = ConvectivePlumeColumn(
        layers=(
            AtmosphericLayer("lower", 2.0),
            AtmosphericLayer("upper", 1.0),
        ),
        updraught_interface_flux_mol_per_year=np.asarray([1000.0, 500.0, 0.0]),
        downdraught_interface_flux_mol_per_year=np.zeros(3),
        updraught_detrainment_mol_per_year=np.asarray(
            [500.0 + positive_entrainment, 500.0]
        ),
        downdraught_detrainment_mol_per_year=np.zeros(2),
        source="synthetic small-positive-entrainment regression",
    )
    if not np.isclose(
        column.updraught_entrainment_mol_per_year[0],
        positive_entrainment,
        rtol=5.0e-4,
        atol=0.0,
    ):
        raise AssertionError("small physical entrainment must not be rounded to zero")


def test_era5_convection_monthly_request_is_source_complete() -> None:
    months = month_starts(2010, 2010)
    if len(months) != 12 or months[0].isoformat() != "2010-01-01":
        raise AssertionError("ERA5 climatology must request each monthly mean")
    if forecast_monthly_mean_valid_time(months[0]).isoformat() != "2010-02-01":
        raise AssertionError("ERA5 moda January mean must be valid on February 1")
    if forecast_monthly_mean_valid_time(months[-1]).isoformat() != "2011-01-01":
        raise AssertionError("ERA5 moda December mean must be valid on next January 1")
    jobs = monthly_retrieval_jobs(
        months[0],
        _Path("external_data") / "synthetic_era5_convection",
        grid_degrees=10.0,
    )
    if len(jobs) != 3:
        raise AssertionError("convection, model state, and surface state need separate jobs")
    convection = jobs[0].request
    if convection["stream"] != "moda" or convection["type"] != "fc":
        raise AssertionError("ERA5 mean convective rates must use the monthly forecast product")
    if convection["levtype"] != "ml" or convection["levelist"] != "1/to/137":
        raise AssertionError("ERA5 convection must retain all native model levels")
    expected_parameters = "/".join(
        str(value) for value in ERA5_CONVECTION_PARAMETERS.values()
    )
    if convection["param"] != expected_parameters:
        raise AssertionError("ERA5 convection request lost a required Table 13 field")
    if jobs[1].request["type"] != "an" or jobs[2].request["levelist"] != "1":
        raise AssertionError("layer geometry requires analysis state and level-1 surface state")

    batched = calendar_month_retrieval_jobs(
        7,
        2010,
        2019,
        _Path("external_data") / "synthetic_era5_convection",
        grid_degrees=10.0,
    )
    expected_dates = "/".join(f"{year}-07-01" for year in range(2010, 2020))
    if len(batched) != 3 or batched[0].request["date"] != expected_dates:
        raise AssertionError("calendar-month retrieval must batch exactly one decade")
    for batch, monthly in zip(batched, jobs, strict=True):
        for key in ("stream", "type", "levtype", "levelist", "param", "grid"):
            if batch.request[key] != monthly.request[key]:
                raise AssertionError(f"batched retrieval changed the {key} selection")


def test_era5_convection_horizontal_integrals_are_conservative() -> None:
    latitude, longitude = regular_latitude_longitude_edges(10.0)
    area = spherical_cell_areas_m2(latitude, longitude)
    expected_earth_area = 4.0 * math.pi * 6_371_000.0**2
    check_close(
        "ERA5 regular-grid Earth area",
        float(np.sum(area)),
        expected_earth_area,
        expected_earth_area * 2.0e-15,
    )
    unit_flux = np.ones((2, *area.shape), dtype=float)
    zonal_flux = zonal_mass_flux_to_mol_per_year(unit_flux, area)
    if zonal_flux.shape != (2, area.shape[0]):
        raise AssertionError("zonal mass-flux integration changed its level-latitude order")
    if not np.allclose(zonal_flux[0], zonal_flux[1], rtol=0.0, atol=0.0):
        raise AssertionError("identical interface fields must integrate identically")

    detrainment = np.full((2, *area.shape), 2.0, dtype=float)
    thickness = np.full_like(detrainment, 3.0)
    integrated_detrainment = zonal_detrainment_to_mol_per_year(
        detrainment,
        thickness,
        area,
    )
    equivalent_mass_flux = zonal_mass_flux_to_mol_per_year(
        np.full_like(detrainment, 6.0),
        area,
    )
    if not np.allclose(
        integrated_detrainment,
        equivalent_mass_flux,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise AssertionError("volumetric detrainment must integrate over exact layer volume")


def test_era5_convection_climatology_averages_raw_fields_before_projection() -> None:
    def synthetic(scale: float) -> Era5ZonalConvectionMonth:
        up_interface = scale * np.asarray([[2.0], [1.0], [0.0]])
        down_interface = scale * np.asarray([[0.0], [0.5], [1.0]])
        up_det = scale * np.asarray([[0.5], [1.0]])
        down_det = scale * np.asarray([[0.5], [0.5]])
        up_entrainment = up_interface[1:] - up_interface[:-1] + up_det
        down_entrainment = down_interface[:-1] - down_interface[1:] + down_det
        return Era5ZonalConvectionMonth(
            latitude_degrees=np.asarray([0.0]),
            layer_air_moles=np.asarray([[2.0], [1.0]]),
            updraught_interface_mol_per_year=up_interface,
            downdraught_interface_mol_per_year=down_interface,
            updraught_detrainment_mol_per_year=up_det,
            downdraught_detrainment_mol_per_year=down_det,
            updraught_negative_entrainment_mol_per_year=np.maximum(
                -up_entrainment, 0.0
            ),
            downdraught_negative_entrainment_mol_per_year=np.maximum(
                -down_entrainment, 0.0
            ),
            source_files=(f"synthetic_{scale}",),
        )

    climatology = mean_era5_zonal_convection_months(
        (synthetic(1.0), synthetic(3.0)),
        weights=np.asarray([3.0, 1.0]),
    )
    expected_scale = 1.5
    if not np.allclose(
        climatology.updraught_interface_mol_per_year,
        expected_scale * np.asarray([[2.0], [1.0], [0.0]]),
    ):
        raise AssertionError("climatology did not apply normalized source weights")
    if climatology.updraught_detrainment_correction_fraction < 0.0:
        raise AssertionError("climatology continuity correction must be non-negative")
    climatology.to_continuity_projected_plume_grid()
    artifact = Era5ConvectionClimatology(
        calendar_months=np.asarray([7]),
        monthly=(climatology,),
        annual_mean=climatology,
        month_weights=np.asarray([31.0]),
        source_years=(2010, 2011),
    )
    path = _root / "outputs" / "_test_era5_convection_climatology.npz"
    try:
        artifact.save(path)
        restored = load_era5_convection_climatology(path)
    finally:
        path.unlink(missing_ok=True)
    if restored.source_years != artifact.source_years:
        raise AssertionError("climatology archive lost source-year metadata")
    if not np.array_equal(restored.calendar_months, artifact.calendar_months):
        raise AssertionError("climatology archive lost calendar-month metadata")
    if not np.allclose(
        restored.annual_mean.updraught_interface_mol_per_year,
        climatology.updraught_interface_mol_per_year,
    ):
        raise AssertionError("climatology archive changed reduced transport fields")


def test_era5_l137_hybrid_pressure_closes_at_surface() -> None:
    a_pa, b = l137_hybrid_coefficients()
    if len(a_pa) != 138 or len(b) != 138:
        raise AssertionError("ERA5 L137 requires 138 half-level coefficients")
    surface_pressure = np.asarray([[101_325.0, 80_000.0]])
    pressure = l137_half_level_pressure_pa(surface_pressure)
    if pressure.shape != (138, 1, 2):
        raise AssertionError("L137 pressure expansion changed its vertical ordering")
    if not np.array_equal(pressure[-1], surface_pressure):
        raise AssertionError("L137 bottom half-level must equal local surface pressure")
    check_close("L137 half-level 1 pressure", pressure[1, 0, 0], 2.000365, 1e-12)
    if np.any(np.diff(pressure, axis=0) <= 0.0):
        raise AssertionError("L137 pressure must increase from model top to surface")


def test_era5_global_center_grid_preserves_earth_area_and_order() -> None:
    latitude_descending = np.arange(90.0, -91.0, -10.0)
    longitude = np.arange(0.0, 360.0, 10.0)
    area = global_cell_areas_from_centers_m2(latitude_descending, longitude)
    expected = 4.0 * math.pi * 6_371_000.0**2
    check_close("ERA5 centre-grid Earth area", float(np.sum(area)), expected, expected * 2e-15)
    if area.shape != (19, 36) or not np.allclose(area[0], area[-1]):
        raise AssertionError("ERA5 centre-grid areas must preserve descending latitude order")


def test_era5_l137_hydrostatic_geometry_is_positive_and_ordered() -> None:
    temperature = np.full((137, 2, 3), 250.0)
    humidity = np.zeros_like(temperature)
    surface_pressure = np.full((2, 3), 101_325.0)
    surface_geopotential = np.zeros((2, 3))
    full_z, half_z, thickness = l137_geopotential_and_thickness(
        temperature,
        humidity,
        surface_pressure,
        surface_geopotential,
    )
    if full_z.shape != temperature.shape or thickness.shape != temperature.shape:
        raise AssertionError("L137 hydrostatic geometry changed the model-level shape")
    if half_z.shape != (138, 2, 3) or not np.array_equal(half_z[-1], surface_geopotential):
        raise AssertionError("L137 half-level geopotential must close at the surface")
    if np.any(thickness <= 0.0) or np.any(np.diff(half_z, axis=0) >= 0.0):
        raise AssertionError("L137 layer thickness must be positive top to bottom")
    if np.any(full_z <= half_z[1:]) or np.any(full_z >= half_z[:-1]):
        raise AssertionError("full-level geopotential must lie inside each layer")


def test_two_reservoir_plume_reduction_preserves_exchange_flux() -> None:
    column = ConvectivePlumeColumn(
        layers=(AtmosphericLayer("lower", 10.0), AtmosphericLayer("upper", 5.0)),
        updraught_interface_flux_mol_per_year=np.asarray([1.0, 1.0, 0.0]),
        downdraught_interface_flux_mol_per_year=np.asarray([0.0, 1.0, 1.0]),
        updraught_detrainment_mol_per_year=np.asarray([0.0, 1.0]),
        downdraught_detrainment_mol_per_year=np.asarray([1.0, 0.0]),
        source="synthetic exact two-reservoir plume",
    )
    reduced = aggregate_plume_grid_to_two_reservoirs(
        ConvectivePlumeGrid((column,)), np.asarray([True, False])
    )
    check_close(
        "two-reservoir bidirectional plume flux",
        reduced.bidirectional_air_flux_mol_per_year,
        2.0,
        2.0e-15,
    )
    if np.max(np.abs(np.sum(reduced.inventory_matrix_per_year, axis=0))) > 1.0e-15:
        raise AssertionError("two-reservoir plume reduction must conserve inventory")


def test_directed_overturning_loop_is_stationary_and_conservative() -> None:
    """A closed two-stream loop can represent upwelling and return flow."""

    layers = (
        AtmosphericLayer("tropical_lower", 8.0),
        AtmosphericLayer("tropical_upper", 2.0),
        AtmosphericLayer("extratropical_upper", 3.0),
        AtmosphericLayer("extratropical_lower", 7.0),
    )
    circulation = ConservativeCirculationNetwork(
        layers=layers,
        fluxes=(
            DirectedAirFlux(
                "tropical_lower",
                "tropical_upper",
                0.5,
                "synthetic closed-loop test",
            ),
            DirectedAirFlux(
                "tropical_upper",
                "extratropical_upper",
                0.5,
                "synthetic closed-loop test",
            ),
            DirectedAirFlux(
                "extratropical_upper",
                "extratropical_lower",
                0.5,
                "synthetic closed-loop test",
            ),
            DirectedAirFlux(
                "extratropical_lower",
                "tropical_lower",
                0.5,
                "synthetic closed-loop test",
            ),
        ),
    )

    if not np.array_equal(
        circulation.air_mass_tendency_mol_per_year(),
        np.zeros(len(layers)),
    ):
        raise AssertionError("closed circulation must leave every cell air inventory fixed")

    uniform = 2.5e-4 * circulation.air_moles
    if not np.allclose(circulation.derivative(uniform), 0.0, rtol=0.0, atol=1.0e-18):
        raise AssertionError("closed circulation must preserve a uniform mixing ratio")

    tracers = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.4, 0.5],
        ]
    )
    tendency = circulation.derivative(tracers)
    relative_residual = np.abs(np.sum(tendency, axis=1)) / np.maximum(
        np.sum(np.abs(tendency), axis=1),
        1.0,
    )
    if not np.all(relative_residual < 1.0e-14):
        raise AssertionError("directed circulation must conserve every passive tracer")


def test_directed_circulation_rejects_air_mass_divergence() -> None:
    layers = (
        AtmosphericLayer("lower", 8.0),
        AtmosphericLayer("upper", 2.0),
    )
    try:
        ConservativeCirculationNetwork(
            layers=layers,
            fluxes=(
                DirectedAirFlux(
                    "lower",
                    "upper",
                    0.5,
                    "synthetic divergent test",
                ),
            ),
        )
    except ValueError as exc:
        if "not stationary" not in str(exc):
            raise
    else:
        raise AssertionError("fixed-inventory circulation must reject divergent air fluxes")


def test_streamfunction_builds_an_exact_closed_overturning_cell() -> None:
    air = np.asarray(
        [
            [8.0, 2.0],
            [7.0, 3.0],
        ]
    )
    streamfunction = np.zeros((3, 3))
    streamfunction[1, 1] = 0.5
    circulation = circulation_network_from_streamfunction(
        air,
        streamfunction,
        "synthetic streamfunction test",
    )

    expected_directions = {
        (
            latitude_vertical_cell_name(0, 1),
            latitude_vertical_cell_name(0, 0),
        ),
        (
            latitude_vertical_cell_name(0, 0),
            latitude_vertical_cell_name(1, 0),
        ),
        (
            latitude_vertical_cell_name(1, 0),
            latitude_vertical_cell_name(1, 1),
        ),
        (
            latitude_vertical_cell_name(1, 1),
            latitude_vertical_cell_name(0, 1),
        ),
    }
    actual_directions = {
        (flux.source_layer, flux.target_layer)
        for flux in circulation.fluxes
    }
    if actual_directions != expected_directions:
        raise AssertionError(f"unexpected streamfunction flux orientation: {actual_directions}")
    if not all(flux.air_flux_mol_per_year == 0.5 for flux in circulation.fluxes):
        raise AssertionError("single-cell streamfunction must produce one common loop flux")
    if not np.array_equal(
        circulation.air_mass_tendency_mol_per_year(),
        np.zeros(4),
    ):
        raise AssertionError("streamfunction-derived circulation must be divergence-free")

    uniform = 1.0e-4 * circulation.air_moles
    if not np.allclose(circulation.derivative(uniform), 0.0, rtol=0.0, atol=1.0e-18):
        raise AssertionError("streamfunction-derived circulation must preserve uniform mixing")


def test_streamfunction_rejects_open_boundary_flow() -> None:
    air = np.ones((2, 2))
    streamfunction = np.zeros((3, 3))
    streamfunction[0, 1] = 1.0
    try:
        circulation_network_from_streamfunction(
            air,
            streamfunction,
            "synthetic open-boundary test",
        )
    except ValueError as exc:
        if "outer boundary" not in str(exc):
            raise
    else:
        raise AssertionError("non-constant boundary streamfunction must be rejected")


def test_open_lower_streamfunction_closes_through_reservoir_row() -> None:
    atmospheric_air = np.asarray([[2.0], [3.0]])
    lower_air = np.asarray([8.0, 7.0])
    streamfunction = np.zeros((3, 2))
    streamfunction[1, 0] = 0.5
    circulation = circulation_network_with_lower_reservoir_row(
        atmospheric_air,
        lower_air,
        streamfunction,
        "synthetic lower-boundary closure test",
    )

    expected_directions = {
        (
            latitude_vertical_cell_name(1, 0),
            latitude_vertical_cell_name(1, 1),
        ),
        (
            latitude_vertical_cell_name(1, 1),
            latitude_vertical_cell_name(0, 1),
        ),
        (
            latitude_vertical_cell_name(0, 1),
            latitude_vertical_cell_name(0, 0),
        ),
        (
            latitude_vertical_cell_name(0, 0),
            latitude_vertical_cell_name(1, 0),
        ),
    }
    actual_directions = {
        (flux.source_layer, flux.target_layer)
        for flux in circulation.fluxes
    }
    if actual_directions != expected_directions:
        raise AssertionError(f"unexpected lower-reservoir circulation: {actual_directions}")
    if not np.array_equal(
        circulation.air_mass_tendency_mol_per_year(),
        np.zeros(4),
    ):
        raise AssertionError("lower-boundary reservoir closure must conserve every cell")
    uniform = 2.0e-4 * circulation.air_moles
    if not np.allclose(circulation.derivative(uniform), 0.0, rtol=0.0, atol=1.0e-18):
        raise AssertionError("lower-boundary closure must preserve uniform mixing")


def test_lower_reservoir_requires_closed_poles_and_model_top() -> None:
    atmospheric_air = np.asarray([[2.0], [3.0]])
    lower_air = np.asarray([8.0, 7.0])
    streamfunction = np.zeros((3, 2))
    streamfunction[1, 1] = 0.5
    try:
        circulation_network_with_lower_reservoir_row(
            atmospheric_air,
            lower_air,
            streamfunction,
            "synthetic open-top test",
        )
    except ValueError as exc:
        if "polar and upper" not in str(exc):
            raise
    else:
        raise AssertionError("lower-reservoir closure must reject an open model top")


def test_diffusive_and_circulation_operators_can_be_combined() -> None:
    layers = (
        AtmosphericLayer("a", 4.0),
        AtmosphericLayer("b", 3.0),
        AtmosphericLayer("c", 2.0),
    )
    mixing = ConservativeColumn(
        layers=layers,
        interfaces=(
            ExchangeInterface("a", "b", 0.2, "synthetic mixing test"),
            ExchangeInterface("b", "c", 0.1, "synthetic mixing test"),
        ),
    )
    circulation = ConservativeCirculationNetwork(
        layers=layers,
        fluxes=(
            DirectedAirFlux("a", "b", 0.05, "synthetic closed-loop test"),
            DirectedAirFlux("b", "c", 0.05, "synthetic closed-loop test"),
            DirectedAirFlux("c", "a", 0.05, "synthetic closed-loop test"),
        ),
    )
    matrix = combined_transport_matrix_per_year(mixing, circulation)
    uniform = 3.0e-4 * mixing.air_moles
    if not np.allclose(matrix @ uniform, 0.0, rtol=0.0, atol=1.0e-18):
        raise AssertionError("combined conservative transport must preserve uniform mixing")
    tracer = np.asarray([1.0, 2.0, 4.0])
    if abs(float(np.sum(matrix @ tracer))) > 1.0e-14:
        raise AssertionError("combined conservative transport must conserve tracer inventory")


def test_morgan_2004_latitude_pressure_grid_is_reproduced_exactly() -> None:
    grid = MORGAN_2004_GRID
    if not np.array_equal(grid.latitude_edges_degrees, np.arange(-90.0, 91.0, 10.0)):
        raise AssertionError("Morgan latitude boxes must retain their printed 10-degree edges")
    if not np.array_equal(grid.latitude_centers_degrees, np.arange(-85.0, 86.0, 10.0)):
        raise AssertionError("Morgan latitude centers must span -85 to 85 degrees")
    check_close("Morgan scale height", grid.scale_height_km, 6.948711710452029, 1.0e-12)
    check_close("Morgan model-top altitude", grid.altitude_edges_km[-1], 80.0, 1.0e-12)
    check_close("Morgan model-top pressure", grid.pressure_edges_mbar[-1], 0.01, 1.0e-14)
    if len(grid.pressure_centers_mbar) != 40:
        raise AssertionError("Morgan grid must contain 40 vertical cell centers")
    if not np.all(np.diff(grid.pressure_edges_mbar) < 0.0):
        raise AssertionError("Morgan pressure must decrease monotonically upward")

    expected_tropopause = np.asarray(
        [
            300.0 if abs(latitude) > 60.0 else 200.0 if abs(latitude) > 30.0 else 100.0
            for latitude in grid.latitude_centers_degrees
        ]
    )
    actual_tropopause = np.asarray(
        [
            grid.tropopause_pressure_mbar(latitude)
            for latitude in grid.latitude_centers_degrees
        ]
    )
    if not np.array_equal(actual_tropopause, expected_tropopause):
        raise AssertionError("Morgan latitude-dependent tropopause rule changed")
    if len(MORGAN_2004_CONTINUITY_TERMS) != 6:
        raise AssertionError("Morgan Appendix A continuity-equation term inventory changed")


def test_caltech_jpl_circulation_scale_is_not_silently_equated_to_young_exchange() -> None:
    shia = SHIA_1989_TROPOPAUSE_FLUX.air_flux_mol_per_year()
    jiang = JIANG_2004_400K_FLUX.air_flux_mol_per_year()
    check_close("Shia circulation-scale conversion", shia, 1.0132529596370756e19, 1.0e6)
    check_close("Jiang circulation-scale conversion", jiang, 1.5253270360128018e19, 1.0e6)
    young_exchange = 1.8e19
    if not shia < jiang < young_exchange:
        raise AssertionError(
            "published circulation-scale fluxes should remain distinct from Young gross exchange"
        )


def test_hydrostatic_latitude_pressure_inventory_conserves_global_area_and_mass() -> None:
    latitude_edges = MORGAN_2004_GRID.latitude_edges_degrees
    pressure_edges_pa = MORGAN_2004_GRID.pressure_edges_mbar * 100.0
    air_moles = hydrostatic_air_moles(latitude_edges, pressure_edges_pa)
    if air_moles.shape != (18, 40):
        raise AssertionError("Morgan hydrostatic inventory must follow the 18x40 grid")
    if not np.all(air_moles > 0.0):
        raise AssertionError("every hydrostatic cell must have positive air inventory")
    check_close(
        "Morgan hydrostatic global inventory",
        float(np.sum(air_moles)),
        1.7956885811360707e20,
        1.0e7,
    )

    # Equal 10-degree latitude widths do not have equal areas, but mirrored
    # latitude belts must have equal hydrostatic inventories.
    if not np.allclose(air_moles, air_moles[::-1, :], rtol=1.0e-14, atol=0.0):
        raise AssertionError("hydrostatic latitude inventory must be hemispherically symmetric")

    altitude_edges = MORGAN_2004_GRID.altitude_edges_km
    interval = np.flatnonzero(
        (altitude_edges[:-1] >= 10.0) & (altitude_edges[1:] <= 60.0)
    )
    check_close(
        "Morgan hydrostatic 10-60 km inventory",
        float(np.sum(air_moles[:, interval])),
        4.255097999812286e19,
        1.0e7,
    )


def test_era5_tem_coordinate_adapter_preserves_source_nodes() -> None:
    latitude = np.arange(90.0, -90.1, -0.5)
    pressure = np.asarray([3.0, 100.0, 10000.0, 100000.0])
    streamfunction = pressure[:, None] + latitude[None, :]
    climatology = Era5TemClimatology(
        latitude_degrees=latitude,
        pressure_pa=pressure,
        streamfunction_kg_per_s=streamfunction,
        month_count=120,
        total_weight_hours=86928.0,
        start_month="201001",
        end_month="201912",
        source="synthetic ERA5 TEM adapter test",
    )
    target_latitude, target_pressure, target_streamfunction = (
        climatology.ten_degree_native_pressure_nodes()
    )
    if not np.array_equal(target_latitude, np.arange(-90.0, 91.0, 10.0)):
        raise AssertionError("ERA5 adapter must preserve exact 10-degree source nodes")
    if not np.array_equal(target_pressure, pressure[::-1]):
        raise AssertionError("ERA5 adapter must orient pressure from lower to upper atmosphere")
    check_close(
        "ERA5 adapter node value",
        target_streamfunction[3, 2],
        target_pressure[2] + target_latitude[3],
        1.0e-12,
    )
    closed = climatology.ten_degree_native_pressure_nodes(
        set_physical_pole_values_to_zero=True
    )[2]
    if not np.array_equal(closed[[0, -1], :], np.zeros((2, len(pressure)))):
        raise AssertionError("optional physical pole closure must set only pole nodes to zero")


def test_vertical_reference_constraints_are_kept_distinct_from_a_grid() -> None:
    references = published_vertical_references()
    if references != (
        LIANG_2006_DOMAIN,
        LIANG_2008_CORE_DOMAIN,
        LIANG_2008_EXTENDED_DOMAIN,
    ):
        raise AssertionError("published vertical references changed unexpectedly")
    if LIANG_2006_DOMAIN.reported_vertical_count != 66:
        raise AssertionError("Liang 2006 reported layer count must remain explicit")
    if LIANG_2008_CORE_DOMAIN.coordinate != "logarithmic pressure":
        raise AssertionError("Liang 2008 coordinate must not be silently converted to altitude")
    if (YOUNG_BULK_STRATOSPHERE.lower_altitude_km, YOUNG_BULK_STRATOSPHERE.upper_altitude_km) != (
        10.0,
        60.0,
    ):
        raise AssertionError("Young Equation 28 diagnostic bounds changed")


def test_vertical_profile_air_weighted_young_diagnostic() -> None:
    cells = tuple(
        VerticalCell(
            lower_altitude_km=float(lower),
            upper_altitude_km=float(lower + 10),
            air_moles=float(100 - lower),
            number_density_molecules_cm3=float(100 - lower),
            temperature_k=250.0,
            pressure_center_bar=10.0 ** (-(lower + 5) / 20.0),
            eddy_diffusivity_cm2_per_s=1.0e5,
        )
        for lower in range(0, 80, 10)
    )
    profile = ValidatedVerticalProfile(
        name="synthetic validation profile",
        cells=cells,
        atmospheric_state_source="test fixture",
        eddy_diffusivity_source="test fixture",
    )
    values = np.arange(len(cells), dtype=float)
    selected = profile.whole_cell_indices(10.0, 60.0)
    if not np.array_equal(selected, np.arange(1, 6)):
        raise AssertionError(f"unexpected Young diagnostic cells: {selected}")
    expected = float(np.average(values[1:6], weights=profile.air_moles[1:6]))
    check_close(
        "air-weighted Young diagnostic",
        profile.air_weighted_mean(values, 10.0, 60.0),
        expected,
        1.0e-15,
    )


def test_vertical_profile_rejects_unresolved_partial_cells() -> None:
    profile = ValidatedVerticalProfile(
        name="single-cell validation profile",
        cells=(
            VerticalCell(
                lower_altitude_km=0.0,
                upper_altitude_km=20.0,
                air_moles=1.0,
                number_density_molecules_cm3=1.0,
                temperature_k=250.0,
                pressure_center_bar=0.3,
                eddy_diffusivity_cm2_per_s=1.0e5,
            ),
        ),
        atmospheric_state_source="test fixture",
        eddy_diffusivity_source="test fixture",
    )
    try:
        profile.whole_cell_indices(10.0, 20.0)
    except ValueError as exc:
        if "exact vertical cell edges" not in str(exc):
            raise
    else:
        raise AssertionError("partial-cell diagnostics must require an explicit remapping method")


def test_profile_eddy_diffusion_operator_is_conservative() -> None:
    cells = tuple(
        VerticalCell(
            lower_altitude_km=float(index),
            upper_altitude_km=float(index + 1),
            air_moles=float(5 - index),
            number_density_molecules_cm3=float(5 - index) * 1.0e18,
            temperature_k=250.0,
            pressure_center_bar=10.0 ** (-(index + 0.5)),
            eddy_diffusivity_cm2_per_s=1.0e5 * (index + 1),
        )
        for index in range(4)
    )
    profile = ValidatedVerticalProfile(
        name="synthetic diffusion profile",
        cells=cells,
        atmospheric_state_source="test fixture",
        eddy_diffusivity_source="test fixture",
    )
    column = eddy_diffusion_column(profile)
    uniform = 2.0e-4 * column.air_moles
    uniform_tendency = column.derivative(uniform)
    tendency_scale = (
        np.max(np.abs(column.transport_matrix_per_year()))
        * np.max(np.abs(uniform))
    )
    if np.max(np.abs(uniform_tendency)) > 1.0e-14 * tendency_scale:
        raise AssertionError("eddy diffusion must preserve a uniform mixing ratio")
    tracer = np.asarray([1.0, 2.0, 4.0, 8.0])
    tendency = column.derivative(tracer)
    if abs(float(np.sum(tendency))) > 1.0e-12 * float(np.sum(np.abs(tendency))):
        raise AssertionError("eddy diffusion must conserve passive tracer inventory")


def test_mean_first_passage_time_for_two_layer_exchange() -> None:
    column = ConservativeColumn(
        layers=(
            AtmosphericLayer("source", 1.0),
            AtmosphericLayer("target", 4.0),
        ),
        interfaces=(
            ExchangeInterface("source", "target", 2.0, "test interface"),
        ),
    )
    check_close(
        "two-layer first-passage time",
        mean_first_passage_years(
            column.transport_matrix_per_year(),
            start_layer=0,
            first_target_layer=1,
        ),
        0.5,
        1.0e-15,
    )


def test_current_young_like_architecture_is_named_and_guarded() -> None:
    metadata = architecture_metadata()
    if metadata["name"] != CURRENT_ARCHITECTURE_NAME:
        raise AssertionError("current architecture metadata should expose the stable candidate name")
    if metadata["preferred_reduced_young_like_name"] != PREFERRED_REDUCED_YOUNG_LIKE_NAME:
        raise AssertionError("metadata should expose the preferred reduced Young-like branch")
    if PREFERRED_REDUCED_YOUNG_LIKE_NAME != YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME:
        raise AssertionError("preferred reduced Young-like branch should use the balanced Fig. 8/low-pO2 candidate")
    if STEADY_SOURCE_LAW.processed_signature != "young_o1d":
        raise AssertionError("Young-like steady architecture should use the Young O1D processed signature")
    check_close("transient residence time", TRANSIENT_LAW.residence_time_yr, 0.36, 1.0e-12)
    if len(SCORE_THRESHOLDS) < 10:
        raise AssertionError("score thresholds should cover steady, transient, and literature guardrails")
    result, _threshold = threshold_status("Fig. 8 50% GPP worst residual", 0.11332282346330791)
    if result != "warning":
        raise AssertionError("known current Fig. 8 50% GPP residual should remain a warning, not disappear")


def test_current_architecture_preset_runs_integrated_source_law() -> None:
    if CURRENT_ARCHITECTURE_NAME not in preset_names():
        raise AssertionError("current architecture should be available as a scenario preset")
    if YOUNG_LIKE_V0_NAME not in preset_names():
        raise AssertionError("v0 architecture should remain available for comparison")
    if YOUNG_LIKE_V1_NAME not in preset_names():
        raise AssertionError("v1 architecture should remain available as a steady candidate")
    if YOUNG_LIKE_V2_NAME not in preset_names():
        raise AssertionError("v2 architecture should remain available as a threshold-access candidate")
    if YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME not in preset_names():
        raise AssertionError("v2 access-reservoir candidate should remain available")
    if YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME not in preset_names():
        raise AssertionError("v2 tau=1 access-reservoir candidate should remain available")
    if YOUNG_LIKE_V2_LOG_LOW_GPP_ACCESS_RESERVOIR_TAU1_NAME not in preset_names():
        raise AssertionError("v2 log-low-GPP candidate should remain available")
    if YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME not in preset_names():
        raise AssertionError("v2 log-low-GPP tail-access candidate should remain available")
    if YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME not in preset_names():
        raise AssertionError("v2 log-low-GPP smooth tail-access candidate should remain available")
    if YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME not in preset_names():
        raise AssertionError("v2 log-low-GPP balanced tail-access candidate should remain available")
    if YOUNG_LIKE_V2_PARALLEL_CO2_NAME not in preset_names():
        raise AssertionError("v2 parallel-column CO2 candidate should remain available")
    result = run_scenario(
        ScenarioInput(
            preset=CURRENT_ARCHITECTURE_NAME,
            p_co2_ppm=2769.866666666667,
            gpp_scale=0.5,
        )
    )
    if result.outputs["photo_o17_source_law"] != "three_term_processed_column":
        raise AssertionError("current architecture should use integrated three-term processed-column source law")
    check_close(
        "integrated current architecture Fig. 8 check point",
        result.outputs["O2_trop_D17O_permil"],
        -2.6512835113432924,
        1.0e-5,
    )
    check_close(
        "integrated current architecture recovery activation",
        result.outputs["processed_column_low_gpp_recovery_activation"],
        0.0,
        1.0e-12,
    )
    v1 = run_scenario(
        ScenarioInput(
            preset=YOUNG_LIKE_V1_NAME,
            p_co2_ppm=2769.866666666667,
            gpp_scale=0.5,
        )
    )
    check_close("v1 steady improvement check point", v1.outputs["O2_trop_D17O_permil"], -2.527440734661587, 1.0e-5)
    check_close("v1 recovery tau", v1.outputs["processed_column_recovery_tau_yr"], 0.317, 1.0e-12)
    v2 = run_scenario(
        ScenarioInput(
            preset=YOUNG_LIKE_V2_NAME,
            p_co2_ppm=2769.866666666667,
            gpp_scale=0.5,
        )
    )
    check_close("v2 steady checkpoint remains v1-like", v2.outputs["O2_trop_D17O_permil"], -2.527440734661587, 1.0e-5)
    if v2.outputs["processed_column_low_gpp_recovery_po2_mode"] != "threshold_access":
        raise AssertionError("v2 should expose threshold-access recovery gating")
    check_close("v2 threshold half PAL", v2.outputs["processed_column_low_gpp_recovery_po2_half_pal"], 0.85, 1.0e-12)
    v2_config = config_from_scenario(ScenarioInput(preset=YOUNG_LIKE_V2_NAME))
    check_close("v2 finite exposure residence", v2_config.r7_finite_exposure_residence_time_yr, 0.405, 1.0e-12)

    v2_access_config = config_from_scenario(ScenarioInput(preset=YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME))
    if v2_access_config.processed_access_reservoir_mode != "lagged_state":
        raise AssertionError("v2 access candidate should enable the lagged processed-access state")
    check_close("v2 access reservoir tau", v2_access_config.processed_access_reservoir_tau_yr, 30.0, 1.0e-12)
    v2_access = run_scenario(ScenarioInput(preset=YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME))
    if v2_access.outputs["processed_access_reservoir_mode"] != "lagged_state":
        raise AssertionError("v2 access candidate should export reservoir metadata")
    check_close(
        "v2 access candidate steady checkpoint remains v2-like",
        v2_access.outputs["O2_trop_D17O_permil"],
        run_scenario(ScenarioInput(preset=YOUNG_LIKE_V2_NAME)).outputs["O2_trop_D17O_permil"],
        1.0e-12,
    )
    v2_access_tau1_config = config_from_scenario(ScenarioInput(preset=YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME))
    check_close("v2 tau=1 access reservoir tau", v2_access_tau1_config.processed_access_reservoir_tau_yr, 1.0, 1.0e-12)
    v2_tail_access_config = config_from_scenario(
        ScenarioInput(preset=YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME)
    )
    check_close(
        "v2 log-low-GPP tail-access fraction",
        v2_tail_access_config.processed_column_low_gpp_tail_po2_fraction,
        0.50,
        1.0e-12,
    )
    check_close(
        "v2 log-low-GPP tail-access power",
        v2_tail_access_config.processed_column_low_gpp_tail_po2_power,
        1.0,
        1.0e-12,
    )
    v2_smooth_tail_config = config_from_scenario(
        ScenarioInput(preset=YOUNG_LIKE_V2_LOG_LOW_GPP_SMOOTH_TAIL_ACCESS_RESERVOIR_TAU1_NAME)
    )
    check_close(
        "v2 smooth-tail gate half ppm",
        v2_smooth_tail_config.processed_column_gate_half_ppm,
        35000.0,
        1.0e-12,
    )
    check_close(
        "v2 smooth-tail gate hill",
        v2_smooth_tail_config.processed_column_gate_hill,
        4.0,
        1.0e-12,
    )
    check_close(
        "v2 smooth-tail access fraction",
        v2_smooth_tail_config.processed_column_low_gpp_tail_po2_fraction,
        0.50,
        1.0e-12,
    )
    v2_balanced_tail_config = config_from_scenario(
        ScenarioInput(preset=YOUNG_LIKE_V2_LOG_LOW_GPP_BALANCED_TAIL_ACCESS_RESERVOIR_TAU1_NAME)
    )
    check_close(
        "v2 balanced-tail gate half ppm",
        v2_balanced_tail_config.processed_column_gate_half_ppm,
        30000.0,
        1.0e-12,
    )
    check_close(
        "v2 balanced-tail gate hill",
        v2_balanced_tail_config.processed_column_gate_hill,
        6.0,
        1.0e-12,
    )
    check_close(
        "v2 balanced-tail shared access fraction",
        v2_balanced_tail_config.processed_column_shared_tail_po2_fraction,
        1.0,
        1.0e-12,
    )
    check_close(
        "v2 balanced-tail access fraction",
        v2_balanced_tail_config.processed_column_low_gpp_tail_po2_fraction,
        0.50,
        1.0e-12,
    )

    v2_parallel = run_scenario(ScenarioInput(preset=YOUNG_LIKE_V2_PARALLEL_CO2_NAME))
    check_close(
        "v2 parallel candidate O2 modern",
        v2_parallel.outputs["O2_trop_D17O_permil"],
        -0.4382522365894701,
        2.0e-11,
    )
    check_close(
        "v2 parallel candidate column flux ratio",
        v2_parallel.outputs["column_processed_flux_ratio_to_young"],
        0.9990638806095485,
        1.0e-10,
    )
    check_close(
        "v2 parallel candidate fixed reported CO2 D17O",
        v2_parallel.outputs["column_processed_mixed_D17O_fixed_reported_permil"],
        1.6147440792460568,
        2.0e-10,
    )


def test_v2_access_reservoir_candidate_preserves_dynamic_behavior() -> None:
    for preset, expected_min in (
        (YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME, -0.7314),
        (YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME, -0.7307),
    ):
        rows = {
            row["constraint"]: row
            for row in dynamic_residual_rows(preset)
        }
        check_close(f"{preset} Fig. 9 min", rows["fig9_min_O2_D17O"]["model"], expected_min, 0.005)
        check_close(f"{preset} Fig. 9 final", rows["fig9_half_rp_final_O2_D17O"]["model"], -0.5348, 0.005)
        check_close(f"{preset} Fig. 10 shift", rows["fig10_150yr_shift"]["model"], -0.0050, 0.003)


def test_young_reproduction_uses_source_derived_bulk_fig10() -> None:
    rows = {
        row["constraint"]: row
        for row in dynamic_residual_rows(CURRENT_YOUNG_REPRODUCTION_PRESET)
    }
    check_close(
        "source-derived Young Fig. 10 shift",
        rows["fig10_150yr_shift"]["model"],
        -0.004915295,
        2.0e-8,
    )
    if "source-derived Young bulk" not in rows["fig10_150yr_shift"]["warnings"]:
        raise AssertionError("Young Fig. 10 score must expose the bulk-R7 provenance")


def test_young_reproduction_uses_source_derived_bulk_fig9() -> None:
    rows = {
        row["constraint"]: row
        for row in dynamic_residual_rows(CURRENT_YOUNG_REPRODUCTION_PRESET)
    }
    check_close(
        "source-derived Young Fig. 9 minimum",
        rows["fig9_min_O2_D17O"]["model"],
        -0.7531,
        0.003,
    )
    check_close(
        "source-derived Young Fig. 9 final",
        rows["fig9_half_rp_final_O2_D17O"]["model"],
        -0.5483,
        0.003,
    )
    if "no-finite-exposure CO2" not in rows["fig9_min_O2_D17O"]["warnings"]:
        raise AssertionError("Young Fig. 9 score must expose the source-derived driver")


def test_young_transient_dispatch_preserves_raw_bulk_baselines() -> None:
    args = SimpleNamespace(
        preset=CURRENT_YOUNG_REPRODUCTION_PRESET,
        model_variant="r7_throughput_diagnostic",
        r5_mode="variant_default",
        r7_throughput_factor=2.25,
        r8c_factor=1.0,
        co2_sink_factor=None,
        co2_ocean_infusion_factor=None,
        fig9_years=200.0,
        fig10_years=150.0,
        samples=5,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    rows, diagnostics = run_experiments(args)
    fig9 = [row for row in rows if row["experiment"] == "fig9_half_photosynthesis"]
    fig10 = [row for row in rows if row["experiment"] == "fig10_co2_step"]
    check_close(
        "raw source-derived Fig. 9 baseline",
        fig9[0]["O2_trop_D17O_permil"],
        -0.4113691751,
        2.0e-9,
    )
    check_close(
        "raw source-derived Fig. 10 baseline",
        fig10[0]["O2_trop_D17O_permil"],
        -0.4113691751,
        2.0e-9,
    )
    if any(float(row["r7_throughput_factor"]) != 1.0 for row in (*fig9, *fig10)):
        raise AssertionError("canonical Young transients must use the printed bulk R7 ledger")
    messages = " | ".join(str(row["message"]) for row in diagnostics)
    if "no finite-exposure adjustment" not in messages:
        raise AssertionError("canonical Young transient provenance is missing")


def test_source_derived_young_bulk_fig10_component() -> None:
    trajectory = integrate_young_bulk_fig10(np.asarray((0.0, 150.0, 5000.0)))
    check_close(
        "Young bulk Fig. 10 reduced 294 ppm equilibrium",
        trajectory.initial_equilibrium.cap_delta17_prime_permil,
        -0.4113691751,
        2.0e-10,
    )
    check_close(
        "Young bulk Fig. 10 150-year shift",
        trajectory.cap_delta17_shift_permil[1],
        -0.0049152950,
        2.0e-10,
    )
    check_close(
        "Young bulk Fig. 10 5000-year shift",
        trajectory.cap_delta17_shift_permil[2],
        -0.0467148120,
        2.0e-10,
    )


def test_full_table3_source_inventory_is_encoded() -> None:
    if len(TABLE3_ISOTOPE_TARGETS) != 21:
        raise AssertionError(f"expected 21 Table 3 isotope targets, got {len(TABLE3_ISOTOPE_TARGETS)}")
    if len(TABLE3_MOLE_TARGETS) != 21:
        raise AssertionError(f"expected 21 Table 3 mole targets, got {len(TABLE3_MOLE_TARGETS)}")
    if len(TABLE3_MOLE_FRACTION_TARGETS) != 4:
        raise AssertionError("expected 4 Table 3 mole-fraction targets")
    if len(TABLE3_FLUX_TARGETS) != 1:
        raise AssertionError("expected 1 Table 3 flux target")
    if len(TABLE3_SCALAR_TARGETS) != len(TABLE3_ISOTOPE_TARGETS):
        raise AssertionError("Table 3 scalar validation targets should expose every printed isotope target")
    for key, value in TABLE3_MOLE_TARGETS.items():
        check_close(f"Table 3 mole target {key}", TABLE3_MOLES[key], value, max(abs(value) * 1e-15, 1e-15))


def test_reduced_mass_alpha() -> None:
    mu = reduced_mass(16.0, 32.0)
    mu_prime = reduced_mass(18.0, 32.0)
    alpha = collision_frequency_alpha(mu, mu_prime)
    if not (0.96 < alpha < 1.0):
        raise AssertionError(alpha)


def test_phanerozoic_o2_conversions() -> None:
    check_close("0.1 PAL Young percent O2", pal_to_percent(0.1, convention="young"), 2.12, 0.001)
    check_close("2.1 PAL Young percent O2", pal_to_percent(2.1, convention="young"), 44.52, 0.001)
    check_close("Mills modern PAL", percent_to_pal(20.6449666166846, convention="mills"), 1.0, 1.0e-12)
    if (MODEL_PO2_WORKING_MIN_PAL, MODEL_PO2_WORKING_MAX_PAL) != (0.1, 2.1):
        raise AssertionError("unexpected Phanerozoic pO2 working envelope")


def test_pco2_envelope_status() -> None:
    if pco2_envelope_status(30000.0).level != "recommended":
        raise AssertionError("30,000 ppm should remain in the recommended Young Fig. 8 envelope")
    if pco2_envelope_status(50000.0).level != "exploratory":
        raise AssertionError("50,000 ppm should be exploratory extrapolation")
    if pco2_envelope_status(100000.0).level != "exploratory":
        raise AssertionError("100,000 ppm should remain available for exploratory extrapolation")


def test_threshold_access_warns_beyond_young_fig8_domain() -> None:
    result = run_scenario(
        ScenarioInput(
            preset="young_reproduction_explicit_lower_split_candidate",
            p_co2_ppm=50000.0,
            gpp_scale=0.5,
        )
    )
    joined = "\n".join(result.warnings)
    if "nonphysical hook" not in joined:
        raise AssertionError("threshold/access extrapolation warning missing")


def test_low_po2_low_gpp_budget_infeasibility_is_flagged() -> None:
    result = run_scenario(
        ScenarioInput(
            preset=YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
            p_o2_pal=0.1,
            gpp_scale=0.05,
            p_co2_ppm=294.4,
        )
    )
    if result.outputs["effective_o2_budget_photosynthesis_feasible"]:
        raise AssertionError("0.1 PAL and 5% GPP should be marked as O2-budget infeasible")
    if result.outputs["domain_status"] != "fail":
        raise AssertionError("O2-budget infeasible scenario should be classified as domain fail")
    if "o2_budget_requires_nonpositive_photosynthesis" not in result.outputs["domain_reasons"]:
        raise AssertionError("O2-budget infeasibility reason missing from domain metadata")
    if result.outputs["effective_o2_budget_required_rp_o2_mol_per_year"] >= 0.0:
        raise AssertionError("0.1 PAL and 5% GPP should require non-positive photosynthetic O2")
    joined = "\n".join(result.warnings)
    if "physically infeasible" not in joined:
        raise AssertionError("O2-budget infeasibility warning missing")

    valid = run_scenario(
        ScenarioInput(
            preset=YOUNG_LIKE_V2_LOG_LOW_GPP_TAIL_ACCESS_RESERVOIR_TAU1_NAME,
            p_o2_pal=0.2,
            gpp_scale=0.10,
            p_co2_ppm=294.4,
        )
    )
    if valid.outputs["domain_status"] != "valid":
        raise AssertionError(f"expected 0.2 PAL and 10% GPP to remain valid, got {valid.outputs['domain_status']}")


def test_reaction_engine() -> None:
    species = ["A", "B", "C"]
    y = np.array([2.0, 3.0, 0.0])
    rxn = Reaction(
        key="R",
        reactants={"A": 1.0, "B": 1.0},
        products={"C": 1.0},
        rate_constant=0.5,
        units="arbitrary",
    )
    dydt = derivative(y, [rxn], species)
    assert np.allclose(dydt, [-3.0, -3.0, 3.0])


def test_reaction_inventory_counts() -> None:
    keys = [record.key for record in REACTION_RECORDS]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate reaction keys")
    if len(keys) < 60:
        raise AssertionError(f"reaction inventory unexpectedly small: {len(keys)}")


def test_source_row_audit_coverage() -> None:
    audited_keys = {row.key for row in REMAINING_TABLE2_ROWS} | {row.key for row in R4_R6_R7_ROWS}
    expected_core = {record.key for record in REACTION_RECORDS if record.key.startswith(("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"))}
    missing = sorted(expected_core - audited_keys)
    if missing:
        raise AssertionError(f"core Table 2 reaction rows missing from source audit: {missing}")
    rows = audit_remaining_table2_rows()
    mismatches = [row for row in rows if row["status"] != "MATCH"]
    if mismatches:
        raise AssertionError(f"remaining Table 2 row audit mismatches: {mismatches}")


def test_young_r2_mass_dependent_ranges() -> None:
    # Young text says 1000*ln(alpha_MD) ranges from about -5.10 per mil
    # for (O + O17O)/(O + O2) to about -38.57 per mil for
    # (18O + O2)/(O + O2).
    got_o_o17o = 1000.0 * math.log(md_alpha("O", "O2", "O", "O17O"))
    got_18o_o2 = 1000.0 * math.log(md_alpha("O", "O2", "18O", "O2"))
    check_close("R2 O+O17O alpha_MD", got_o_o17o, -5.10, 0.1)
    check_close("R2 18O+O2 alpha_MD", got_18o_o2, -38.57, 0.2)


def test_executable_reaction_count() -> None:
    keys = [reaction.key for reaction in executable_reactions()]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate executable reaction keys")
    if len(keys) != 90:
        raise AssertionError(f"expected 90 executable reactions, got {len(keys)}")


def test_r8_rate_scales() -> None:
    rates = {reaction.key: reaction.rate_constant for reaction in r8_co2_h2o_exchange_reactions()}
    # R8a should be about 1.041 * 1.00525 * R18_SMOW.
    check_close("R8a", rates["R8a"], 0.00210, 0.00002)
    # R8c should be about 1.041^0.528 * 1.00525^0.528 * R17_SMOW.
    check_close("R8c", rates["R8c"], 0.000389, 0.000005)


def test_co2_photo_sink_factor_scales_default_mode() -> None:
    base = {
        reaction.key: reaction.rate_constant
        for reaction in executable_reactions(
            closure_mode="modern",
            co2_photo_sink_mode="o2_source_water",
            co2_photo_sink_factor=1.0,
        )
        if hasattr(reaction, "rate_constant")
    }
    doubled = {
        reaction.key: reaction.rate_constant
        for reaction in executable_reactions(
            closure_mode="modern",
            co2_photo_sink_mode="o2_source_water",
            co2_photo_sink_factor=2.0,
        )
        if hasattr(reaction, "rate_constant")
    }
    check_close("CO18O photosynthetic sink factor", doubled["photo_CO18O_sink"] / base["photo_CO18O_sink"], 2.0, 1.0e-12)
    check_close("CO17O photosynthetic sink factor", doubled["photo_CO17O_sink"] / base["photo_CO17O_sink"], 2.0, 1.0e-12)
    check_close("C16O2 photosynthetic sink unchanged", doubled["photo_CO2_sink"] / base["photo_CO2_sink"], 1.0, 1.0e-12)


def test_model_runner_table3_isotopes() -> None:
    summaries = {summary.label: summary for summary in isotope_summaries(scaled_table3_state(1.0, 294.4))}
    check_close("runner O2 d18 prime", summaries["O2_trop"].delta18_prime, 23.212, 0.001)
    check_close("runner O2 D17 prime", summaries["O2_trop"].cap_delta17, -0.369, 0.001)
    check_close("runner CO2 strat D17 prime", summaries["CO2_strat"].cap_delta17, 1.613, 0.001)


def test_fast_stratosphere_solver_table3_ozone() -> None:
    y = scaled_table3_state(1.0, 294.4)
    reactions = executable_reactions(r5_collision_partner_moles=5.589807e18)
    result = solve_fast_species(y, reactions)
    if not result.converged:
        raise AssertionError(f"fast solver did not converge: {result.residual_norm}")
    summaries = {summary.label: summary for summary in isotope_summaries(result.y)}
    check_close("fast solver O3 D17 prime", summaries["O3_strat"].cap_delta17, 29.497, 0.01)


def test_tropospheric_co2_isotope_solver_converges() -> None:
    y = scaled_table3_state(1.0, 294.4)
    reactions = executable_reactions(
        r5_collision_partner_moles=5.589807e18,
        closure_mode="modern",
        co2_photo_sink_mode="o2_source_water",
    )
    result = solve_co2_isotopes(y, reactions, tolerance=1.0e-8)
    if not result.converged:
        raise AssertionError(f"CO2 isotope solver did not converge: {result.residual_norm}")


def test_coupled_isotope_subsystem_converges() -> None:
    y = scaled_table3_state(1.0, 294.4)
    reactions = executable_reactions(
        r5_collision_partner_moles=5.589807e18,
        closure_mode="modern",
        co2_photo_sink_mode="o2_source_water",
    )
    result = solve_subsystem(y, reactions)
    if not result.converged:
        raise AssertionError(f"coupled isotope subsystem did not converge: {result.residual_norm}")
    summaries = {summary.label: summary for summary in isotope_summaries(result.y)}
    check_close("coupled solver O3 D17 prime", summaries["O3_strat"].cap_delta17, 29.497, 0.02)


def test_fixed_reservoir_solver_converges() -> None:
    y = scaled_table3_state(1.0, 294.4)
    reactions = executable_reactions(
        r5_collision_partner_moles=5.589807e18,
        closure_mode="modern",
        co2_photo_sink_mode="o2_source_water",
    )
    result = solve_fixed_reservoir_isotopes(y, reactions)
    if not result.converged:
        raise AssertionError(f"fixed-reservoir isotope solver did not converge: {result.residual_norm}")
    summaries = {summary.label: summary for summary in isotope_summaries(result.y)}
    check_close("fixed-reservoir solver O3 D17 prime", summaries["O3_strat"].cap_delta17, 29.497, 0.02)


def test_o2_mass_balance_branch_zeroes_heavy_o2_tendencies() -> None:
    result = solve_tropospheric_o2_mass_balance(config_from_scenario(ScenarioInput(preset="physical_o2_budget")))
    for balance in result.balances:
        residual_key = f"{balance.species}_residual_mol_per_year"
        residual = abs(float(result.outputs[residual_key]))
        if residual / balance.source_mol_per_year > 1.0e-12:
            raise AssertionError(f"{balance.species} mass-balance residual too large: {residual}")


def test_modern_validation_rows_include_scored_and_contextual_constraints() -> None:
    outputs = {
        "O2_trop_D17O_permil": -0.432,
        "CO2_trop_D17O_permil": 0.04,
        "CO2_strat_D17O_flux_permil_mol_per_year": 4.27e15,
        "effective_gpp_pgC_per_year": 290.0,
    }
    rows = modern_validation_rows(outputs)
    if len(rows) < 8:
        raise AssertionError("modern validation table unexpectedly small")
    if not any(row["numeric_comparison"] for row in rows):
        raise AssertionError("expected at least one scored modern comparison")
    if not any(not row["numeric_comparison"] for row in rows):
        raise AssertionError("expected at least one contextual modern reference")


def test_public_preset_aliases_exist() -> None:
    names = set(preset_names())
    for preset in ("young_printed_inputs", "young_reproduction", "updated_physical"):
        if preset not in names:
            raise AssertionError(f"missing public preset alias: {preset}")
        if "Diagnostic or legacy" in preset_role(preset):
            raise AssertionError(f"missing preset role for {preset}")


def test_gpp_normalization_preserves_young_default_and_records_user_scale() -> None:
    default = config_from_scenario(ScenarioInput(preset="young_reproduction", gpp_scale=1.0))
    check_close("default internal Young GPP scale", default.gpp_scale, 1.0, 1.0e-12)

    beerling = config_from_scenario(
        ScenarioInput(
            preset="young_reproduction",
            gpp_scale=1.0,
            gpp_normalization="beerling_1999",
        )
    )
    check_close(
        "Beerling-normalized internal scale",
        beerling.gpp_scale,
        BEERLING_1999_MODERN_GPP_PGC_PER_YEAR / YOUNG_MODERN_GPP_PGC_PER_YEAR,
        1.0e-12,
    )

    liang = config_from_scenario(
        ScenarioInput(
            preset="young_reproduction",
            gpp_scale=1.0,
            gpp_normalization="liang_2023",
        )
    )
    check_close("Liang-normalized internal scale", liang.gpp_scale, 290.0 / YOUNG_MODERN_GPP_PGC_PER_YEAR, 1.0e-12)

    result = run_scenario(
        ScenarioInput(
            preset="young_reproduction",
            gpp_scale=0.25,
            gpp_normalization="liang_2023",
        )
    )
    check_close("user GPP percent", result.outputs["gpp_user_percent_modern"], 25.0, 1.0e-12)
    check_close("requested Liang GPP PgC/yr", result.outputs["gpp_requested_pgC_per_year"], 72.5, 1.0e-12)
    if DEFAULT_GPP_NORMALIZATION != "young_2014":
        raise AssertionError("default GPP normalization should remain the Young/Beerling gross O2-production scale")
    if result.outputs["gpp_normalization_role"] != "explicit_modern_reference_sensitivity":
        raise AssertionError("non-default GPP normalizations should be exported as explicit sensitivities")
    if result.outputs["gpp_normalization_label"] != "Liang et al., 2023":
        raise AssertionError("GPP normalization label should be exported for reproducibility")
    check_close(
        "stored internal Young GPP scale",
        result.outputs["gpp_internal_young_scale"],
        0.25 * 290.0 / YOUNG_MODERN_GPP_PGC_PER_YEAR,
        1.0e-12,
    )


def test_two_box_export_branch_reduces_export_flux_not_bulk_state() -> None:
    bulk = run_scenario(ScenarioInput(preset="physical_directional_r7_no_damping_candidate"))
    two_box = run_scenario(ScenarioInput(preset="updated_physical_two_box_export_experimental"))
    check_close(
        "two-box keeps solved bulk CO2_strat D17O",
        two_box.outputs["CO2_strat_D17O_permil"],
        bulk.outputs["CO2_strat_D17O_permil"],
        1.0e-12,
    )
    if not two_box.outputs["CO2_export_D17O_permil"] < bulk.outputs["CO2_export_D17O_permil"]:
        raise AssertionError("two-box export signature should be lower than bulk stratospheric export")
    if not two_box.outputs["CO2_strat_D17O_flux_permil_mol_per_year"] < bulk.outputs["CO2_strat_D17O_flux_permil_mol_per_year"]:
        raise AssertionError("two-box export flux should be lower than one-box bulk export flux")
    check_close(
        "one-box diagnostic flux preserved",
        two_box.outputs["CO2_strat_D17O_flux_one_box_permil_mol_per_year"],
        bulk.outputs["CO2_strat_D17O_flux_permil_mol_per_year"],
        1.0e-6,
    )


def test_explicit_lower_threshold_candidate_matches_fig8_endpoint() -> None:
    preset = "young_reproduction_explicit_lower_threshold_candidate"
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")
    if "explicit lower-stratospheric" not in preset_role(preset):
        raise AssertionError("explicit lower threshold preset role should describe the lower box")

    result = run_scenario(ScenarioInput(preset=preset, p_co2_ppm=30000.0, gpp_scale=0.5))
    check_close(
        "explicit lower threshold Fig. 8 50% endpoint",
        result.outputs["O2_trop_D17O_permil"],
        -12.56296881169322,
        0.05,
    )
    if result.outputs["CO2_export_signature_mode"] != "explicit_lower_box":
        raise AssertionError("explicit lower candidate should report explicit lower-box export")
    if result.outputs["explicit_lower_box_species_count"] != 30:
        raise AssertionError("explicit lower candidate should expose 30 species")


def test_explicit_lower_net_export_rate_only_scales_flux() -> None:
    preset = "young_reproduction_explicit_lower_threshold_candidate"
    base = run_scenario(ScenarioInput(preset=preset, p_co2_ppm=30000.0, gpp_scale=0.5))
    split = run_scenario(
        ScenarioInput(
            preset=preset,
            p_co2_ppm=30000.0,
            gpp_scale=0.5,
            explicit_lower_box_net_export_rate_per_year=1.5,
        )
    )
    check_close(
        "net export split preserves O2 isotope solution",
        split.outputs["O2_trop_D17O_permil"],
        base.outputs["O2_trop_D17O_permil"],
        1.0e-10,
    )
    check_close(
        "net export split scales lower-box flux",
        split.outputs["CO2_strat_D17O_flux_permil_mol_per_year"] / base.outputs["CO2_strat_D17O_flux_permil_mol_per_year"],
        0.5,
        1.0e-10,
    )
    check_close(
        "net export fraction metadata",
        split.outputs["explicit_lower_box_net_export_fraction_of_internal_exchange"],
        0.5,
        1.0e-12,
    )


def test_explicit_lower_split_candidate_uses_transport_bounded_rates() -> None:
    preset = "young_reproduction_explicit_lower_split_candidate"
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")
    if "Transport-bounded" not in preset_role(preset):
        raise AssertionError("split candidate role should describe transport-bounded status")

    result = run_scenario(ScenarioInput(preset=preset))
    check_close(
        "split candidate internal lower-box exchange",
        result.outputs["explicit_lower_box_lower_to_trop_rate_per_year"],
        2.0,
        1.0e-12,
    )
    check_close(
        "split candidate net lower-box export",
        result.outputs["explicit_lower_box_net_export_rate_per_year"],
        1.0,
        1.0e-12,
    )
    if result.outputs["explicit_lower_box_source_weight_mode"] != "none":
        raise AssertionError("split candidate should not include low-pCO2 source weighting")
    check_close(
        "split candidate source weight disabled",
        result.outputs["explicit_lower_box_source_weight"],
        1.0,
        1.0e-12,
    )


def test_updated_explicit_export_bridge_matches_modern_anchors() -> None:
    preset = "updated_physical_from_young_explicit_export"
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")
    config = config_from_scenario(ScenarioInput(preset=preset))
    check_close("explicit export Pack a_MIF", config.a_mif, PACK2021_EXPLICIT_EXPORT_A_MIF, 1.0e-12)
    check_close(
        "explicit export net rate",
        config.explicit_lower_box_net_export_rate_per_year,
        ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
        1.0e-12,
    )

    result = run_scenario(ScenarioInput(preset=preset))
    outputs = result.outputs
    check_close("explicit export modern O2 Pack anchor", outputs["O2_trop_D17O_permil"], -0.432, 1.0e-6)
    check_close(
        "explicit export Adnew isoflux",
        permil_mol_per_year_to_permil_pgc_per_year(outputs["CO2_strat_D17O_flux_permil_mol_per_year"]),
        51.3,
        1.0e-6,
    )
    check_close(
        "explicit export residence time",
        1.0 / outputs["explicit_lower_box_net_export_rate_per_year"],
        1.0419931743319633,
        1.0e-12,
    )
    if outputs["CO2_export_signature_mode"] != "explicit_lower_box":
        raise AssertionError("updated explicit export bridge should use the explicit lower-box signature")

    alias = run_scenario(ScenarioInput(preset="updated_physical"))
    if alias.outputs["CO2_export_signature_mode"] != "explicit_lower_box":
        raise AssertionError("updated_physical should retain the explicit lower-box export diagnostic")


def test_updated_validated_young_explicit_export_candidate_matches_pack_anchor() -> None:
    preset = "updated_physical_from_validated_young_explicit_export"
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")
    config = config_from_scenario(ScenarioInput(preset=preset))
    check_close(
        "validated young explicit export Pack a_MIF",
        config.a_mif,
        PACK2021_VALIDATED_YOUNG_EXPLICIT_EXPORT_A_MIF,
        1.0e-12,
    )
    check_close(
        "validated young explicit export net rate",
        config.explicit_lower_box_net_export_rate_per_year,
        ADNEW2025_EXPLICIT_EXPORT_NET_RATE_PER_YEAR,
        1.0e-12,
    )
    result = run_scenario(ScenarioInput(preset=preset))
    check_close(
        "validated young explicit export modern O2 Pack anchor",
        result.outputs["O2_trop_D17O_permil"],
        -0.432,
        1.0e-6,
    )
    if result.outputs["CO2_export_signature_mode"] != "explicit_lower_box":
        raise AssertionError("validated young updated candidate should use explicit lower-box signature")


def test_updated_local_pack_anchor_preserves_raw_young_ode_output() -> None:
    preset = UPDATED_VALIDATED_YOUNG_LOCAL_PACK_PRESET
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")

    raw_updated = run_scenario(
        ScenarioInput(
            preset=preset,
            solve_mode="full_atmosphere",
            o2_d17o_calibration_mode="none",
        )
    )
    updated = run_scenario(ScenarioInput(preset=preset, solve_mode="full_atmosphere"))
    check_close(
        "local Pack raw O2 Delta17O preserves uncalibrated ODE result",
        updated.outputs["O2_trop_D17O_raw_permil"],
        raw_updated.outputs["O2_trop_D17O_raw_permil"],
        1.0e-12,
    )
    check_close(
        "local Pack reported O2 Delta17O",
        updated.outputs["O2_trop_D17O_permil"],
        PACK2021_O2_D17O_TARGET_PERMIL,
        1.0e-12,
    )
    check_close(
        "local Pack offset bookkeeping",
        updated.outputs["O2_trop_D17O_permil"] - updated.outputs["O2_trop_D17O_raw_permil"],
        updated.outputs["o2_d17o_calibration_offset_permil"],
        1.0e-12,
    )
    if updated.outputs["o2_d17o_calibration_mode"] != "pack2021_validated_young_local_offset":
        raise AssertionError("local Pack preset should expose its calibration mode")


def test_updated_beta_respiration_pack_candidate_is_mechanistic_not_offset() -> None:
    preset = UPDATED_VALIDATED_YOUNG_BETA_PACK_PRESET
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")
    config = config_from_scenario(ScenarioInput(preset=preset))
    check_close(
        "beta Pack candidate beta_respiration_17",
        config.beta_respiration_17,
        PACK2021_BETA_RESPIRATION_17_CANDIDATE,
        1.0e-12,
    )
    result = run_scenario(ScenarioInput(preset=preset, solve_mode="full_atmosphere"))
    outputs = result.outputs
    check_close(
        "beta Pack candidate keeps Young a_MIF",
        outputs["a_mif"],
        1.065,
        1.0e-12,
    )
    if outputs["o2_d17o_calibration_mode"] != "none":
        raise AssertionError("beta Pack candidate should not use output calibration")
    check_close(
        "beta Pack candidate O2 Delta17O near Pack",
        outputs["O2_trop_D17O_permil"],
        PACK2021_O2_D17O_TARGET_PERMIL,
        5.0e-4,
    )


def test_updated_water_beta_pack_candidate_is_mechanistic_not_offset() -> None:
    preset = UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET
    if preset not in set(preset_names()):
        raise AssertionError(f"missing preset {preset}")
    config = config_from_scenario(ScenarioInput(preset=preset))
    check_close(
        "water beta Pack candidate evapotranspiration_beta_17",
        config.evapotranspiration_beta_17,
        PACK2021_EVAPOTRANSPIRATION_BETA_17_CANDIDATE,
        1.0e-12,
    )
    result = run_scenario(ScenarioInput(preset=preset, solve_mode="full_atmosphere"))
    outputs = result.outputs
    check_close(
        "water beta Pack candidate keeps Young a_MIF",
        outputs["a_mif"],
        1.065,
        1.0e-12,
    )
    if outputs["o2_d17o_calibration_mode"] != "none":
        raise AssertionError("water beta Pack candidate should not use output calibration")
    check_close(
        "water beta Pack candidate O2 Delta17O near Pack",
        outputs["O2_trop_D17O_permil"],
        PACK2021_O2_D17O_TARGET_PERMIL,
        5.0e-4,
    )


def test_updated_physical_alias_points_to_water_beta_public_candidate() -> None:
    alias_config = config_from_scenario(ScenarioInput(preset="updated_physical"))
    water_config = config_from_scenario(ScenarioInput(preset=UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET))
    check_close(
        "updated_physical alias evapotranspiration beta",
        alias_config.evapotranspiration_beta_17,
        water_config.evapotranspiration_beta_17,
        1.0e-12,
    )
    alias = run_scenario(ScenarioInput(preset="updated_physical", solve_mode="full_atmosphere"))
    water = run_scenario(ScenarioInput(preset=UPDATED_VALIDATED_YOUNG_WATER_BETA_PACK_PRESET, solve_mode="full_atmosphere"))
    check_close(
        "updated_physical alias keeps Young a_MIF",
        alias.outputs["a_mif"],
        1.065,
        1.0e-12,
    )
    check_close(
        "updated_physical alias O2 Delta17O",
        alias.outputs["O2_trop_D17O_permil"],
        water.outputs["O2_trop_D17O_permil"],
        1.0e-12,
    )
    if alias.outputs["o2_d17o_calibration_mode"] != "none":
        raise AssertionError("updated_physical alias should use mechanistic parameters, not output calibration")


def test_processed_altitude_reservoir_repairs_modern_co2_diagnostics() -> None:
    result = run_scenario(ScenarioInput(preset="young_reproduction_explicit_lower_split_candidate"))
    processed_d17 = 15.0
    fraction = joint_required_fraction(
        base_strat_d17o_permil=result.outputs["CO2_strat_D17O_permil"],
        base_export_d17o_permil=result.outputs["CO2_export_D17O_permil"],
        base_flux_permil_mol_per_year=result.outputs["CO2_strat_D17O_flux_permil_mol_per_year"],
        processed_d17o_permil=processed_d17,
    )
    if not 0.04 < fraction < 0.07:
        raise AssertionError(f"unexpected processed altitude fraction: {fraction}")
    mixed = apply_processed_reservoir_mix(result.outputs, processed_d17, fraction)
    check_close(
        "processed altitude CO2 strat diagnostic",
        mixed["CO2_strat_D17O_permil"],
        TABLE3_TARGETS["D17_CO2_strat_permil"],
        0.01,
    )
    check_close(
        "processed altitude CO2 flux diagnostic",
        mixed["CO2_strat_D17O_flux_permil_mol_per_year"],
        TABLE3_FLUX_TARGETS["D17_CO2_flux_permil_mol_per_year"],
        2.0e13,
    )


def test_young_o1d_can_generate_processed_altitude_signature() -> None:
    activation = finite_exposure_activation(1.0, 1.0)
    check_close("finite exposure activation", activation, 1.0 - math.exp(-1.0), 1.0e-12)

    source = ProcessedSignatureSource(
        o1d_d17o_permil=TABLE3_TARGETS["D17_O1D_permil"],
        activation=1.0,
        transfer_efficiency=1.0,
        mode="inherit_o1d_delta",
    )
    processed_d17 = source.processed_d17o_permil()
    check_close("Young O1D-derived processed signature", processed_d17, TABLE3_TARGETS["D17_O1D_permil"], 1.0e-12)

    result = run_scenario(ScenarioInput(preset="young_reproduction_explicit_lower_split_candidate"))
    fraction = joint_required_fraction(
        base_strat_d17o_permil=result.outputs["CO2_strat_D17O_permil"],
        base_export_d17o_permil=result.outputs["CO2_export_D17O_permil"],
        base_flux_permil_mol_per_year=result.outputs["CO2_strat_D17O_flux_permil_mol_per_year"],
        processed_d17o_permil=processed_d17,
    )
    if not 0.02 < fraction < 0.04:
        raise AssertionError(f"unexpected Young O1D-derived processed fraction: {fraction}")
    mixed = apply_processed_reservoir_mix(result.outputs, processed_d17, fraction)
    check_close(
        "Young O1D-derived CO2 strat diagnostic",
        mixed["CO2_strat_D17O_permil"],
        TABLE3_TARGETS["D17_CO2_strat_permil"],
        0.01,
    )


def test_isotope_resolved_processed_reservoir_matches_young_o1d_limit() -> None:
    base = CO2IsotopeSignature(
        TABLE3_TARGETS["d17_CO2_strat_permil"],
        TABLE3_TARGETS["d18_CO2_strat_permil"],
    )
    source = O1DIsotopeTransferSource(
        o1d_delta17_prime_permil=TABLE3_TARGETS["d17_O1D_permil"],
        o1d_delta18_prime_permil=TABLE3_TARGETS["d18_O1D_permil"],
        activation=1.0,
        transfer_efficiency=1.0,
        mode="inherit_o1d_primes",
    )
    processed = source.processed_signature()
    check_close("processed O1D d17", processed.delta17_prime_permil, TABLE3_TARGETS["d17_O1D_permil"], 1.0e-12)
    check_close("processed O1D d18", processed.delta18_prime_permil, TABLE3_TARGETS["d18_O1D_permil"], 1.0e-12)
    check_close("processed O1D D17", processed.cap_delta17_permil, TABLE3_TARGETS["D17_O1D_permil"], 0.002)
    fraction = required_fraction_isotope_resolved(base, processed, 2.0)
    mixed = mix_co2_isotope_signatures(base, processed, fraction)
    check_close("isotope-resolved target D17", mixed.cap_delta17_permil, 2.0, 1.0e-8)


def test_explicit_processed_reservoir_prototype_converges() -> None:
    scenario = ScenarioInput(preset="young_reproduction_explicit_lower_split_candidate")
    config = config_from_scenario(scenario)
    lower_box = LowerBoxConfig(
        upper_survival_fraction=explicit_lower_box_upper_fraction(config),
        lower_to_trop_rate_per_year=config.explicit_lower_box_lower_to_trop_rate_per_year,
        net_export_rate_per_year=config.explicit_lower_box_net_export_rate_per_year,
        lower_major_scale=config.explicit_lower_box_lower_major_scale,
    )
    processed_box = ProcessedBoxConfig(
        reservoir_fraction_of_lower=0.01,
        source_flux_mol_per_year=1.35e13,
    )
    result, _, _ = solve_processed_box_fixed_reservoir(scenario, lower_box, processed_box)
    if not result.converged:
        raise AssertionError(f"processed reservoir prototype did not converge: {result.residual_norm}")
    summaries = summarize_processed_state(result.y)
    if not summaries["CO2_lower"].cap_delta17 > 1.5:
        raise AssertionError("processed reservoir should raise lower-box CO2 anomaly toward Young")
    check_close(
        "processed reservoir source signature",
        summaries["CO2_proc"].cap_delta17,
        TABLE3_TARGETS["D17_O1D_permil"],
        0.002,
    )


def test_parallel_processed_reservoir_preserves_lower_state() -> None:
    scenario = ScenarioInput(preset="young_reproduction_explicit_lower_split_candidate")
    config = config_from_scenario(scenario)
    lower_box = LowerBoxConfig(
        upper_survival_fraction=explicit_lower_box_upper_fraction(config),
        lower_to_trop_rate_per_year=config.explicit_lower_box_lower_to_trop_rate_per_year,
        net_export_rate_per_year=config.explicit_lower_box_net_export_rate_per_year,
        lower_major_scale=config.explicit_lower_box_lower_major_scale,
    )
    lower_only = run_scenario(scenario)
    processed_box = ProcessedBoxConfig(
        reservoir_fraction_of_lower=0.01,
        source_flux_mol_per_year=1.43e14,
        route_to_lower=False,
    )
    result, _, _ = solve_processed_box_fixed_reservoir(scenario, lower_box, processed_box)
    if not result.converged:
        raise AssertionError(f"parallel processed reservoir did not converge: {result.residual_norm}")
    summaries = summarize_processed_state(result.y)
    check_close(
        "parallel route preserves lower CO2 isotope state",
        summaries["CO2_lower"].cap_delta17,
        lower_only.outputs["CO2_export_D17O_permil"],
        0.002,
    )
    idx = {name: i for i, name in enumerate(PROCESSED_SPECIES_ORDER)}
    lower_major_flux = lower_box.effective_net_export_rate_per_year * result.y[idx["CO2_lower"]]
    processed_major_flux = processed_box.source_flux_mol_per_year
    total_anomaly_flux = (
        lower_major_flux * summaries["CO2_lower"].cap_delta17
        + processed_major_flux * summaries["CO2_proc"].cap_delta17
    )
    check_close(
        "parallel processed anomaly flux",
        total_anomaly_flux,
        TABLE3_FLUX_TARGETS["D17_CO2_flux_permil_mol_per_year"],
        5.0e13,
    )


def test_explicit_lower_scenario_reports_parallel_processed_requirement() -> None:
    result = run_scenario(ScenarioInput(preset="young_reproduction_explicit_lower_split_candidate"))
    outputs = result.outputs
    fraction = outputs["processed_parallel_fraction_for_young_anomaly"]
    if not 0.02 < fraction < 0.04:
        raise AssertionError(f"unexpected parallel processed fraction: {fraction}")
    check_close(
        "parallel processed O1D signature",
        outputs["processed_parallel_CO2_D17O_permil"],
        TABLE3_TARGETS["D17_O1D_permil"],
        1.0e-12,
    )
    anomaly_flux = (
        outputs["CO2_strat_D17O_flux_permil_mol_per_year"]
        + outputs["processed_parallel_flux_for_young_anomaly_mol_per_year"]
        * outputs["processed_parallel_CO2_D17O_permil"]
    )
    check_close(
        "scenario parallel processed flux target",
        anomaly_flux,
        TABLE3_FLUX_TARGETS["D17_CO2_flux_permil_mol_per_year"],
        1.0e-6,
    )


def test_column_processed_fraction_matches_young_column_requirement() -> None:
    column = ColumnProcessedFraction(upper_activation=0.30)
    check_close("Young upper column fraction", column.upper_column_fraction, 1.0 / 11.0, 1.0e-12)
    check_close("column processed fraction", column.processed_fraction, 0.30 / 11.0, 1.0e-12)
    activation = column.activation_for_processed_fraction(0.028380867347182388)
    if not 0.30 < activation < 0.32:
        raise AssertionError(f"unexpected Eq. 28 upper-column activation: {activation}")


def test_explicit_lower_scenario_reports_column_processed_law() -> None:
    result = run_scenario(ScenarioInput(preset="young_reproduction_explicit_lower_split_candidate"))
    outputs = result.outputs
    check_close("scenario column activation", outputs["column_processed_upper_activation"], 0.30, 1.0e-12)
    check_close("scenario column fraction", outputs["column_processed_fraction"], 0.30 / 11.0, 1.0e-12)
    if not 0.99 < outputs["column_processed_flux_ratio_to_young"] < 1.02:
        raise AssertionError(
            f"column processed law should stay near Young modern flux: {outputs['column_processed_flux_ratio_to_young']}"
        )
    check_close(
        "scenario column fixed D17",
        outputs["column_processed_mixed_D17O_fixed_reported_permil"],
        1.620020378425691,
        1.0e-11,
    )


def test_scorecard_uses_parallel_processed_modern_co2_context() -> None:
    rows = scalar_residual_rows("young_reproduction_explicit_lower_split_candidate")
    by_key = {row["constraint"]: row for row in rows}
    if "modern_CO2_strat_D17O_raw_lower_box" not in by_key:
        raise AssertionError("scorecard should retain raw lower-box CO2 context")
    if by_key["modern_CO2_strat_D17O_raw_lower_box"]["weight"] != 0.0:
        raise AssertionError("raw lower-box context should not affect score")
    if by_key["modern_CO2_strat_D17O"]["model"] == by_key["modern_CO2_strat_D17O_raw_lower_box"]["model"]:
        raise AssertionError("processed-aware CO2 strat score should differ from raw lower-box value")
    check_close(
        "preferred column-law CO2 flux score",
        by_key["co2_d17_flux"]["model"],
        8.569907801871906e15,
        1.0e5,
    )
    check_close(
        "preferred column-law CO2 strat score",
        by_key["modern_CO2_strat_D17O"]["model"],
        1.620020378425691,
        1.0e-11,
    )
    if "modern_CO2_strat_D17O_column_law_extra_major" not in by_key:
        raise AssertionError("scorecard should retain source-derived column law context")
    if by_key["modern_CO2_strat_D17O_column_law_extra_major"]["weight"] != 0.0:
        raise AssertionError("column law context should not affect the primary preferred score")
    if "co2_d17_flux_exact_target" not in by_key:
        raise AssertionError("scorecard should retain exact-target processed QA context")
    if by_key["co2_d17_flux_exact_target"]["weight"] != 0.0:
        raise AssertionError("exact-target context should not affect the preferred score")
    check_close(
        "scorecard column law flux context",
        by_key["co2_d17_flux_column_law"]["model"] / TABLE3_FLUX_TARGETS["D17_CO2_flux_permil_mol_per_year"],
        1.002328397879755,
        1.0e-11,
    )


def test_scorecard_can_compare_processed_modes() -> None:
    preset = "young_reproduction_explicit_lower_split_candidate"
    preferred = {row["constraint"]: row for row in scalar_residual_rows(preset, processed_mode="preferred")}
    exact = {row["constraint"]: row for row in scalar_residual_rows(preset, processed_mode="exact")}
    exact_fixed = {row["constraint"]: row for row in scalar_residual_rows(preset, processed_mode="exact_fixed")}
    column_extra = {row["constraint"]: row for row in scalar_residual_rows(preset, processed_mode="column_extra")}
    column_fixed = {row["constraint"]: row for row in scalar_residual_rows(preset, processed_mode="column_fixed")}
    raw = {row["constraint"]: row for row in scalar_residual_rows(preset, processed_mode="raw")}
    check_close(
        "exact processed flux score",
        exact["co2_d17_flux"]["model"],
        TABLE3_FLUX_TARGETS["D17_CO2_flux_permil_mol_per_year"],
        1.0,
    )
    check_close(
        "column extra-major D17O",
        column_extra["modern_CO2_strat_D17O"]["model"],
        1.5770109878480176,
        1.0e-11,
    )
    if not column_fixed["modern_CO2_strat_D17O"]["model"] > exact["modern_CO2_strat_D17O"]["model"]:
        raise AssertionError("column fixed-denominator mode should report higher mixed CO2 D17O than exact mode")
    check_close(
        "exact fixed reported CO2 D17O",
        exact_fixed["modern_CO2_strat_D17O"]["model"],
        TABLE3_TARGETS["D17_CO2_strat_permil"],
        1.0e-12,
    )
    if raw["modern_CO2_strat_D17O"]["model"] != raw["modern_CO2_strat_D17O_raw_lower_box"]["model"]:
        raise AssertionError("raw mode should score the raw lower-box CO2 signature")
    check_close(
        "preferred resolves to column fixed",
        preferred["modern_CO2_strat_D17O"]["model"],
        column_fixed["modern_CO2_strat_D17O"]["model"],
        1.0e-12,
    )
    summaries = summarize(
        score_preset(preset, processed_mode="preferred")
        + score_preset(preset, processed_mode="exact")
        + score_preset(preset, processed_mode="exact_fixed")
        + score_preset(preset, processed_mode="column_extra")
        + score_preset(preset, processed_mode="column_fixed")
        + score_preset(preset, processed_mode="raw")
    )
    labels = {row["preset"] for row in summaries}
    expected = {
        preset,
        f"{preset} [exact]",
        f"{preset} [exact_fixed]",
        f"{preset} [column_extra]",
        f"{preset} [column_fixed]",
        f"{preset} [raw]",
    }
    if labels != expected:
        raise AssertionError(f"unexpected processed-mode labels: {labels}")
    by_label = {row["preset"]: row for row in summaries}
    if not by_label[f"{preset} [raw]"]["overall_score_percent"] < by_label[preset]["overall_score_percent"]:
        raise AssertionError("raw lower-box score should stay below processed-score modes")


def test_physical_extrapolation_preset_extrapolates_monotonically() -> None:
    """The gate-off physics branch must stay anchored, reproduce Young, and
    extrapolate monotonically across the literature pO2/pCO2/GPP envelopes."""
    preset = "physical_extrapolation"

    def d17(pco2: float, po2: float, gpp: float) -> float:
        return float(
            run_scenario(
                ScenarioInput(preset=preset, p_co2_ppm=pco2, p_o2_pal=po2, gpp_scale=gpp)
            ).outputs["O2_trop_D17O_permil"]
        )

    # Modern anchor stays near the Young/Pack modern O2 Delta'17O.
    modern = d17(294.4, 1.0, 1.0)
    assert -0.55 < modern < -0.33, f"modern O2 Delta'17O out of range: {modern}"

    # Monotonic in pCO2 (more negative with more CO2) across the full regime,
    # including low pO2 and reduced GPP where the old gate switches were placed.
    for po2 in (2.1, 1.0, 0.5, 0.1):
        for gpp in (1.0, 0.5):
            seq = [d17(p, po2, gpp) for p in (1000.0, 10000.0, 30000.0, 60000.0, 100000.0)]
            assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), (
                f"non-monotonic pCO2 response at pO2={po2}, gpp={gpp}: {seq}"
            )

    # More negative at lower pO2 and at lower GPP (fixed other inputs).
    assert d17(30000.0, 0.5, 1.0) < d17(30000.0, 1.0, 1.0)
    assert d17(30000.0, 1.0, 0.5) < d17(30000.0, 1.0, 1.0)

    # The application regime down to about -20 per mil is reachable.
    assert d17(100000.0, 0.5, 0.5) < -18.0


def test_extrapolation_bounds_literature_band() -> None:
    """Opt-in bound reporting adds a literature O(1D) Delta'17O band: tight near
    modern, wider toward extreme states, off by default."""
    # Off by default: no bound fields, no extra cost.
    plain = run_scenario(ScenarioInput(preset="physical_extrapolation"))
    assert not any("bound" in k or "cap" in k for k in plain.outputs)

    def bounds(pco2: float, po2: float, gpp: float) -> dict:
        return run_scenario(
            ScenarioInput(
                preset="physical_extrapolation",
                p_co2_ppm=pco2,
                p_o2_pal=po2,
                gpp_scale=gpp,
                report_extrapolation_bounds=True,
            )
        ).outputs

    near = bounds(294.4, 1.0, 1.0)
    far = bounds(60000.0, 0.5, 0.5)
    for o in (near, far):
        assert o["O2_trop_D17O_bound_low_permil"] <= o["O2_trop_D17O_bound_high_permil"]
        # estimate sits at/above the more-negative (Barkan-Luz) bound
        assert o["O2_trop_D17O_bound_low_permil"] - 1e-6 <= o["O2_trop_D17O_permil"]
        # nothing can be more negative than the physical cap
        assert o["O2_trop_D17O_bound_low_permil"] >= O2_CAP_BARKAN_LUZ_PERMIL - 5.0
    # Literature uncertainty is small near modern and larger toward extreme states.
    assert near["O2_trop_D17O_bound_spread_permil"] < far["O2_trop_D17O_bound_spread_permil"]


def test_dynvar_streamfunction_sign_and_passive_age_solver() -> None:
    """DynVarMIP Psi must produce tropical upwelling after adapter conversion."""

    latitude_ascending = np.arange(-90.0, 91.0, 10.0)
    pressure_descending = np.asarray([30000.0, 10000.0, 3.0])
    dynvar_streamfunction = np.zeros((19, 3), dtype=float)
    dynvar_streamfunction[:, 1] = 1.0e9 * np.sin(
        2.0 * np.deg2rad(latitude_ascending)
    )
    climatology = Era5TemClimatology(
        latitude_degrees=latitude_ascending[::-1],
        pressure_pa=pressure_descending[::-1],
        streamfunction_kg_per_s=dynvar_streamfunction.T[::-1, ::-1],
        month_count=1,
        total_weight_hours=1.0,
        start_month="synthetic",
        end_month="synthetic",
        source="synthetic DynVarMIP sign test",
    )
    synthetic_profile = ValidatedVerticalProfile(
        name="synthetic pressure profile",
        cells=tuple(
            VerticalCell(
                lower_altitude_km=lower,
                upper_altitude_km=upper,
                air_moles=1.0,
                number_density_molecules_cm3=number_density,
                temperature_k=250.0,
                pressure_center_bar=pressure_bar,
                eddy_diffusivity_cm2_per_s=1.0e-30,
            )
            for lower, upper, pressure_bar, number_density in (
                (0.0, 10.0, 2.0, 2.0e19),
                (10.0, 40.0, 0.5, 5.0e18),
                (40.0, 70.0, 0.05, 5.0e17),
                (70.0, 100.0, 1.0e-5, 1.0e12),
            )
        ),
        atmospheric_state_source="synthetic",
        eddy_diffusivity_source="synthetic near-zero Kzz",
    )
    transport = build_passive_age_transport(climatology, synthetic_profile)
    tropical_southern_fluxes = [
        flux
        for flux in transport.circulation.fluxes
        if flux.source_layer == latitude_vertical_cell_name(8, 1)
        and flux.target_layer == latitude_vertical_cell_name(8, 2)
    ]
    if len(tropical_southern_fluxes) != 1:
        raise AssertionError(
            "DynVarMIP sign conversion should produce tropical upwelling"
        )
    result = solve_passive_mean_age(transport)
    if np.min(result.mean_age_years) < 0.0:
        raise AssertionError("passive mean age must remain non-negative")
    if result.active_equation_max_residual_years_per_year > 1.0e-10:
        raise AssertionError("passive mean-age equations did not close")
    circulation_only = select_passive_transport_components(
        transport,
        include_vertical_diffusion=False,
        include_meridional_diffusion=False,
    )
    if not np.allclose(
        circulation_only.inventory_transport_matrix_per_year(),
        transport.circulation.transport_matrix_per_year(),
    ):
        raise AssertionError("component selector did not isolate circulation")
    interface_altitudes = 0.5 * (
        transport.altitude_centers_km[:-1]
        + transport.altitude_centers_km[1:]
    )
    split_altitude = float(interface_altitudes[1])
    lower_kzz = select_vertical_diffusion_altitude_domain(
        transport,
        maximum_interface_altitude_km=split_altitude,
    )
    upper_kzz = select_vertical_diffusion_altitude_domain(
        transport,
        minimum_interface_altitude_km=split_altitude,
    )
    retained = (
        lower_kzz.vertical_diffusion.interfaces
        + upper_kzz.vertical_diffusion.interfaces
    )
    if len(retained) != len(transport.vertical_diffusion.interfaces):
        raise AssertionError("altitude domains must partition all Kzz interfaces")
    original_rates = {
        (interface.first_layer, interface.second_layer):
        interface.gross_air_flux_mol_per_year
        for interface in transport.vertical_diffusion.interfaces
    }
    for interface in retained:
        key = (interface.first_layer, interface.second_layer)
        check_close(
            "altitude-selected Kzz interface rate",
            interface.gross_air_flux_mol_per_year,
            original_rates[key],
            0.0,
        )
    tropospheric_kzz = select_vertical_diffusion_below_local_tropopause(transport)
    for interface in tropospheric_kzz.vertical_diffusion.interfaces:
        first = transport.vertical_diffusion.layer_names.index(interface.first_layer)
        second = transport.vertical_diffusion.layer_names.index(interface.second_layer)
        latitude_index, vertical_index = divmod(
            first, transport.air_moles.shape[1]
        )
        second_latitude_index, second_vertical_index = divmod(
            second, transport.air_moles.shape[1]
        )
        if latitude_index != second_latitude_index:
            raise AssertionError("vertical Kzz interface changed latitude")
        if second_vertical_index != vertical_index + 1:
            raise AssertionError("vertical Kzz interface does not join adjacent cells")
        if not (
            transport.reset_mask[latitude_index, vertical_index]
            and transport.reset_mask[second_latitude_index, second_vertical_index]
        ):
            raise AssertionError(
                "local-tropopause selector retained a cross-tropopause Kzz interface"
            )
        key = (interface.first_layer, interface.second_layer)
        check_close(
            "local-tropopause Kzz interface rate",
            interface.gross_air_flux_mol_per_year,
            original_rates[key],
            0.0,
        )


def test_meridional_eddy_diffusion_is_conservative() -> None:
    latitude_edges = np.asarray([-90.0, -30.0, 30.0, 90.0])
    pressure_edges = np.asarray([100000.0, 50000.0])
    air = hydrostatic_air_moles(latitude_edges, pressure_edges)
    kyy = np.full((2, 1), 1.0e6)
    operator = meridional_eddy_diffusion_operator(
        air_moles=air,
        latitude_edges_degrees=latitude_edges,
        pressure_edges_pa=pressure_edges,
        kyy_m2_per_s=kyy,
        source="synthetic Kyy conservation test",
    )
    if len(operator.interfaces) != 2:
        raise AssertionError("Kyy operator should contain one exchange per internal face")
    tracer = 0.25 * air.ravel()
    tendency = operator.derivative(tracer)
    if float(np.max(np.abs(tendency / air.ravel()))) > 1.0e-12:
        raise AssertionError("Kyy should leave a uniform mixing ratio stationary")
    nonuniform_tracer = air.ravel() * np.asarray([0.1, 0.2, 0.3])
    nonuniform_tendency = operator.derivative(nonuniform_tracer)
    tendency_scale = max(float(np.sum(np.abs(nonuniform_tendency))), 1.0)
    residual = float(operator.conservation_residual(nonuniform_tracer))
    if abs(residual) / tendency_scale > 1.0e-12:
        raise AssertionError("Kyy operator should conserve tracer inventory")


def test_gridded_species_transport_conserves_each_isotopologue() -> None:
    """One air operator must conserve 16O, 17O, and 18O species separately."""

    rate = 0.4
    air = np.asarray([2.0, 3.0, 5.0])
    matrix = np.asarray(
        [
            [-rate / air[0], rate / air[1], 0.0],
            [rate / air[0], -2.0 * rate / air[1], rate / air[2]],
            [0.0, rate / air[1], -rate / air[2]],
        ]
    )
    system = GriddedSpeciesSystem(
        species_names=("O2", "O17O", "O18O"),
        air_moles=air,
        inventory_transport_matrix_per_year=matrix,
        source="synthetic three-cell isotope transport",
    )
    inventory = np.asarray(
        [
            [1.0, 8.0, 3.0],
            [2.0e-3, 1.0e-3, 4.0e-3],
            [5.0e-3, 9.0e-3, 2.0e-3],
        ]
    )
    tendency = system.transport_derivative_mol_per_year(inventory)
    if not np.allclose(np.sum(tendency, axis=1), 0.0, atol=1.0e-15):
        raise AssertionError("transport must conserve every isotopologue independently")
    if not np.allclose(system.species_conservation_residual_mol_per_year(inventory), 0.0):
        raise AssertionError("reported species conservation residual is nonzero")

    uniform_ratio = np.asarray([0.21, 0.21, 0.21]) * system.air_moles
    uniform_inventory = np.vstack((uniform_ratio, 1.0e-3 * uniform_ratio, 2.0e-3 * uniform_ratio))
    if not np.allclose(
        system.transport_derivative_mol_per_year(uniform_inventory),
        0.0,
        atol=1.0e-15,
    ):
        raise AssertionError("transport must leave uniform mixing ratios stationary")


def test_gridded_species_transport_keeps_chemistry_and_boundaries_explicit() -> None:
    system = GriddedSpeciesSystem(
        species_names=("CO2", "CO17O", "CO18O"),
        air_moles=np.asarray([1.0, 1.0]),
        inventory_transport_matrix_per_year=np.zeros((2, 2)),
        source="synthetic chemistry coupling test",
    )
    inventory = np.ones((3, 2))
    chemistry = np.asarray([[-1.0, -1.0], [-0.01, -0.01], [-0.02, -0.02]])
    boundary = -chemistry
    tendency = system.derivative_mol_per_year(
        0.0,
        inventory,
        local_chemistry=lambda _time, _state: chemistry,
        boundary_flux_mol_per_year=boundary,
    )
    if not np.array_equal(tendency, np.zeros_like(inventory)):
        raise AssertionError("explicit chemistry and boundary terms should close exactly")
    atoms = system.atom_inventory_moles(
        inventory,
        {
            "CO2": {"C": 1.0, "16O": 2.0},
            "CO17O": {"C": 1.0, "16O": 1.0, "17O": 1.0},
            "CO18O": {"C": 1.0, "16O": 1.0, "18O": 1.0},
        },
    )
    assert atoms == {"C": 6.0, "16O": 8.0, "17O": 2.0, "18O": 2.0}


def test_gridded_oxygen_reaction_kernel_preserves_isotope_atoms() -> None:
    reactions = (
        ElementaryGridReaction(
            key="O3_photolysis",
            reactants={"O3": 1},
            products={"O2": 1, "O": 1},
            rate_coefficient=2.0e-4,
            source="synthetic first-order photolysis",
        ),
        ElementaryGridReaction(
            key="CO2_O17_1D_exchange",
            reactants={"CO2": 1, "O17_1D": 1},
            products={"CO17O": 1, "O": 1},
            rate_coefficient=np.asarray([4.0e-11, 5.0e-11]),
            source="synthetic isotope exchange",
        ),
    )
    validate_atom_balanced_reactions(reactions)
    assert reaction_atom_residual(reactions[0]) == {}
    assert reaction_atom_residual(reactions[1]) == {}

    air = np.asarray([2.0e10, 1.0e10])
    pressure = np.asarray([1.0e3, 5.0e2])
    temperature = np.asarray([230.0, 240.0])
    mixing_ratios = np.zeros((len(ATMOSPHERIC_OXYGEN_SPECIES), len(air)))
    index = {name: i for i, name in enumerate(ATMOSPHERIC_OXYGEN_SPECIES)}
    mixing_ratios[index["O3"]] = 5.0e-6
    mixing_ratios[index["O2"]] = 0.21
    mixing_ratios[index["CO2"]] = 400.0e-6
    mixing_ratios[index["O17_1D"]] = 1.0e-15
    inventory = mixing_ratios * air[None, :]
    tendency = local_reaction_tendency_mol_per_year(
        inventory,
        species_names=ATMOSPHERIC_OXYGEN_SPECIES,
        air_moles=air,
        pressure_pa=pressure,
        temperature_k=temperature,
        reactions=reactions,
    )
    for atom in ("C", "16O", "17O", "18O"):
        atom_tendency = sum(
            ATMOSPHERIC_ATOM_COUNTS[species].get(atom, 0.0)
            * np.sum(tendency[species_index])
            for species_index, species in enumerate(ATMOSPHERIC_OXYGEN_SPECIES)
        )
        scale = max(float(np.sum(np.abs(tendency))), 1.0)
        if abs(atom_tendency) / scale > 1.0e-14:
            raise AssertionError(f"local chemistry does not conserve {atom}")


def test_gridded_oxygen_reaction_kernel_rejects_unbalanced_reaction() -> None:
    invalid = ElementaryGridReaction(
        key="invalid_ozone_loss",
        reactants={"O3": 1},
        products={"O2": 1},
        rate_coefficient=1.0e-4,
        source="synthetic invalid reaction",
    )
    try:
        validate_atom_balanced_reactions((invalid,))
    except ValueError as error:
        if "16O" not in str(error):
            raise AssertionError("atom-balance error should identify the lost isotope")
    else:
        raise AssertionError("unbalanced grid reaction was accepted")


def test_finite_clock_one_cell_analytic_limit() -> None:
    """A one-cell clock must follow (1-exp(-kt))/k and preserve constants."""

    class OneCellTransport:
        air_moles = np.ones((1, 2), dtype=float)
        reset_mask = np.asarray([[True, False]])
        altitude_centers_km = np.asarray([0.0, 20.0])
        latitude_edges_degrees = np.asarray([-5.0, 5.0])

        @property
        def latitude_centers_degrees(self) -> np.ndarray:
            return np.asarray([0.0])

        def mixing_ratio_transport_matrix_per_year(self) -> np.ndarray:
            rate = 0.5
            return np.asarray([[0.0, 0.0], [rate, -rate]])

    transport = OneCellTransport()
    time = np.asarray([0.0, 1.0, 4.0, 20.0])
    result = solve_finite_clock_age(transport, time)
    expected = (1.0 - np.exp(-0.5 * time)) / 0.5
    if not np.allclose(result.mean_age_years[:, 0, 1], expected, atol=1.0e-12):
        raise AssertionError("finite clock failed the one-cell analytic solution")
    if not np.allclose(result.clock_tracer_years[:, 0, 0], time):
        raise AssertionError("clock boundary must equal elapsed time")

    constant = propagate_boundary_history(
        transport,
        boundary_time_years=np.asarray([0.0, 10.0]),
        boundary_values=np.asarray([3.0, 3.0]),
        output_time_years=np.asarray([0.0, 2.0, 10.0]),
    )
    if not np.allclose(constant.tracer_mixing_ratio, 3.0, atol=1.0e-10):
        raise AssertionError("uniform constant tracer should remain uniform")
    periodic = solve_periodic_clock_age(
        (transport, transport),
        np.asarray([0.4, 0.6]),
    )
    if not np.allclose(
        periodic.duration_weighted_annual_mean_age_years[0, 1],
        2.0,
        atol=2.0e-9,
    ):
        raise AssertionError("periodic clock failed its constant-operator limit")
    if periodic.annual_cycle_closure_max_years > 1.0e-10:
        raise AssertionError("periodic clock did not close its annual cycle")


def test_e90_one_cell_source_loss_limit() -> None:
    class OneCellSurfaceTransport:
        air_moles = np.ones((1, 1), dtype=float)

        def mixing_ratio_transport_matrix_per_year(self) -> np.ndarray:
            return np.zeros((1, 1), dtype=float)

    transport = OneCellSurfaceTransport()
    steady = solve_e90_steady(transport)
    if not np.allclose(steady.mixing_ratio_ppb, 100.0, atol=1.0e-12):
        raise AssertionError("one-cell E90 must equal its target global mean")
    if not np.isclose(
        steady.surface_source_ppb_per_year,
        100.0 / E90_LIFETIME_YEARS,
        rtol=1.0e-12,
    ):
        raise AssertionError("one-cell E90 source does not balance 90-day loss")
    if steady.maximum_tendency_residual_ppb_per_year > 1.0e-10:
        raise AssertionError("one-cell E90 steady tendency does not close")
    periodic = solve_e90_periodic(
        (transport, transport),
        np.asarray([0.4, 0.6]),
    )
    if not np.allclose(
        periodic.mixing_ratio_at_interval_midpoint_ppb,
        100.0,
        atol=1.0e-9,
    ):
        raise AssertionError("constant periodic E90 must equal the steady state")
    if periodic.annual_cycle_maximum_relative_closure > 1.0e-10:
        raise AssertionError("periodic E90 annual cycle did not close")


def test_randel_garcia_qgpv_gradient_resting_atmosphere() -> None:
    """Equation 2 must reduce to the smoothed planetary PV gradient."""

    latitude = np.arange(-80.0, 81.0, 10.0)
    pressure = np.asarray([100000.0, 50000.0, 10000.0, 1000.0])
    wind = np.zeros((len(pressure), len(latitude)))
    temperature = np.full_like(wind, 250.0)
    gradient, stability = quasigeostrophic_pv_gradient_per_m_s(
        latitude_degrees=latitude,
        pressure_pa=pressure,
        zonal_wind_m_per_s=wind,
        temperature_k=temperature,
    )
    expected_row = (
        2.0
        * EARTH_ROTATION_PER_S
        / 6_371_000.0
        * np.cos(np.deg2rad(latitude))
    )
    expected_row[1:-1] = (
        expected_row[:-2]
        + 2.0 * expected_row[1:-1]
        + expected_row[2:]
    ) / 4.0
    if not np.allclose(gradient, expected_row[None, :], rtol=1.0e-13):
        raise AssertionError("Randel-Garcia QGPV gradient failed beta-only limit")
    if np.any(stability <= 0.0):
        raise AssertionError("synthetic isothermal atmosphere should be stable")


def test_summers_table1_and_zero_phase_drag() -> None:
    """Summers Table 1 and Equation 1 must be represented literally."""

    assert [(wave.phase_speed_m_per_s, wave.source_displacement_m, wave.multiplicity) for wave in SUMMERS_TABLE1_WAVES] == [
        (40.0, 15.0, 3),
        (40.0, 30.0, 3),
        (20.0, 40.0, 5),
        (-20.0, 40.0, 5),
        (-40.0, 30.0, 3),
        (-40.0, 15.0, 3),
    ]
    latitude = np.asarray([-45.0, 45.0])
    altitude = np.asarray([45_000.0, 55_000.0, 65_000.0, 75_000.0])
    wind = np.full((2, 4), 60.0)
    drag = zero_phase_drag_m_per_s2(latitude, altitude, wind)
    assert drag[0, 0] == 0.0
    assert drag[0, 1] == 0.0
    check_close("southern ramp", drag[0, 2], -0.5 * GAMMA1_S_PER_M2 * 60.0**3, 0.0)
    assert drag[1, 0] == 0.0
    check_close("northern ramp", drag[1, 1], -0.5 * GAMMA1_S_PER_M2 * 60.0**3, 0.0)
    check_close("northern full gamma", drag[1, 3], -GAMMA1_S_PER_M2 * 60.0**3, 0.0)


def test_summers_unsaturated_wave_conserves_momentum_flux() -> None:
    """Equations 2-3 must give zero Kzz while a wave remains unsaturated."""

    altitude = SOURCE_ALTITUDE_M + VERTICAL_SPACING_M * np.arange(5)
    density = np.exp(-altitude / 7_000.0)
    frequency = np.full(5, 0.02)
    wind = np.zeros(5)
    kzz, drag, critical_level = _wave_kzz_column(
        altitude_m=altitude,
        density_kg_per_m3=density,
        brunt_vaisala_per_s=frequency,
        zonal_wind_m_per_s=wind,
        wave=GravityWave(40.0, 0.01, 1),
        prandtl_number=PRANDTL_NUMBER,
    )
    if not np.allclose(kzz, 0.0, atol=1.0e-12):
        raise AssertionError("unsaturated propagation must conserve momentum flux")
    if critical_level is not None:
        raise AssertionError("constant intrinsic speed must not create a critical level")
    if not np.allclose(drag, 0.0, atol=1.0e-12):
        raise AssertionError("unsaturated propagation must produce zero drag")


def test_summers_wave_is_absorbed_at_critical_level() -> None:
    """A wave must deposit its flux once and not reappear above U=c."""

    altitude = SOURCE_ALTITUDE_M + VERTICAL_SPACING_M * np.arange(7)
    density = np.exp(-altitude / 7_000.0)
    frequency = np.full(7, 0.02)
    wind = np.asarray([0.0, 10.0, 25.0, 35.0, 45.0, 55.0, 65.0])
    kzz, drag, critical_level = _wave_kzz_column(
        altitude_m=altitude,
        density_kg_per_m3=density,
        brunt_vaisala_per_s=frequency,
        zonal_wind_m_per_s=wind,
        wave=GravityWave(40.0, 10.0, 1),
        prandtl_number=PRANDTL_NUMBER,
    )
    if critical_level != 4:
        raise AssertionError("critical-level crossing was not detected")
    if kzz[critical_level] <= 0.0:
        raise AssertionError("remaining momentum flux must be deposited at absorption")
    if np.any(kzz[critical_level + 1 :] != 0.0):
        raise AssertionError("an absorbed wave must not reappear aloft")
    if drag[critical_level] == 0.0:
        raise AssertionError("critical-level absorption must deposit momentum")


def test_summers_native_grid_returns_nonnegative_components() -> None:
    altitude = SOURCE_ALTITUDE_M + VERTICAL_SPACING_M * np.arange(25)
    pressure = 100_000.0 * np.exp(-altitude / 7_000.0)
    latitude = np.asarray([-60.0, 60.0])
    shape = (2, len(altitude))
    result = summers_kzz_on_native_grid(
        latitude_degrees=latitude,
        altitude_m=altitude,
        pressure_pa=pressure,
        temperature_k=np.full(shape, 230.0),
        zonal_wind_m_per_s=np.tile(np.linspace(5.0, 70.0, len(altitude)), (2, 1)),
        static_stability_per_s2=np.full(shape, 4.0e-4),
    )
    if np.any(result.kzz_m2_per_s < 0.0):
        raise AssertionError("Summers Kzz must be non-negative")
    if not np.allclose(
        result.kzz_m2_per_s,
        result.zero_phase_kzz_m2_per_s + result.nonzero_phase_kzz_m2_per_s,
    ):
        raise AssertionError("Summers wave components must close")
    if not np.allclose(
        result.gravity_wave_drag_m_per_s2,
        result.zero_phase_drag_m_per_s2 + result.nonzero_phase_drag_m_per_s2,
    ):
        raise AssertionError("Summers drag components must close")


def test_merra2_archive_metadata_is_pinned() -> None:
    expected = {
        "DUDTGWD": "9cfce6764ec688d09458672881f089ca",
        "u": "c78ca35b1d81d23ee2d8a1aadd571e18",
        "t": "9e2d1849f36ca1fadfbc8905e73131e4",
    }
    if {key: value["md5"] for key, value in MERRA2_DAILY_ARCHIVES.items()} != expected:
        raise AssertionError("MERRA-2 archive provenance changed")


if __name__ == "__main__":
    test_young_table3_delta()
    test_full_table3_source_inventory_is_encoded()
    test_reduced_mass_alpha()
    test_phanerozoic_o2_conversions()
    test_reaction_engine()
    test_reaction_inventory_counts()
    test_source_row_audit_coverage()
    test_young_r2_mass_dependent_ranges()
    test_executable_reaction_count()
    test_r8_rate_scales()
    test_co2_photo_sink_factor_scales_default_mode()
    test_model_runner_table3_isotopes()
    test_fast_stratosphere_solver_table3_ozone()
    test_tropospheric_co2_isotope_solver_converges()
    test_coupled_isotope_subsystem_converges()
    test_fixed_reservoir_solver_converges()
    test_o2_mass_balance_branch_zeroes_heavy_o2_tendencies()
    test_two_box_export_branch_reduces_export_flux_not_bulk_state()
    test_explicit_lower_threshold_candidate_matches_fig8_endpoint()
    test_explicit_lower_net_export_rate_only_scales_flux()
    test_explicit_lower_split_candidate_uses_transport_bounded_rates()
    test_processed_altitude_reservoir_repairs_modern_co2_diagnostics()
    test_physical_extrapolation_preset_extrapolates_monotonically()
    test_extrapolation_bounds_literature_band()
    test_dynvar_streamfunction_sign_and_passive_age_solver()
    test_meridional_eddy_diffusion_is_conservative()
    test_finite_clock_one_cell_analytic_limit()
    test_e90_one_cell_source_loss_limit()
    test_randel_garcia_qgpv_gradient_resting_atmosphere()
    test_summers_table1_and_zero_phase_drag()
    test_summers_unsaturated_wave_conserves_momentum_flux()
    test_summers_wave_is_absorbed_at_critical_level()
    test_summers_native_grid_returns_nonnegative_components()
    test_merra2_archive_metadata_is_pinned()
    test_young_reproduction_uses_source_derived_bulk_fig10()
    test_young_reproduction_uses_source_derived_bulk_fig9()
    test_young_transient_dispatch_preserves_raw_bulk_baselines()
    test_source_derived_young_bulk_fig10_component()
    print("core checks passed")
