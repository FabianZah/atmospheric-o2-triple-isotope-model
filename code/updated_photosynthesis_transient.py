"""Updated-model response to photosynthesis changes at fixed respiration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from global_o2_isotope_reservoir import GlobalO2Reservoir, IsotopologueTendency, sum_tendencies
from gpp_normalization import YOUNG_MODERN_GPP_PGC_PER_YEAR
from integrate_fig9_fig10_transients import integrate_photosynthesis_carbon_driver
from model_scenarios import CURRENT_PUBLICATION_PRESET, ScenarioInput, config_from_scenario
from updated_molecular_forward_model import (
    UpdatedForwardInput,
    build_updated_central_forcing,
    run_updated_central_state,
)
from updated_molecular_transient import OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


@dataclass(frozen=True)
class UpdatedPhotosynthesisTransientInput:
    initial: UpdatedForwardInput
    photosynthesis_fraction: float = 0.5
    duration_years: float = 12000.0
    sample_count: int = 161
    equilibrium_search_max_years: float = 100000.0


@dataclass(frozen=True)
class UpdatedPhotosynthesisTransientResult:
    request: UpdatedPhotosynthesisTransientInput
    time_years: tuple[float, ...]
    states: tuple[GlobalO2Reservoir, ...]
    pco2_ppm: tuple[float, ...]
    carbon_driver_po2_pal: tuple[float, ...]
    long_run_state: GlobalO2Reservoir
    long_run_pco2_ppm: float
    operational_equilibrium_time_years: float | None
    model_data_id: str
    carbon_driver_preset: str
    solver: dict[str, Any]

    def rows(self) -> list[dict[str, float | str]]:
        return [
            {
                "time_yr": time,
                "pO2_pal": state.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL,
                "pCO2_ppm": pco2,
                "carbon_driver_pO2_pal": carbon_po2,
                "O2_delta18_prime_permil": state.delta18_prime_permil,
                "O2_cap_delta17_prime_permil": state.cap_delta17_prime_permil,
                "model_data_id": self.model_data_id,
            }
            for time, state, pco2, carbon_po2 in zip(
                self.time_years,
                self.states,
                self.pco2_ppm,
                self.carbon_driver_po2_pal,
                strict=True,
            )
        ]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["states"] = [
            {
                "o16o16_mol": state.o16o16,
                "o16o17_mol": state.o16o17,
                "o16o18_mol": state.o16o18,
                "delta18_prime_permil": state.delta18_prime_permil,
                "cap_delta17_prime_permil": state.cap_delta17_prime_permil,
            }
            for state in self.states
        ]
        data["long_run_state"] = {
            "pO2_pal": self.long_run_state.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL,
            "pCO2_ppm": self.long_run_pco2_ppm,
            "delta18_prime_permil": self.long_run_state.delta18_prime_permil,
            "cap_delta17_prime_permil": self.long_run_state.cap_delta17_prime_permil,
        }
        data["operational_equilibrium"] = {
            "criterion": "both isotope coordinates enter and remain within tolerance of the long-run state",
            "tolerance_permil": OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL,
            "time_years": self.operational_equilibrium_time_years,
        }
        return data


def _equilibrium_time(
    times: np.ndarray,
    states: tuple[GlobalO2Reservoir, ...],
    target: GlobalO2Reservoir,
) -> float | None:
    distances = np.asarray(
        [
            max(
                abs(state.cap_delta17_prime_permil - target.cap_delta17_prime_permil),
                abs(state.delta18_prime_permil - target.delta18_prime_permil),
            )
            for state in states
        ]
    )
    within = distances <= OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL
    for index in range(len(within)):
        if not np.all(within[index:]):
            continue
        if index == 0:
            return float(times[0])
        previous = distances[index - 1]
        current = distances[index]
        if previous > current:
            fraction = (
                (previous - OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL)
                / (previous - current)
            )
            return float(times[index - 1] + fraction * (times[index] - times[index - 1]))
        return float(times[index])
    return None


def run_updated_photosynthesis_transient(
    request: UpdatedPhotosynthesisTransientInput,
) -> UpdatedPhotosynthesisTransientResult:
    """Halve or scale photosynthesis while retaining initial respiration rates."""

    values = np.asarray(
        (
            request.photosynthesis_fraction,
            request.duration_years,
            request.equilibrium_search_max_years,
        )
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("photosynthesis fraction and transient times must be positive")
    if request.sample_count < 2:
        raise ValueError("transient requires at least two output samples")

    horizon = max(request.duration_years, request.equilibrium_search_max_years)
    initial_state = run_updated_central_state(request.initial)
    forcing = build_updated_central_forcing(request.initial)
    initial_reservoir = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=request.initial.p_o2_pal * GLOBAL_MAJOR_O2_MOLES_1PAL,
        delta18_prime_permil=initial_state.delta18_prime_permil,
        cap_delta17_prime_permil=initial_state.cap_delta17_prime_permil,
        source="updated-model pre-perturbation equilibrium",
    )
    initial_photo = forcing.photochemical_tendency(
        np.asarray(
            [initial_state.delta18_prime_permil, initial_state.cap_delta17_prime_permil]
        )
    )
    biology = forcing.biological_budget(initial_photo).scaled_photosynthesis(
        request.photosynthesis_fraction,
        source=(
            "updated photosynthesis perturbation at fixed pre-perturbation "
            "respiration coefficients"
        ),
    )

    carbon_config = config_from_scenario(
        ScenarioInput(
            preset=CURRENT_PUBLICATION_PRESET,
            p_o2_pal=request.initial.p_o2_pal,
            p_co2_ppm=request.initial.p_co2_ppm,
            gpp_scale=(
                request.initial.gpp_pgC_per_year / YOUNG_MODERN_GPP_PGC_PER_YEAR
            ),
            gpp_normalization="young_2014",
            solve_mode="full_atmosphere",
        )
    )
    carbon_driver = integrate_photosynthesis_carbon_driver(
        carbon_config,
        photosynthesis_fraction=request.photosynthesis_fraction,
        duration_years=horizon,
        sample_count=max(request.sample_count, 201),
    )
    carbon_time = carbon_driver.time_years
    carbon_pco2 = carbon_driver.pco2_ppm
    carbon_po2 = carbon_driver.po2_pal
    carbon_solved = carbon_driver.solver

    scale = initial_reservoir.isotopologue_moles

    def tendency(time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * scale),
            source="updated photosynthesis-transient state",
        )
        live_po2 = reservoir.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL
        live_pco2 = float(np.interp(time, carbon_time, carbon_pco2))
        native = forcing.surface.evaluate_prime_tendency_at(
            np.asarray(
                [reservoir.delta18_prime_permil, reservoir.cap_delta17_prime_permil]
            ),
            po2_pal=live_po2,
            pco2_ppm=live_pco2,
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        photo = IsotopologueTendency(
            *map(float, forcing.forcing_scale * native.values),
            source="updated molecular R7 forcing at live pO2 and pCO2",
        )
        return sum_tendencies(
            (biology.tendency(reservoir), photo),
            source="fixed-respiration biology plus live updated R7 forcing",
        ).values / scale

    solved = solve_ivp(
        tendency,
        (0.0, horizon),
        initial_reservoir.isotopologue_moles / scale,
        method="DOP853",
        dense_output=True,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=min(100.0, horizon / 100.0),
    )
    if not solved.success or solved.sol is None:
        raise RuntimeError(f"updated photosynthesis transient failed: {solved.message}")

    display_time = np.linspace(0.0, request.duration_years, request.sample_count)
    display_inventory = solved.sol(display_time).T * scale
    display_states = tuple(
        GlobalO2Reservoir(*map(float, inventory), source=f"state at {time:g} yr")
        for time, inventory in zip(display_time, display_inventory, strict=True)
    )
    search_time = np.linspace(0.0, horizon, max(2001, int(horizon / 50.0) + 1))
    search_inventory = solved.sol(search_time).T * scale
    search_states = tuple(
        GlobalO2Reservoir(*map(float, inventory), source=f"search state at {time:g} yr")
        for time, inventory in zip(search_time, search_inventory, strict=True)
    )
    long_run_state = search_states[-1]
    equilibrium_time = _equilibrium_time(search_time, search_states, long_run_state)

    return UpdatedPhotosynthesisTransientResult(
        request=request,
        time_years=tuple(map(float, display_time)),
        states=display_states,
        pco2_ppm=tuple(map(float, np.interp(display_time, carbon_time, carbon_pco2))),
        carbon_driver_po2_pal=tuple(
            map(float, np.interp(display_time, carbon_time, carbon_po2))
        ),
        long_run_state=long_run_state,
        long_run_pco2_ppm=float(carbon_pco2[-1]),
        operational_equilibrium_time_years=equilibrium_time,
        model_data_id=forcing.model_data_id,
        carbon_driver_preset=CURRENT_PUBLICATION_PRESET,
        solver={
            "oxygen_method": "scipy.solve_ivp DOP853",
            "oxygen_success": bool(solved.success),
            "oxygen_function_evaluations": int(solved.nfev),
            "carbon_method": "detailed atmospheric box-model BDF driver",
            "carbon_success": bool(carbon_solved.success),
            "carbon_function_evaluations": int(carbon_solved.nfev),
            "carbon_internal_runtime_warning_events": int(
                getattr(carbon_solved, "internal_runtime_warning_events", 0)
            ),
            "carbon_internal_runtime_warnings": tuple(
                getattr(carbon_solved, "internal_runtime_warnings", ())
            ),
            "carbon_minimum_unclipped_inventory_mol": float(
                getattr(carbon_solved, "minimum_unclipped_inventory_mol", np.nan)
            ),
            "carbon_negative_inventory_count_before_clipping": int(
                getattr(carbon_solved, "negative_inventory_count_before_clipping", 0)
            ),
            "display_duration_years": request.duration_years,
            "equilibrium_search_horizon_years": horizon,
        },
    )
