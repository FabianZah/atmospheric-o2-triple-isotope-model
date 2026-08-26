"""Atmospheric O2 isotope response to a prescribed gradual pCO2 trajectory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp

from global_o2_isotope_reservoir import GlobalO2Reservoir, sum_tendencies
from updated_molecular_forward_model import (
    UpdatedForwardInput,
    build_updated_central_forcing,
    run_updated_central_state,
)
from updated_molecular_transient import OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


TrajectoryInterpolation = Literal["linear", "smoothstep"]


@dataclass(frozen=True)
class UpdatedPCO2TrajectoryInput:
    initial: UpdatedForwardInput
    final_pco2_ppm: float
    transition_duration_years: float
    interpolation: TrajectoryInterpolation = "smoothstep"
    duration_years: float = 12000.0
    sample_count: int = 181
    equilibrium_search_max_years: float = 100000.0


@dataclass(frozen=True)
class UpdatedPCO2TrajectoryResult:
    request: UpdatedPCO2TrajectoryInput
    time_years: tuple[float, ...]
    pco2_ppm: tuple[float, ...]
    states: tuple[GlobalO2Reservoir, ...]
    initial_steady_state: dict[str, float]
    final_steady_state: dict[str, float]
    model_data_id: str
    transfer_convention: str
    solver: dict[str, Any]
    equilibrium_time_years: float | None

    def rows(self) -> list[dict[str, float | str]]:
        return [
            {
                "time_yr": time,
                "pO2_pal": state.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL,
                "pCO2_ppm": pco2,
                "GPP_PgC_per_year": self.request.initial.gpp_pgC_per_year,
                "O2_delta18_prime_permil": state.delta18_prime_permil,
                "O2_cap_delta17_prime_permil": state.cap_delta17_prime_permil,
                "model_data_id": self.model_data_id,
            }
            for time, pco2, state in zip(
                self.time_years, self.pco2_ppm, self.states, strict=True
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
                "source": state.source,
            }
            for state in self.states
        ]
        data["operational_equilibrium"] = {
            "criterion": (
                "after the prescribed trajectory, both isotope coordinates "
                "enter and remain within tolerance of the final steady state"
            ),
            "tolerance_permil": OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL,
            "time_years": self.equilibrium_time_years,
        }
        return data


def trajectory_pco2_ppm(
    time_years: float,
    *,
    start_pco2_ppm: float,
    final_pco2_ppm: float,
    transition_duration_years: float,
    interpolation: TrajectoryInterpolation,
) -> float:
    """Evaluate the prescribed monotonic endpoint trajectory."""

    fraction = float(np.clip(time_years / transition_duration_years, 0.0, 1.0))
    if interpolation == "smoothstep":
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    elif interpolation != "linear":
        raise ValueError("pCO2 trajectory interpolation must be linear or smoothstep")
    return float(start_pco2_ppm + fraction * (final_pco2_ppm - start_pco2_ppm))


def _trajectory_sample_times(
    duration_years: float,
    transition_duration_years: float,
    sample_count: int,
) -> np.ndarray:
    """Resolve a short trajectory and its longer relaxation in one output grid."""

    if transition_duration_years >= duration_years or sample_count < 5:
        return np.linspace(0.0, duration_years, sample_count)
    transition_count = min(max(17, sample_count // 3), sample_count - 2)
    relaxation_count = sample_count - transition_count + 1
    return np.concatenate(
        (
            np.linspace(0.0, transition_duration_years, transition_count)[:-1],
            np.linspace(
                transition_duration_years,
                duration_years,
                relaxation_count,
            ),
        )
    )


def run_updated_pco2_trajectory(
    request: UpdatedPCO2TrajectoryInput,
) -> UpdatedPCO2TrajectoryResult:
    """Integrate the global O2 reservoir under a prescribed pCO2 trajectory."""

    numeric = np.asarray(
        (
            request.final_pco2_ppm,
            request.transition_duration_years,
            request.duration_years,
            request.equilibrium_search_max_years,
        ),
        dtype=float,
    )
    if np.any(~np.isfinite(numeric)) or np.any(numeric <= 0.0):
        raise ValueError("pCO2 trajectory values and times must be finite and positive")
    if request.sample_count < 2:
        raise ValueError("pCO2 trajectory requires at least two output samples")
    if request.interpolation not in {"linear", "smoothstep"}:
        raise ValueError("pCO2 trajectory interpolation must be linear or smoothstep")

    final_input = replace(request.initial, p_co2_ppm=request.final_pco2_ppm)
    initial_state = run_updated_central_state(request.initial)
    final_state = run_updated_central_state(final_input)
    forcing_template = build_updated_central_forcing(request.initial)
    major_o2 = request.initial.p_o2_pal * GLOBAL_MAJOR_O2_MOLES_1PAL
    initial_reservoir = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=major_o2,
        delta18_prime_permil=initial_state.delta18_prime_permil,
        cap_delta17_prime_permil=initial_state.cap_delta17_prime_permil,
        source="OXYTIB pre-trajectory steady reservoir",
    )
    final_reservoir = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=major_o2,
        delta18_prime_permil=final_state.delta18_prime_permil,
        cap_delta17_prime_permil=final_state.cap_delta17_prime_permil,
        source="OXYTIB final pCO2 trajectory steady reservoir",
    )
    scale = final_reservoir.isotopologue_moles

    def pco2_at(time: float) -> float:
        return trajectory_pco2_ppm(
            time,
            start_pco2_ppm=request.initial.p_co2_ppm,
            final_pco2_ppm=request.final_pco2_ppm,
            transition_duration_years=request.transition_duration_years,
            interpolation=request.interpolation,
        )

    def tendency(time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * scale),
            source="OXYTIB pCO2 trajectory integration state",
        )
        forcing = replace(
            forcing_template,
            request=replace(request.initial, p_co2_ppm=pco2_at(time)),
        )
        photo = forcing.photochemical_tendency(
            np.asarray(
                [
                    reservoir.delta18_prime_permil,
                    reservoir.cap_delta17_prime_permil,
                ]
            )
        )
        biology = forcing.biological_budget(photo)
        total = sum_tendencies(
            (biology.tendency(reservoir), photo),
            source=(
                "partitioned biological turnover plus state- and pCO2-dependent "
                "R7 forcing"
            ),
        )
        return total.values / scale

    def equilibrium_event(time: float, scaled_inventory: np.ndarray) -> float:
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * scale),
            source="OXYTIB pCO2 trajectory equilibrium event state",
        )
        distance = max(
            abs(
                reservoir.cap_delta17_prime_permil
                - final_state.cap_delta17_prime_permil
            ),
            abs(
                reservoir.delta18_prime_permil
                - final_state.delta18_prime_permil
            ),
        )
        if time < request.transition_duration_years:
            return distance + (request.transition_duration_years - time) / max(
                request.transition_duration_years, 1.0
            )
        return distance - OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL

    equilibrium_event.direction = 0.0
    equilibrium_event.terminal = False

    times = _trajectory_sample_times(
        request.duration_years,
        request.transition_duration_years,
        request.sample_count,
    )
    integration_horizon = max(
        request.duration_years,
        request.transition_duration_years,
        request.equilibrium_search_max_years,
    )
    solved = solve_ivp(
        tendency,
        (0.0, integration_horizon),
        initial_reservoir.isotopologue_moles / scale,
        method="DOP853",
        t_eval=times,
        events=equilibrium_event,
        dense_output=True,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=min(500.0, integration_horizon / 20.0),
    )
    if not solved.success or solved.y.shape[1] != len(times):
        raise RuntimeError(f"OXYTIB pCO2 trajectory failed: {solved.message}")

    states = tuple(
        GlobalO2Reservoir(
            *map(float, solved.y[:, index] * scale),
            source=f"OXYTIB pCO2 trajectory state at {time:g} years",
        )
        for index, time in enumerate(times)
    )
    pco2_values = tuple(pco2_at(float(time)) for time in times)
    final_event_value = equilibrium_event(
        integration_horizon, solved.sol(integration_horizon)
    )
    events = solved.t_events[0]
    if final_event_value > 0.0 or not events.size:
        equilibrium_time = None
    else:
        equilibrium_time = float(events[-1])

    return UpdatedPCO2TrajectoryResult(
        request=request,
        time_years=tuple(map(float, times)),
        pco2_ppm=pco2_values,
        states=states,
        initial_steady_state={
            "delta18_prime_permil": initial_state.delta18_prime_permil,
            "cap_delta17_prime_permil": initial_state.cap_delta17_prime_permil,
        },
        final_steady_state={
            "delta18_prime_permil": final_state.delta18_prime_permil,
            "cap_delta17_prime_permil": final_state.cap_delta17_prime_permil,
        },
        model_data_id=forcing_template.model_data_id,
        transfer_convention=forcing_template.transfer_convention,
        solver={
            "method": "scipy.solve_ivp DOP853",
            "relative_tolerance": 1.0e-10,
            "absolute_tolerance_scaled_inventory": 1.0e-12,
            "maximum_step_years": min(500.0, integration_horizon / 20.0),
            "display_duration_years": request.duration_years,
            "trajectory_duration_years": request.transition_duration_years,
            "equilibrium_search_horizon_years": integration_horizon,
            "function_evaluations": int(solved.nfev),
            "success": bool(solved.success),
            "message": solved.message,
        },
        equilibrium_time_years=equilibrium_time,
    )
