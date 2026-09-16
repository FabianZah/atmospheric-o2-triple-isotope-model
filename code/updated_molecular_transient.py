"""Slow global-O2 response to updated-model boundary-condition steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from global_o2_isotope_reservoir import (
    GlobalO2Reservoir,
    sum_tendencies,
)
from updated_molecular_forward_model import (
    UpdatedForwardInput,
    build_updated_central_forcing,
    run_updated_central_state,
)
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL = 0.001


@dataclass(frozen=True)
class UpdatedTransientInput:
    initial: UpdatedForwardInput
    final: UpdatedForwardInput
    duration_years: float = 12000.0
    sample_count: int = 161
    equilibrium_search_max_years: float = 100000.0


@dataclass(frozen=True)
class UpdatedTransientResult:
    request: UpdatedTransientInput
    time_years: tuple[float, ...]
    states: tuple[GlobalO2Reservoir, ...]
    initial_steady_state: dict[str, float]
    final_steady_state: dict[str, float]
    model_data_id: str
    transfer_convention: str
    solver: dict[str, Any]
    instantaneous_po2_inventory_step: bool
    equilibrium_time_years: float | None

    def operational_equilibrium_time_years(
        self,
    ) -> float | None:
        """Return the independently searched operational-equilibrium time."""

        return self.equilibrium_time_years

    def rows(self) -> list[dict[str, float | str | bool]]:
        target_d18 = self.final_steady_state["delta18_prime_permil"]
        target_d17 = self.final_steady_state["cap_delta17_prime_permil"]
        return [
            {
                "time_yr": time,
                "pO2_pal": state.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL,
                "O2_delta18_prime_permil": state.delta18_prime_permil,
                "O2_cap_delta17_prime_permil": state.cap_delta17_prime_permil,
                "target_delta18_prime_permil": target_d18,
                "target_cap_delta17_prime_permil": target_d17,
                "model_data_id": self.model_data_id,
                "instantaneous_pO2_inventory_step": (
                    self.instantaneous_po2_inventory_step
                ),
            }
            for time, state in zip(self.time_years, self.states, strict=True)
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
            "criterion": "both isotope coordinates enter and remain within tolerance",
            "tolerance_permil": OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL,
            "time_years": self.operational_equilibrium_time_years(),
        }
        return data


def run_updated_transient(request: UpdatedTransientInput) -> UpdatedTransientResult:
    """Apply a boundary step at year zero and integrate global O2 isotopologues."""

    if not np.isfinite(request.duration_years) or request.duration_years <= 0.0:
        raise ValueError("transient duration must be finite and positive")
    if (
        not np.isfinite(request.equilibrium_search_max_years)
        or request.equilibrium_search_max_years <= 0.0
    ):
        raise ValueError("equilibrium search horizon must be finite and positive")
    if request.sample_count < 2:
        raise ValueError("transient requires at least two output samples")

    initial_state = run_updated_central_state(request.initial)
    final_state = run_updated_central_state(request.final)
    po2_changed = not np.isclose(
        request.initial.p_o2_pal,
        request.final.p_o2_pal,
        rtol=0.0,
        atol=1.0e-14,
    )
    initial_major = (
        request.final.p_o2_pal if po2_changed else request.initial.p_o2_pal
    ) * GLOBAL_MAJOR_O2_MOLES_1PAL
    initial_reservoir = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=initial_major,
        delta18_prime_permil=initial_state.delta18_prime_permil,
        cap_delta17_prime_permil=initial_state.cap_delta17_prime_permil,
        source=(
            "updated-model initial steady isotope composition after the "
            "year-zero pO2 inventory step"
            if po2_changed
            else "updated-model initial steady reservoir"
        ),
    )
    final_reservoir = GlobalO2Reservoir.from_prime_composition(
        major_o2_moles=request.final.p_o2_pal * GLOBAL_MAJOR_O2_MOLES_1PAL,
        delta18_prime_permil=final_state.delta18_prime_permil,
        cap_delta17_prime_permil=final_state.cap_delta17_prime_permil,
        source="updated-model final steady reservoir",
    )

    forcing = build_updated_central_forcing(request.final)
    final_photo = forcing.photochemical_tendency(
        np.asarray(
            [
                final_state.delta18_prime_permil,
                final_state.cap_delta17_prime_permil,
            ]
        )
    )
    biology = forcing.biological_budget(final_photo)
    scale = final_reservoir.isotopologue_moles

    def tendency(_time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        inventory = scaled_inventory * scale
        reservoir = GlobalO2Reservoir(
            *map(float, inventory),
            source="updated-model transient integration state",
        )
        photo = forcing.photochemical_tendency(
            np.asarray(
                [
                    reservoir.delta18_prime_permil,
                    reservoir.cap_delta17_prime_permil,
                ]
            )
        )
        total = sum_tendencies(
            (biology.tendency(reservoir), photo),
            source="updated partitioned biology plus state-dependent R7 forcing",
        )
        return total.values / scale

    def equilibrium_event(_time: float, scaled_inventory: np.ndarray) -> float:
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * scale),
            source="updated-model equilibrium event state",
        )
        return max(
            abs(
                reservoir.cap_delta17_prime_permil
                - final_state.cap_delta17_prime_permil
            ),
            abs(
                reservoir.delta18_prime_permil
                - final_state.delta18_prime_permil
            ),
        ) - OPERATIONAL_EQUILIBRIUM_TOLERANCE_PERMIL

    equilibrium_event.direction = 0.0
    equilibrium_event.terminal = False

    times = np.linspace(0.0, request.duration_years, request.sample_count)
    integration_horizon = max(
        request.duration_years, request.equilibrium_search_max_years
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
        raise RuntimeError(f"updated molecular transient failed: {solved.message}")
    states = tuple(
        GlobalO2Reservoir(
            *map(float, solved.y[:, index] * scale),
            source=f"updated-model transient state at {time:g} years",
        )
        for index, time in enumerate(times)
    )
    initial_distance = equilibrium_event(
        0.0, initial_reservoir.isotopologue_moles / scale
    )
    final_distance = equilibrium_event(
        integration_horizon, solved.sol(integration_horizon)
    )
    if initial_distance <= 0.0:
        equilibrium_time = 0.0
    elif final_distance > 0.0 or not solved.t_events[0].size:
        equilibrium_time = None
    else:
        # The last inward crossing is the point after which the stable solution
        # remains inside the operational-equilibrium band.
        equilibrium_time = float(solved.t_events[0][-1])
    return UpdatedTransientResult(
        request=request,
        time_years=tuple(map(float, times)),
        states=states,
        initial_steady_state={
            "delta18_prime_permil": initial_state.delta18_prime_permil,
            "cap_delta17_prime_permil": initial_state.cap_delta17_prime_permil,
        },
        final_steady_state={
            "delta18_prime_permil": final_state.delta18_prime_permil,
            "cap_delta17_prime_permil": final_state.cap_delta17_prime_permil,
        },
        model_data_id=forcing.model_data_id,
        transfer_convention=forcing.transfer_convention,
        solver={
            "method": "scipy.solve_ivp DOP853",
            "relative_tolerance": 1.0e-10,
            "absolute_tolerance_scaled_inventory": 1.0e-12,
            "maximum_step_years": min(500.0, integration_horizon / 20.0),
            "display_duration_years": request.duration_years,
            "equilibrium_search_horizon_years": integration_horizon,
            "function_evaluations": int(solved.nfev),
            "success": bool(solved.success),
            "message": solved.message,
        },
        instantaneous_po2_inventory_step=po2_changed,
        equilibrium_time_years=equilibrium_time,
    )
