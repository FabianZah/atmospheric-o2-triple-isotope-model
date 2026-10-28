"""Updated-model response to photosynthesis changes at fixed respiration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult, root

from global_o2_isotope_reservoir import GlobalO2Reservoir, IsotopologueTendency, sum_tendencies
from shared_carbon_oxygen_budget import (
    DEFAULT_CARBON_SINK_MULTIPLIER, SHARED_BUDGET_ID, SharedCarbonBudget,
)
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
    carbon_sink_multiplier: float = DEFAULT_CARBON_SINK_MULTIPLIER


@dataclass(frozen=True)
class UpdatedPhotosynthesisTransientResult:
    request: UpdatedPhotosynthesisTransientInput
    time_years: tuple[float, ...]
    states: tuple[GlobalO2Reservoir, ...]
    pco2_ppm: tuple[float, ...]
    carbon_driver_po2_pal: tuple[float, ...]
    carbon_inventories_mol: tuple[tuple[float, float], ...]
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
                "CO2_trop_mol": carbon_inventory[0],
                "CO2_strat_mol": carbon_inventory[1],
                "O2_delta18_prime_permil": state.delta18_prime_permil,
                "O2_cap_delta17_prime_permil": state.cap_delta17_prime_permil,
                "model_data_id": self.model_data_id,
            }
            for time, state, pco2, carbon_po2, carbon_inventory in zip(
                self.time_years,
                self.states,
                self.pco2_ppm,
                self.carbon_driver_po2_pal,
                self.carbon_inventories_mol,
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
            request.carbon_sink_multiplier,
        )
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("photosynthesis fraction, transient times and carbon sink multiplier must be finite and positive")
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
    initial_biology = forcing.biological_budget(initial_photo)
    biology = initial_biology.scaled_photosynthesis(
        request.photosynthesis_fraction,
        source=(
            "updated photosynthesis perturbation at fixed pre-perturbation "
            "respiration coefficients"
        ),
    )

    carbon = SharedCarbonBudget(request.initial.p_co2_ppm,
        initial_biology.tendency(initial_reservoir).o16o16, request.carbon_sink_multiplier)
    scale = np.concatenate((initial_reservoir.isotopologue_moles, carbon.initial_inventory))
    po2_bounds = (forcing.surface.po2_nodes_pal[0], forcing.surface.po2_nodes_pal[-1])
    co2_bounds = (forcing.surface.pco2_nodes_ppm[0], forcing.surface.pco2_nodes_ppm[-1])

    def tendency(time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        reservoir = GlobalO2Reservoir(
            *map(float, (scaled_inventory * scale)[:3]),
            source="updated photosynthesis-transient state",
        )
        live_po2 = reservoir.o16o16 / GLOBAL_MAJOR_O2_MOLES_1PAL
        carbon_inventory = (scaled_inventory * scale)[3:]
        live_pco2 = carbon.pco2(carbon_inventory)
        native = forcing.surface.evaluate_prime_tendency_at(
            np.asarray(
                [reservoir.delta18_prime_permil, reservoir.cap_delta17_prime_permil]
            ),
            # Implicit solver probes can cross a boundary. Terminal events
            # below reject such physical trajectories; no out-of-domain result
            # is returned and the chemistry itself is never extrapolated.
            po2_pal=float(np.clip(live_po2, *po2_bounds)),
            pco2_ppm=float(np.clip(live_pco2, *co2_bounds)),
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        photo = IsotopologueTendency(
            *map(float, forcing.forcing_scale * native.values),
            source="updated molecular R7 forcing at live pO2 and pCO2",
        )
        biological = biology.tendency(reservoir)
        oxygen_tendency = sum_tendencies(
            (biological, photo),
            source="fixed-respiration biology plus live updated R7 forcing",
        ).values
        carbon_tendency = carbon.tendency(carbon_inventory, biological.o16o16)
        return np.concatenate((oxygen_tendency, carbon_tendency)) / scale

    def make_boundary(index, conversion, bound, lower):
        def event(_time, state):
            value = state[index]*scale[index]*conversion
            return value-bound if lower else bound-value
        event.direction = -1.
        event.terminal = True
        return event

    events = [make_boundary(0,1/GLOBAL_MAJOR_O2_MOLES_1PAL,b,lower)
              for b,lower in zip(po2_bounds,(True,False))]
    events += [make_boundary(3,carbon.initial_pco2_ppm/carbon.initial_inventory[0],b,lower)
               for b,lower in zip(co2_bounds,(True,False))]
    initial_rate = tendency(0., np.ones(5))
    for index, value, bounds in ((0, request.initial.p_o2_pal, po2_bounds),
                                 (3, request.initial.p_co2_ppm, co2_bounds)):
        for bound, lower in zip(bounds, (True, False)):
            outward = initial_rate[index] < -1e-10 if lower else initial_rate[index] > 1e-10
            if np.isclose(value, bound, rtol=1e-12, atol=0.) and outward:
                raise ValueError("Photosynthesis trajectory leaves the supported pO2/pCO2 domain at the initial boundary")
    caught = []
    if request.photosynthesis_fraction == 1.:
        if np.max(abs(tendency(0.,np.ones(5)))) > 1e-10:
            raise RuntimeError("initial shared carbon/O2 state is not stationary")
        def stationary(times):
            return np.ones(5) if np.ndim(times)==0 else np.ones((5,len(times)))
        solved = OptimizeResult(success=True,status=0,t=np.array([0.,horizon]),
            y=np.ones((5,2)),sol=stationary,nfev=1,message="stationary initial equilibrium")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            solved = solve_ivp(
                tendency,
                (0.0, horizon),
                np.ones(5),
                method="Radau",
                dense_output=True,
                rtol=1.0e-9,
                atol=1.0e-11,
                max_step=min(100.0, horizon / 100.0),
                events=events,
            )
    if getattr(solved,'status',0) == 1:
        raise ValueError("Photosynthesis trajectory leaves the supported pO2/pCO2 domain "
                         f"at {solved.t[-1]:.3g} years; reduce the step or change the initial state")
    if (not solved.success or solved.sol is None
            or not np.isclose(solved.t[-1], horizon, rtol=0., atol=1e-8)):
        raise RuntimeError(f"shared carbon/O2 integration did not complete the requested horizon: {solved.message}")
    internal_inventory = solved.y.T * scale
    if not np.all(np.isfinite(internal_inventory)) or np.any(internal_inventory <= 0.):
        raise RuntimeError("shared carbon/O2 integration returned nonpositive or nonfinite inventories")

    display_time = np.linspace(0.0, request.duration_years, request.sample_count)
    display_inventory = solved.sol(display_time).T * scale
    display_states = tuple(
        GlobalO2Reservoir(*map(float, inventory[:3]), source=f"state at {time:g} yr")
        for time, inventory in zip(display_time, display_inventory, strict=True)
    )
    search_time = np.linspace(0.0, horizon, max(2001, int(horizon / 50.0) + 1))
    search_inventory = solved.sol(search_time).T * scale
    if (np.any(display_inventory <= 0.) or np.any(search_inventory <= 0.)
            or not np.all(np.isfinite(display_inventory)) or not np.all(np.isfinite(search_inventory))):
        raise RuntimeError("shared carbon/O2 dense output returned nonpositive or nonfinite inventories")
    for inventories in (internal_inventory, display_inventory, search_inventory):
        for values, bounds in (
            (inventories[:, 0] / GLOBAL_MAJOR_O2_MOLES_1PAL, po2_bounds),
            (inventories[:, 3] * carbon.initial_pco2_ppm / carbon.initial_inventory[0], co2_bounds),
        ):
            if np.any(values < bounds[0]*(1-1e-10)) or np.any(values > bounds[1]*(1+1e-10)):
                raise ValueError("Photosynthesis trajectory leaves the supported pO2/pCO2 domain")
    search_states = tuple(
        GlobalO2Reservoir(*map(float, inventory[:3]), source=f"search state at {time:g} yr")
        for time, inventory in zip(search_time, search_inventory, strict=True)
    )
    equilibrium = root(lambda state: tendency(0., state), search_inventory[-1]/scale,
                       method="hybr", options={"xtol": 1e-10})
    equilibrium_residual = float(np.max(abs(tendency(0., equilibrium.x))))
    equilibrium_methods = ["hybr"]
    # Carbon and isotope relaxation rates differ substantially. HYBR can report
    # success before the isotope residual meets the independent acceptance gate.
    if (not equilibrium.success or not np.isfinite(equilibrium_residual)
            or equilibrium_residual > 1e-12):
        equilibrium = root(lambda state: tendency(0., state), search_inventory[-1]/scale,
                           method="lm", options={"ftol": 1e-12, "xtol": 1e-12,
                                                 "gtol": 1e-12, "maxiter": 1200})
        equilibrium_residual = float(np.max(abs(tendency(0., equilibrium.x))))
        equilibrium_methods.append("lm")
    equilibrium_inventory = equilibrium.x * scale
    if (not equilibrium.success or not np.isfinite(equilibrium_residual)
            or equilibrium_residual > 1e-12
            or not np.all(np.isfinite(equilibrium_inventory)) or np.any(equilibrium_inventory <= 0)):
        raise RuntimeError("shared carbon/O2 equilibrium did not converge")
    equilibrium_po2 = equilibrium_inventory[0] / GLOBAL_MAJOR_O2_MOLES_1PAL
    equilibrium_pco2 = carbon.pco2(equilibrium_inventory[3:])
    if (not po2_bounds[0]*(1-1e-10) <= equilibrium_po2 <= po2_bounds[1]*(1+1e-10)
            or not co2_bounds[0]*(1-1e-10) <= equilibrium_pco2 <= co2_bounds[1]*(1+1e-10)):
        raise ValueError("Photosynthesis equilibrium leaves the supported pO2/pCO2 domain")
    long_run_state = GlobalO2Reservoir(*map(float, equilibrium_inventory[:3]),
                                     source="solved shared carbon/O2 equilibrium")
    equilibrium_time = _equilibrium_time(search_time, search_states, long_run_state)

    return UpdatedPhotosynthesisTransientResult(
        request=request,
        time_years=tuple(map(float, display_time)),
        states=display_states,
        pco2_ppm=tuple(carbon.pco2(inventory[3:]) for inventory in display_inventory),
        # Backward-compatible output: carbon now uses this same O2 state.
        carbon_driver_po2_pal=tuple(state.o16o16/GLOBAL_MAJOR_O2_MOLES_1PAL for state in display_states),
        carbon_inventories_mol=tuple(tuple(map(float, inventory[3:])) for inventory in display_inventory),
        long_run_state=long_run_state,
        long_run_pco2_ppm=equilibrium_pco2,
        operational_equilibrium_time_years=equilibrium_time,
        model_data_id=forcing.model_data_id,
        carbon_driver_preset=SHARED_BUDGET_ID,
        solver={
            "carbon_budget": carbon.metadata(),
            "coupling": "single five-state integration; shared biological fluxes and O2 inventory",
            "initial_production_mol_per_year": float(initial_biology.photosynthetic_sources_mol_per_year[0]),
            "initial_effective_respiration_per_year": float(initial_biology.effective_respiration_rates_per_year[0]),
            "oxygen_method": "stationary equilibrium" if request.photosynthesis_fraction == 1. else "scipy.solve_ivp Radau",
            "oxygen_success": bool(solved.success),
            "oxygen_function_evaluations": int(solved.nfev),
            "carbon_method": "stationary equilibrium" if request.photosynthesis_fraction == 1. else "scipy.solve_ivp Radau (joint O2/carbon integration)",
            "carbon_success": bool(solved.success),
            "carbon_function_evaluations": int(solved.nfev),
            "carbon_internal_runtime_warning_events": len(caught),
            "carbon_internal_runtime_warnings": tuple(str(item.message) for item in caught),
            "carbon_minimum_unclipped_inventory_mol": float(np.min(internal_inventory[:,3:])),
            "carbon_negative_inventory_count_before_clipping": 0,
            "display_duration_years": request.duration_years,
            "equilibrium_search_horizon_years": horizon,
            "equilibrium_method": "joint five-state stationary root",
            "equilibrium_root_methods": equilibrium_methods,
            "equilibrium_max_scaled_residual_per_year": equilibrium_residual,
        },
    )
