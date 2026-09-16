"""Source-derived reduced Young et al. (2014) Fig. 9 O2 response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from global_o2_isotope_reservoir import (
    GlobalO2Reservoir,
    IsotopologueTendency,
    YoungBiologicalO2Budget,
    frozen_photochemical_steady_state,
    sum_tendencies,
)
from young_bulk_fig10 import (
    YOUNG_FIG10_INITIAL_PCO2_PPM,
    young_printed_biological_budget,
    young_table3_r7_tendencies,
)


@dataclass(frozen=True)
class YoungBulkFig9Trajectory:
    time_years: np.ndarray
    pco2_ppm: np.ndarray
    states: tuple[GlobalO2Reservoir, ...]
    initial_equilibrium: GlobalO2Reservoir

    @property
    def cap_delta17_permil(self) -> np.ndarray:
        return np.asarray([state.cap_delta17_prime_permil for state in self.states])

    @property
    def delta18_prime_permil(self) -> np.ndarray:
        return np.asarray([state.delta18_prime_permil for state in self.states])

    @property
    def major_o2_ratio(self) -> np.ndarray:
        initial = self.initial_equilibrium.o16o16
        return np.asarray([state.o16o16 / initial for state in self.states])


def half_photosynthesis_fixed_respiration(
    biology: YoungBiologicalO2Budget,
) -> YoungBiologicalO2Budget:
    return YoungBiologicalO2Budget(
        photosynthetic_o16o16_mol_per_year=(
            0.5 * biology.photosynthetic_o16o16_mol_per_year
        ),
        respiration_rate_per_year=biology.respiration_rate_per_year,
        tropospheric_fraction=biology.tropospheric_fraction,
        alpha_respiration_18=biology.alpha_respiration_18,
        beta_respiration_17=biology.beta_respiration_17,
        source_alpha_18=biology.source_alpha_18,
        source_beta_17=biology.source_beta_17,
        source=(
            "Young et al. (2014) Fig. 9: instantaneous half photosynthesis at "
            "fixed respiration rate"
        ),
    )


def integrate_young_bulk_fig9(
    time_years: np.ndarray,
    pco2_ppm: np.ndarray,
    *,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
) -> YoungBulkFig9Trajectory:
    times = np.asarray(time_years, dtype=float)
    co2 = np.asarray(pco2_ppm, dtype=float)
    if times.ndim != 1 or times.size < 2 or co2.shape != times.shape:
        raise ValueError("Fig. 9 time and pCO2 arrays must have the same 1-D shape")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(co2)) or np.any(co2 <= 0.0):
        raise ValueError("Fig. 9 time and pCO2 arrays must be finite and positive")
    if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Fig. 9 times must start at zero and increase strictly")

    biology = young_printed_biological_budget()
    perturbed_biology = half_photosynthesis_fixed_respiration(biology)
    r7_294, _carbon = young_table3_r7_tendencies()
    initial = frozen_photochemical_steady_state(
        biology, r7_294, source="reduced Young bulk pre-Fig. 9 equilibrium"
    )
    inventory_scale = initial.isotopologue_moles

    def rhs(time: float, scaled_inventory: np.ndarray) -> np.ndarray:
        reservoir = GlobalO2Reservoir(
            *map(float, scaled_inventory * inventory_scale),
            source="Young bulk Fig. 9 integration state",
        )
        live_pco2 = float(np.interp(time, times, co2))
        photochemical = IsotopologueTendency(
            *map(
                float,
                r7_294.values * live_pco2 / YOUNG_FIG10_INITIAL_PCO2_PPM,
            ),
            source=(
                "Young Table 3 homogeneous-box R7 tendency scaled by the live "
                "Fig. 9 tropospheric CO2 trajectory"
            ),
        )
        return sum_tendencies(
            (perturbed_biology.tendency(reservoir), photochemical),
            source="Young half-photosynthesis biology plus live-CO2 bulk R7",
        ).values / inventory_scale

    solved = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.ones(3),
        t_eval=times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=20.0,
    )
    if not solved.success or solved.y.shape[1] != times.size:
        raise RuntimeError(solved.message)
    states = tuple(
        GlobalO2Reservoir(
            *map(float, inventory), source=f"Young bulk Fig. 9 at {time:g} yr"
        )
        for time, inventory in zip(
            times, solved.y.T * inventory_scale, strict=True
        )
    )
    return YoungBulkFig9Trajectory(
        time_years=times.copy(),
        pco2_ppm=co2.copy(),
        states=states,
        initial_equilibrium=initial,
    )
