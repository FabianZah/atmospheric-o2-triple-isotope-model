"""Source-faithful reduced Cao and Bao (2013) atmospheric O2 model.

This module implements supporting-information Eqs. S1c-S3c and the isotope
forcing functions printed in the main-paper Table 1.  It is an independent
benchmark implementation, not part of the updated production forward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


PRESENT_PO2_BAR = 0.2095
PRESENT_PCO2_PPM = 375.0
PRESENT_FOP_MOL_O2_PER_YEAR = 1.09e16
PRESENT_GAMMA_PER_YEAR = 0.1321
PRESENT_THETA = 0.017

# A one-bar atmosphere contains approximately this many moles of gas.  This
# physical pressure-inventory conversion, together with the printed fluxes,
# reproduces Fig. S6.  It is intentionally separate from the paper's
# inconsistent "1 PV = 1,244 y" residence-time caption.
ATMOSPHERIC_MOLES_PER_BAR = 1.765e20


def delta18_co2_minus_o2_permil(rho: float | np.ndarray) -> float | np.ndarray:
    """Return Cao-Bao Table 1 delta18O(CO2-O2) at pO2/pCO2 = rho."""

    rho_array = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(rho_array)) or np.any(rho_array < 0.0):
        raise ValueError("rho must be finite and non-negative")
    result = (64.0 + 146.0 * (rho_array / 1.23)) / (1.0 + rho_array / 1.23)
    return float(result) if result.ndim == 0 else result


def phi_permil(rho: float | np.ndarray) -> float | np.ndarray:
    """Return Phi(rho) in Cao-Bao's native linear Delta17O convention."""

    result = 0.519 * np.asarray(delta18_co2_minus_o2_permil(rho)) - 7.1738
    return float(result) if result.ndim == 0 else result


def residence_time_years(
    p_o2_bar: float,
    fop_mol_o2_per_year: float = PRESENT_FOP_MOL_O2_PER_YEAR,
) -> float:
    """Return tau = atmospheric O2 inventory / Fop."""

    if not np.isfinite(p_o2_bar) or p_o2_bar < 0.0:
        raise ValueError("pO2 must be finite and non-negative")
    if not np.isfinite(fop_mol_o2_per_year) or fop_mol_o2_per_year <= 0.0:
        raise ValueError("Fop must be finite and positive")
    return p_o2_bar * ATMOSPHERIC_MOLES_PER_BAR / fop_mol_o2_per_year


def steady_delta17_linear_052_permil(
    *,
    p_o2_bar: float,
    p_co2_bar: float,
    fop_mol_o2_per_year: float = PRESENT_FOP_MOL_O2_PER_YEAR,
    gamma_per_year: float = PRESENT_GAMMA_PER_YEAR,
    theta: float = PRESENT_THETA,
) -> float:
    """Evaluate the steady solution of supporting-information Eq. S2c."""

    if p_o2_bar < 0.0 or p_co2_bar <= 0.0:
        raise ValueError("steady pO2 must be non-negative and pCO2 positive")
    tau = residence_time_years(p_o2_bar, fop_mol_o2_per_year)
    rho = p_o2_bar / p_co2_bar
    gamma_theta_tau = gamma_per_year * theta * tau
    return float(
        -gamma_theta_tau
        * phi_permil(rho)
        / (1.0 + rho + gamma_theta_tau)
    )


@dataclass(frozen=True)
class CaoBaoRateExperiment:
    """Constant-rate experiment defined by SI Eqs. S1c-S3c and Fig. S6."""

    initial_p_o2_bar: float = 0.0
    initial_p_co2_bar: float = 0.1
    initial_delta17_linear_052_permil: float = 0.0
    o2_rise_mol_per_year: float = 1.0e13
    co2_drawdown_mol_per_year: float = 1.7e13
    fop_mol_o2_per_year: float = PRESENT_FOP_MOL_O2_PER_YEAR
    gamma_per_year: float = PRESENT_GAMMA_PER_YEAR
    theta: float = PRESENT_THETA
    duration_years: float = 1.0e6
    sample_count: int = 5001


@dataclass(frozen=True)
class CaoBaoRateResult:
    request: CaoBaoRateExperiment
    time_years: np.ndarray
    p_o2_bar: np.ndarray
    p_co2_bar: np.ndarray
    delta17_linear_052_permil: np.ndarray
    minimum_delta17_linear_052_permil: float
    minimum_time_years: float


def run_rate_experiment(request: CaoBaoRateExperiment) -> CaoBaoRateResult:
    """Integrate the reduced isotope ODE along linearly evolving O2 and CO2."""

    scalars = (
        request.initial_p_o2_bar,
        request.initial_p_co2_bar,
        request.initial_delta17_linear_052_permil,
        request.o2_rise_mol_per_year,
        request.co2_drawdown_mol_per_year,
        request.fop_mol_o2_per_year,
        request.gamma_per_year,
        request.theta,
        request.duration_years,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("all experiment values must be finite")
    if request.initial_p_o2_bar < 0.0 or request.initial_p_co2_bar <= 0.0:
        raise ValueError("initial pO2 must be non-negative and pCO2 positive")
    if request.o2_rise_mol_per_year < 0.0:
        raise ValueError("O2 rise rate must be non-negative")
    if request.co2_drawdown_mol_per_year < 0.0:
        raise ValueError("CO2 drawdown rate must be non-negative")
    if request.fop_mol_o2_per_year <= 0.0 or request.duration_years <= 0.0:
        raise ValueError("Fop and duration must be positive")
    if request.sample_count < 2:
        raise ValueError("sample_count must be at least two")

    def pressures(time_years: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(time_years, dtype=float)
        p_o2 = request.initial_p_o2_bar + (
            request.o2_rise_mol_per_year * time / ATMOSPHERIC_MOLES_PER_BAR
        )
        p_co2 = request.initial_p_co2_bar - (
            request.co2_drawdown_mol_per_year * time / ATMOSPHERIC_MOLES_PER_BAR
        )
        return p_o2, np.maximum(p_co2, 1.0e-12)

    def tendency(time_years: float, state: np.ndarray) -> np.ndarray:
        p_o2, p_co2 = pressures(time_years)
        rho = float(p_o2 / p_co2)
        tau = residence_time_years(float(p_o2), request.fop_mol_o2_per_year)
        gamma_theta = request.gamma_per_year * request.theta
        negative_forcing = gamma_theta * phi_permil(rho) / (1.0 + rho)
        if tau <= 1.0e-14:
            # At pO2=0 and Delta17O=0, Eq. S2c has this finite right limit.
            return np.asarray([-negative_forcing])
        relaxation = (
            1.0 + rho + gamma_theta * tau
        ) / ((1.0 + rho) * tau)
        return np.asarray([-relaxation * state[0] - negative_forcing])

    times = np.linspace(0.0, request.duration_years, request.sample_count)
    solved = solve_ivp(
        tendency,
        (0.0, request.duration_years),
        [request.initial_delta17_linear_052_permil],
        t_eval=times,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=min(500.0, request.duration_years / 100.0),
    )
    if not solved.success or solved.y.shape[1] != request.sample_count:
        raise RuntimeError(f"Cao-Bao reduced integration failed: {solved.message}")
    p_o2, p_co2 = pressures(times)
    delta17 = solved.y[0]
    minimum_index = int(np.argmin(delta17))
    return CaoBaoRateResult(
        request=request,
        time_years=times,
        p_o2_bar=p_o2,
        p_co2_bar=p_co2,
        delta17_linear_052_permil=delta17,
        minimum_delta17_linear_052_permil=float(delta17[minimum_index]),
        minimum_time_years=float(times[minimum_index]),
    )
