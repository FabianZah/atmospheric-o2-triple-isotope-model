"""Artificial 90-day tropospheric-air tracer on conservative transport grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply, spsolve


E90_LIFETIME_YEARS = 90.0 / 365.0
E90_GLOBAL_MEAN_PPB = 100.0
E90_TROPOPAUSE_THRESHOLD_PPB = 90.0


@dataclass(frozen=True)
class E90SteadyResult:
    mixing_ratio_ppb: np.ndarray
    surface_source_ppb_per_year: float
    maximum_tendency_residual_ppb_per_year: float
    positive_state: bool


@dataclass(frozen=True)
class E90PeriodicResult:
    mixing_ratio_at_interval_midpoint_ppb: np.ndarray
    surface_source_ppb_per_year: float
    spinup_years: int
    annual_cycle_maximum_relative_closure: float
    positive_state: bool


def _surface_source(shape: tuple[int, int]) -> np.ndarray:
    source = np.zeros(shape, dtype=float)
    source[:, 0] = 1.0
    return source.ravel()


def _scaled_to_global_mean(
    field: np.ndarray,
    air_moles: np.ndarray,
    *,
    target_global_mean_ppb: float,
    interval_years: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    values = np.asarray(field, dtype=float)
    air = np.asarray(air_moles, dtype=float)
    if interval_years is None:
        raw_mean = float(np.sum(values * air) / np.sum(air))
    else:
        duration = np.asarray(interval_years, dtype=float)
        monthly_mean = np.sum(values * air[None, :, :], axis=(1, 2)) / np.sum(air)
        raw_mean = float(np.sum(duration * monthly_mean))
    if not np.isfinite(raw_mean) or raw_mean <= 0.0:
        raise ValueError("E90 raw global mean must be positive")
    scale = float(target_global_mean_ppb / raw_mean)
    return values * scale, scale


def solve_e90_steady(
    transport: object,
    *,
    lifetime_years: float = E90_LIFETIME_YEARS,
    target_global_mean_ppb: float = E90_GLOBAL_MEAN_PPB,
) -> E90SteadyResult:
    """Solve steady E90 with uniform surface emission and uniform decay."""

    if lifetime_years <= 0.0 or target_global_mean_ppb <= 0.0:
        raise ValueError("E90 lifetime and target mean must be positive")
    air = np.asarray(transport.air_moles, dtype=float)
    operator = np.asarray(
        transport.mixing_ratio_transport_matrix_per_year(), dtype=float
    )
    if operator.shape != (air.size, air.size):
        raise ValueError("E90 transport operator does not match the air grid")
    source = _surface_source(air.shape)
    system = csc_matrix(operator) - diags(
        np.full(air.size, 1.0 / lifetime_years), format="csc"
    )
    raw = np.asarray(spsolve(system, -source), dtype=float)
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("steady E90 solve did not produce a positive state")
    scaled, source_scale = _scaled_to_global_mean(
        raw.reshape(air.shape), air, target_global_mean_ppb=target_global_mean_ppb
    )
    tendency = (
        operator @ scaled.ravel()
        + source_scale * source
        - scaled.ravel() / lifetime_years
    )
    return E90SteadyResult(
        mixing_ratio_ppb=scaled,
        surface_source_ppb_per_year=source_scale,
        maximum_tendency_residual_ppb_per_year=float(np.max(np.abs(tendency))),
        positive_state=bool(np.all(scaled > 0.0)),
    )


def solve_e90_periodic(
    transports: tuple[object, ...],
    interval_years: np.ndarray,
    *,
    lifetime_years: float = E90_LIFETIME_YEARS,
    target_global_mean_ppb: float = E90_GLOBAL_MEAN_PPB,
    closure_relative_tolerance: float = 1.0e-10,
    maximum_spinup_years: int = 50,
) -> E90PeriodicResult:
    """Solve the exact repeating E90 cycle for periodic transport."""

    if not transports:
        raise ValueError("at least one E90 transport interval is required")
    duration = np.asarray(interval_years, dtype=float)
    if duration.shape != (len(transports),) or np.any(duration <= 0.0):
        raise ValueError("E90 durations must be positive and match transports")
    if not np.isclose(np.sum(duration), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("E90 interval durations must sum to one year")
    if lifetime_years <= 0.0 or target_global_mean_ppb <= 0.0:
        raise ValueError("E90 lifetime and target mean must be positive")
    if closure_relative_tolerance <= 0.0 or maximum_spinup_years <= 0:
        raise ValueError("E90 periodic solve controls must be positive")
    air = np.asarray(transports[0].air_moles, dtype=float)
    source = _surface_source(air.shape)
    augmented = []
    for transport in transports:
        current_air = np.asarray(transport.air_moles, dtype=float)
        if not np.array_equal(current_air, air):
            raise ValueError("periodic E90 transports must share an air grid")
        operator = np.asarray(
            transport.mixing_ratio_transport_matrix_per_year(), dtype=float
        )
        active = csc_matrix(operator) - diags(
            np.full(air.size, 1.0 / lifetime_years), format="csc"
        )
        augmented.append(
            bmat(
                [[active, source[:, None]], [None, csr_matrix((1, 1))]],
                format="csr",
            )
        )
    annual = solve_e90_steady(
        transports[0],
        lifetime_years=lifetime_years,
        target_global_mean_ppb=target_global_mean_ppb,
    )
    state = np.concatenate((annual.mixing_ratio_ppb.ravel(), np.asarray([1.0])))
    closure = float("inf")
    spinup_years = 0
    for year in range(1, maximum_spinup_years + 1):
        start = state[:-1].copy()
        for matrix, interval in zip(augmented, duration):
            state = np.asarray(
                expm_multiply(matrix * float(interval), state), dtype=float
            )
        closure = float(
            np.max(
                np.abs(state[:-1] - start)
                / np.maximum(np.abs(state[:-1]), np.finfo(float).tiny)
            )
        )
        spinup_years = year
        if closure <= closure_relative_tolerance:
            break
    else:
        raise RuntimeError(
            "periodic E90 did not converge within "
            f"{maximum_spinup_years} years; relative closure={closure:g}"
        )
    midpoint = np.empty((len(transports), *air.shape), dtype=float)
    for interval_index, (matrix, interval) in enumerate(zip(augmented, duration)):
        half = np.asarray(
            expm_multiply(matrix * (0.5 * float(interval)), state), dtype=float
        )
        midpoint[interval_index] = half[:-1].reshape(air.shape)
        state = np.asarray(
            expm_multiply(matrix * (0.5 * float(interval)), half), dtype=float
        )
    scaled, source_scale = _scaled_to_global_mean(
        midpoint,
        air,
        target_global_mean_ppb=target_global_mean_ppb,
        interval_years=duration,
    )
    return E90PeriodicResult(
        mixing_ratio_at_interval_midpoint_ppb=scaled,
        surface_source_ppb_per_year=source_scale,
        spinup_years=spinup_years,
        annual_cycle_maximum_relative_closure=closure,
        positive_state=bool(np.all(scaled > 0.0)),
    )
