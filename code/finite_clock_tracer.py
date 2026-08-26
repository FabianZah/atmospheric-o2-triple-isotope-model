"""Finite-time passive tracers on the latitude-pressure transport grid.

The steady mean-age equation in :mod:`passive_age_of_air` is the asymptotic
limit of a linearly increasing lower-boundary clock tracer. This module keeps
that finite-time experiment explicit and also supports arbitrary, spatially
uniform lower-boundary histories such as the observed tropical SF6 record.

All transport coefficients come from the supplied ``PassiveAgeTransport``.
No age target, relaxation factor, or chemistry is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import expm_multiply

from passive_age_of_air import PassiveAgeTransport


@dataclass(frozen=True)
class FiniteClockResult:
    """Finite-time age and clock-tracer fields."""

    transport: PassiveAgeTransport
    elapsed_years: np.ndarray
    mean_age_years: np.ndarray
    clock_tracer_years: np.ndarray

    def age_at(
        self,
        elapsed_years: float,
        altitude_km: float,
        latitude_degrees: float = 0.0,
    ) -> float:
        """Interpolate the finite-clock age in time, altitude, and latitude."""

        time = np.asarray(self.elapsed_years, dtype=float)
        if not time[0] <= elapsed_years <= time[-1]:
            raise ValueError("requested time lies outside the clock solution")
        altitude = np.asarray(self.transport.altitude_centers_km, dtype=float)
        latitude = self.transport.latitude_centers_degrees
        if not altitude[0] <= altitude_km <= altitude[-1]:
            raise ValueError("requested altitude lies outside the age grid")
        if not latitude[0] <= latitude_degrees <= latitude[-1]:
            raise ValueError("requested latitude lies outside the age grid")
        time_profiles = np.asarray(
            [
                [
                    np.interp(
                        altitude_km,
                        altitude,
                        self.mean_age_years[time_index, latitude_index, :],
                    )
                    for latitude_index in range(len(latitude))
                ]
                for time_index in range(len(time))
            ],
            dtype=float,
        )
        latitude_profile = np.asarray(
            [
                np.interp(latitude_degrees, latitude, profile)
                for profile in time_profiles
            ],
            dtype=float,
        )
        return float(np.interp(elapsed_years, time, latitude_profile))


@dataclass(frozen=True)
class BoundaryHistoryResult:
    """Passive tracer field forced by a prescribed lower-boundary history."""

    transport: PassiveAgeTransport
    time_years: np.ndarray
    boundary_value: np.ndarray
    tracer_mixing_ratio: np.ndarray
    solver_success: bool
    solver_message: str

    def value_at(
        self,
        time_years: float,
        altitude_km: float,
        latitude_degrees: float = 0.0,
    ) -> float:
        """Interpolate tracer mixing ratio in time, altitude, and latitude."""

        time = np.asarray(self.time_years, dtype=float)
        if not time[0] <= time_years <= time[-1]:
            raise ValueError("requested time lies outside the tracer solution")
        altitude = np.asarray(self.transport.altitude_centers_km, dtype=float)
        latitude = self.transport.latitude_centers_degrees
        time_profiles = np.asarray(
            [
                [
                    np.interp(
                        altitude_km,
                        altitude,
                        self.tracer_mixing_ratio[
                            time_index,
                            latitude_index,
                            :,
                        ],
                    )
                    for latitude_index in range(len(latitude))
                ]
                for time_index in range(len(time))
            ],
            dtype=float,
        )
        latitude_profile = np.asarray(
            [
                np.interp(latitude_degrees, latitude, profile)
                for profile in time_profiles
            ],
            dtype=float,
        )
        return float(np.interp(time_years, time, latitude_profile))


@dataclass(frozen=True)
class PeriodicClockResult:
    """Periodic finite-clock solution for a sequence of transport operators."""

    transports: tuple[PassiveAgeTransport, ...]
    interval_years: np.ndarray
    midpoint_mean_age_years: np.ndarray
    interval_end_mean_age_years: np.ndarray
    duration_weighted_annual_mean_age_years: np.ndarray
    spinup_years: int
    annual_cycle_closure_max_years: float


def _validated_times(times: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _partition_operator(
    transport: PassiveAgeTransport,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    operator = transport.mixing_ratio_transport_matrix_per_year()
    reset = np.asarray(transport.reset_mask, dtype=bool).ravel()
    active = ~reset
    active_operator = operator[np.ix_(active, active)]
    boundary_operator = operator[np.ix_(active, reset)]
    return operator, active, active_operator, boundary_operator


def solve_finite_clock_age(
    transport: PassiveAgeTransport,
    elapsed_years: np.ndarray,
) -> FiniteClockResult:
    """Propagate a finite linearly increasing clock exactly.

    The lower boundary is held at clock value ``t`` and zero age. For active
    cells, ``Gamma = t - clock`` obeys

        dGamma/dt = 1 + L_aa Gamma,

    where ``L_aa`` is the active-cell mixing-ratio transport operator. A sparse
    augmented matrix exponential evaluates this affine system without a time
    stepping tolerance.
    """

    time = _validated_times(elapsed_years, name="elapsed_years")
    if time[0] < 0.0:
        raise ValueError("elapsed_years cannot be negative")
    _, active, active_operator, _ = _partition_operator(transport)
    active_count = int(np.count_nonzero(active))
    augmented = bmat(
        [
            [csr_matrix(active_operator), np.ones((active_count, 1))],
            [None, csr_matrix((1, 1))],
        ],
        format="csr",
    )
    state = np.zeros(active_count + 1, dtype=float)
    state[-1] = 1.0
    previous_time = 0.0
    active_age_by_time = []
    for current_time in time:
        interval = float(current_time - previous_time)
        if interval > 0.0:
            state = np.asarray(
                expm_multiply(augmented * interval, state),
                dtype=float,
            )
        active_age_by_time.append(state[:-1].copy())
        previous_time = float(current_time)

    shape = transport.air_moles.shape
    age = np.zeros((len(time), active.size), dtype=float)
    for index, active_age in enumerate(active_age_by_time):
        age[index, active] = active_age
    if float(np.min(age)) < -1.0e-9:
        raise ValueError("finite clock produced negative age")
    age = np.maximum(age, 0.0).reshape((len(time), *shape))
    clock = time[:, None, None] - age
    reset = np.asarray(transport.reset_mask, dtype=bool)
    clock[:, reset] = np.broadcast_to(
        time[:, None],
        (len(time), int(np.count_nonzero(reset))),
    )
    return FiniteClockResult(
        transport=transport,
        elapsed_years=time,
        mean_age_years=age,
        clock_tracer_years=clock,
    )


def propagate_boundary_history(
    transport: PassiveAgeTransport,
    *,
    boundary_time_years: np.ndarray,
    boundary_values: np.ndarray,
    output_time_years: np.ndarray,
    initial_uniform_value: float | None = None,
    relative_tolerance: float = 2.0e-8,
    absolute_tolerance: float = 1.0e-10,
) -> BoundaryHistoryResult:
    """Propagate a spatially uniform lower-boundary tracer history.

    The transport is linear and chemistry-free. Boundary values are linearly
    interpolated in time and held in every reset cell. The initial active
    atmosphere is spatially uniform, using the first boundary value unless an
    explicit value is supplied.
    """

    boundary_time = _validated_times(
        boundary_time_years,
        name="boundary_time_years",
    )
    boundary = np.asarray(boundary_values, dtype=float)
    if boundary.shape != boundary_time.shape:
        raise ValueError("boundary_values must match boundary_time_years")
    if not np.all(np.isfinite(boundary)):
        raise ValueError("boundary_values must be finite")
    output_time = _validated_times(
        output_time_years,
        name="output_time_years",
    )
    if output_time[0] < boundary_time[0] or output_time[-1] > boundary_time[-1]:
        raise ValueError("output times must lie inside the boundary history")
    _, active, active_operator, boundary_operator = _partition_operator(transport)
    active_sparse = csr_matrix(active_operator)
    boundary_coupling = boundary_operator @ np.ones(
        boundary_operator.shape[1],
        dtype=float,
    )
    initial = (
        float(boundary[0])
        if initial_uniform_value is None
        else float(initial_uniform_value)
    )
    if not np.isfinite(initial):
        raise ValueError("initial_uniform_value must be finite")
    active_initial = np.full(
        int(np.count_nonzero(active)),
        initial,
        dtype=float,
    )

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        boundary_value = float(np.interp(time, boundary_time, boundary))
        return active_sparse @ state + boundary_coupling * boundary_value

    solution = solve_ivp(
        rhs,
        (float(boundary_time[0]), float(output_time[-1])),
        active_initial,
        method="BDF",
        t_eval=output_time,
        jac=active_sparse,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    if not solution.success:
        raise RuntimeError(f"boundary-history integration failed: {solution.message}")
    shape = transport.air_moles.shape
    field = np.empty((len(output_time), active.size), dtype=float)
    reset = ~active
    output_boundary = np.interp(output_time, boundary_time, boundary)
    field[:, active] = solution.y.T
    field[:, reset] = output_boundary[:, None]
    return BoundaryHistoryResult(
        transport=transport,
        time_years=output_time,
        boundary_value=output_boundary,
        tracer_mixing_ratio=field.reshape((len(output_time), *shape)),
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
    )


def solve_periodic_clock_age(
    transports: tuple[PassiveAgeTransport, ...],
    interval_years: np.ndarray,
    *,
    closure_tolerance_years: float = 1.0e-10,
    maximum_spinup_years: int = 200,
) -> PeriodicClockResult:
    """Solve a repeating seasonal clock with exact interval propagation.

    The interval operators and reset masks must share one grid. Repeated annual
    propagation reaches the unique periodic attractor; no relaxation factor is
    used. Midpoint fields are then evaluated for a duration-weighted annual
    climatology.
    """

    if not transports:
        raise ValueError("at least one periodic transport is required")
    duration = np.asarray(interval_years, dtype=float)
    if duration.shape != (len(transports),):
        raise ValueError("interval_years must match the transport count")
    if not np.all(np.isfinite(duration)) or np.any(duration <= 0.0):
        raise ValueError("periodic intervals must be finite and positive")
    reference = transports[0]
    reference_reset = np.asarray(reference.reset_mask, dtype=bool)
    reference_air = np.asarray(reference.air_moles, dtype=float)
    active = ~reference_reset.ravel()
    active_count = int(np.count_nonzero(active))
    augmented_operators = []
    for transport in transports:
        if not np.array_equal(transport.reset_mask, reference_reset):
            raise ValueError("periodic transports must share one reset mask")
        if not np.allclose(
            transport.air_moles,
            reference_air,
            rtol=1.0e-13,
            atol=0.0,
        ):
            raise ValueError("periodic transports must share one air inventory")
        operator = transport.mixing_ratio_transport_matrix_per_year()
        active_operator = operator[np.ix_(active, active)]
        augmented_operators.append(
            bmat(
                [
                    [
                        csr_matrix(active_operator),
                        np.ones((active_count, 1)),
                    ],
                    [None, csr_matrix((1, 1))],
                ],
                format="csr",
            )
        )

    state = np.zeros(active_count + 1, dtype=float)
    state[-1] = 1.0
    closure = float("inf")
    spinup_years = 0
    for year in range(1, maximum_spinup_years + 1):
        start = state[:-1].copy()
        for augmented, interval in zip(augmented_operators, duration):
            state = np.asarray(
                expm_multiply(augmented * float(interval), state),
                dtype=float,
            )
        closure = float(np.max(np.abs(state[:-1] - start)))
        spinup_years = year
        if closure <= closure_tolerance_years:
            break
    else:
        raise RuntimeError(
            "periodic clock did not converge within "
            f"{maximum_spinup_years} years; closure={closure:g}"
        )

    shape = reference.air_moles.shape
    midpoint = np.zeros((len(transports), active.size), dtype=float)
    interval_end = np.zeros_like(midpoint)
    cycle_start = state.copy()
    for index, (augmented, interval) in enumerate(
        zip(augmented_operators, duration)
    ):
        midpoint_state = np.asarray(
            expm_multiply(augmented * (0.5 * float(interval)), cycle_start),
            dtype=float,
        )
        end_state = np.asarray(
            expm_multiply(augmented * float(interval), cycle_start),
            dtype=float,
        )
        midpoint[index, active] = midpoint_state[:-1]
        interval_end[index, active] = end_state[:-1]
        cycle_start = end_state
    normalized_duration = duration / np.sum(duration)
    annual_mean = np.tensordot(normalized_duration, midpoint, axes=(0, 0))
    return PeriodicClockResult(
        transports=tuple(transports),
        interval_years=duration,
        midpoint_mean_age_years=midpoint.reshape((len(transports), *shape)),
        interval_end_mean_age_years=interval_end.reshape(
            (len(transports), *shape)
        ),
        duration_weighted_annual_mean_age_years=annual_mean.reshape(shape),
        spinup_years=spinup_years,
        annual_cycle_closure_max_years=closure,
    )
