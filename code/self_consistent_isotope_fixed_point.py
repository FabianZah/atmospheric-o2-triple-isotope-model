"""Newton solver for a mechanistically evaluated isotope fixed point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class FixedPointStep:
    """One evaluated state in a Newton fixed-point solve."""

    iteration: int
    state: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    maximum_absolute_residual: float
    response_jacobian: np.ndarray | None
    linear_system_condition: float | None
    newton_step: np.ndarray | None


@dataclass(frozen=True)
class FixedPointResult:
    """Final state and numerical diagnostics for T(x) = x."""

    state: np.ndarray
    target: np.ndarray
    converged: bool
    maximum_absolute_residual: float
    target_evaluations: int
    history: tuple[FixedPointStep, ...]
    solver_method: str = "undamped_newton"


def solve_mechanistic_fixed_point(
    evaluate_target: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    *,
    finite_difference_step: float | np.ndarray = 0.5,
    tolerance: float = 1.0e-5,
    maximum_iterations: int = 6,
    maximum_condition: float = 1.0e8,
) -> FixedPointResult:
    """Solve ``evaluate_target(x) = x`` without physical damping.

    The response Jacobian is calculated from fresh mechanistic evaluations.
    Newton's method is applied to ``T(x) - x``.  No relaxation coefficient is
    introduced, so the converged state is independent of the unstable or
    oscillatory behavior of ordinary fixed-point iteration.
    """

    state = np.asarray(initial_state, dtype=float).copy()
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("initial fixed-point state must be a finite vector")
    steps = np.broadcast_to(
        np.asarray(finite_difference_step, dtype=float), state.shape
    ).copy()
    if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
        raise ValueError("finite-difference steps must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("fixed-point tolerance must be finite and positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum fixed-point iterations must be positive")
    if not np.isfinite(maximum_condition) or maximum_condition <= 1.0:
        raise ValueError("maximum condition must be finite and greater than one")

    evaluations = 0
    history: list[FixedPointStep] = []

    def evaluate(value: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        target = np.asarray(evaluate_target(np.asarray(value, dtype=float)), dtype=float)
        evaluations += 1
        if target.shape != state.shape or not np.all(np.isfinite(target)):
            raise ValueError("fixed-point target must be finite and state aligned")
        return target

    for iteration in range(1, maximum_iterations + 1):
        target = evaluate(state)
        residual = target - state
        maximum_residual = float(np.max(np.abs(residual)))
        if maximum_residual <= tolerance:
            history.append(
                FixedPointStep(
                    iteration=iteration,
                    state=state.copy(),
                    target=target.copy(),
                    residual=residual.copy(),
                    maximum_absolute_residual=maximum_residual,
                    response_jacobian=None,
                    linear_system_condition=None,
                    newton_step=None,
                )
            )
            return FixedPointResult(
                state=state.copy(),
                target=target.copy(),
                converged=True,
                maximum_absolute_residual=maximum_residual,
                target_evaluations=evaluations,
                history=tuple(history),
            )

        jacobian = np.empty((state.size, state.size), dtype=float)
        for column in range(state.size):
            perturbed = state.copy()
            perturbed[column] += steps[column]
            jacobian[:, column] = (evaluate(perturbed) - target) / steps[column]
        linear_system = np.eye(state.size) - jacobian
        condition = float(np.linalg.cond(linear_system))
        if not np.isfinite(condition) or condition > maximum_condition:
            raise np.linalg.LinAlgError(
                "mechanistic fixed-point Jacobian is ill-conditioned: "
                f"condition={condition:.6g}"
            )
        newton_step = np.linalg.solve(linear_system, residual)
        if not np.all(np.isfinite(newton_step)):
            raise FloatingPointError("fixed-point Newton step is not finite")
        history.append(
            FixedPointStep(
                iteration=iteration,
                state=state.copy(),
                target=target.copy(),
                residual=residual.copy(),
                maximum_absolute_residual=maximum_residual,
                response_jacobian=jacobian.copy(),
                linear_system_condition=condition,
                newton_step=newton_step.copy(),
            )
        )
        state = state + newton_step
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("fixed-point Newton update is not finite")

    target = evaluate(state)
    residual = target - state
    maximum_residual = float(np.max(np.abs(residual)))
    history.append(
        FixedPointStep(
            iteration=maximum_iterations + 1,
            state=state.copy(),
            target=target.copy(),
            residual=residual.copy(),
            maximum_absolute_residual=maximum_residual,
            response_jacobian=None,
            linear_system_condition=None,
            newton_step=None,
        )
    )
    return FixedPointResult(
        state=state.copy(),
        target=target.copy(),
        converged=maximum_residual <= tolerance,
        maximum_absolute_residual=maximum_residual,
        target_evaluations=evaluations,
        history=tuple(history),
    )


def solve_mechanistic_fixed_point_with_hybr_fallback(
    evaluate_target: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    *,
    finite_difference_step: float | np.ndarray = 0.5,
    tolerance: float = 1.0e-5,
    maximum_iterations: int = 6,
    maximum_condition: float = 1.0e8,
) -> FixedPointResult:
    """Use undamped Newton first, then MINPACK HYBR only if needed.

    The fallback solves the same residual ``T(x)-x`` and is accepted only after
    a fresh mechanistic target evaluation satisfies the requested tolerance.
    It changes no physical equation, relaxation coefficient, or target state.
    """

    primary = solve_mechanistic_fixed_point(
        evaluate_target,
        initial_state,
        finite_difference_step=finite_difference_step,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        maximum_condition=maximum_condition,
    )
    if primary.converged:
        return primary

    fallback_evaluations = 0

    def residual(state: np.ndarray) -> np.ndarray:
        nonlocal fallback_evaluations
        target = np.asarray(evaluate_target(np.asarray(state, dtype=float)), dtype=float)
        fallback_evaluations += 1
        if target.shape != primary.state.shape or not np.all(np.isfinite(target)):
            raise ValueError("fallback fixed-point target must be finite and state aligned")
        return target - state

    fallback = root(
        residual,
        primary.state,
        method="hybr",
        options={"xtol": min(tolerance * 0.01, 1.0e-10), "maxfev": 400},
    )
    state = np.asarray(fallback.x, dtype=float)
    target = np.asarray(evaluate_target(state), dtype=float)
    fallback_evaluations += 1
    final_residual = target - state
    maximum_residual = float(np.max(np.abs(final_residual)))
    converged = bool(
        fallback.success
        and np.all(np.isfinite(state))
        and np.all(np.isfinite(target))
        and maximum_residual <= tolerance
    )
    history = (
        *primary.history,
        FixedPointStep(
            iteration=len(primary.history) + 1,
            state=state.copy(),
            target=target.copy(),
            residual=final_residual.copy(),
            maximum_absolute_residual=maximum_residual,
            response_jacobian=None,
            linear_system_condition=None,
            newton_step=None,
        ),
    )
    return FixedPointResult(
        state=state,
        target=target,
        converged=converged,
        maximum_absolute_residual=maximum_residual,
        target_evaluations=primary.target_evaluations + fallback_evaluations,
        history=history,
        solver_method="undamped_newton_then_scipy_hybr",
    )
