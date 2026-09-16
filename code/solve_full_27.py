"""Solve all 27 species in the Young model reconstruction.

This is intended for diagnostic reservoir-closure variants. It uses the same
Newton/log approach as the atmospheric solver but includes the formal
biosphere/hydrosphere and geosphere species from Young Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reactions import derivative
from solver_utils import throughput_scales
from table3_state import SPECIES_ORDER


ALL_SPECIES = list(SPECIES_ORDER)


@dataclass(frozen=True)
class SolveResult:
    y: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float


def residual(x: np.ndarray, y_base: np.ndarray, solve_indices: list[int], reactions, scales: np.ndarray) -> np.ndarray:
    y = y_base.copy()
    y[solve_indices] = np.exp(x)
    dydt = derivative(y, reactions, SPECIES_ORDER)
    return dydt[solve_indices] / scales


def solve_full_27(y0: np.ndarray, reactions, max_iter: int = 80, tolerance: float = 1.0e-10) -> SolveResult:
    solve_indices = list(range(len(SPECIES_ORDER)))
    x = np.log(np.maximum(y0[solve_indices], 1.0e-300))
    scales = throughput_scales(y0, reactions, SPECIES_ORDER, solve_indices)

    best_x = x.copy()
    best_norm = float(np.linalg.norm(residual(x, y0, solve_indices, reactions, scales)))
    for iteration in range(1, max_iter + 1):
        f0 = residual(x, y0, solve_indices, reactions, scales)
        norm0 = float(np.linalg.norm(f0))
        if norm0 < best_norm:
            best_norm = norm0
            best_x = x.copy()
        if norm0 < tolerance:
            y = y0.copy()
            y[solve_indices] = np.exp(x)
            return SolveResult(y, True, iteration - 1, norm0)

        jac = np.empty((len(x), len(x)), dtype=float)
        step_size = 1.0e-5
        for col in range(len(x)):
            xp = x.copy()
            xp[col] += step_size
            jac[:, col] = (residual(xp, y0, solve_indices, reactions, scales) - f0) / step_size

        try:
            step = np.linalg.solve(jac, -f0)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(jac, -f0, rcond=None)

        max_abs_step = float(np.max(np.abs(step)))
        if max_abs_step > 2.0:
            step *= 2.0 / max_abs_step

        accepted = False
        damping = 1.0
        while damping >= 1.0e-5:
            candidate = x + damping * step
            candidate_norm = float(np.linalg.norm(residual(candidate, y0, solve_indices, reactions, scales)))
            if candidate_norm < norm0:
                x = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break

    y = y0.copy()
    y[solve_indices] = np.exp(best_x)
    return SolveResult(y, False, iteration, best_norm)
