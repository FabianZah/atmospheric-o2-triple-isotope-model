"""Solve isotope composition at fixed major O2 and CO2 reservoirs.

This is the preferred diagnostic for Fig. 8-style experiments where pO2 and
pCO2 are treated as coordinates. It holds the major 16O reservoirs fixed:

- O2_trop, O2_strat
- CO2_trop, CO2_strat

and solves the fast stratospheric species plus O2/CO2 isotopologues.
"""

from __future__ import annotations

# --- path bootstrap (direct execution) ---
import sys as _sys
from pathlib import Path as _Path
_root = next((p for p in _Path(__file__).resolve().parents if (p / ".project-root").exists()), None)
if _root is not None:
    for _sub in ("code", "validation"):
        _p = str(_root / _sub)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
# --- end path bootstrap ---
import argparse
from dataclasses import dataclass

import numpy as np

from model_runner import isotope_summaries, scaled_table3_state
from reactions import derivative
from solver_utils import throughput_scales
from table3_state import SPECIES_ORDER
from young_reactions import executable_reactions


FIXED_RESERVOIR_SOLVE_SPECIES = [
    "O_strat",
    "O17_strat",
    "O18_strat",
    "O1D_strat",
    "O17_1D_strat",
    "O18_1D_strat",
    "O17O_strat",
    "O18O_strat",
    "CO17O_strat",
    "CO18O_strat",
    "O3_strat",
    "OO17O_strat",
    "OO18O_strat",
    "O18O_trop",
    "O17O_trop",
    "CO18O_trop",
    "CO17O_trop",
]


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


def solve_fixed_reservoir_isotopes(
    y0: np.ndarray,
    reactions,
    max_iter: int = 80,
    tolerance: float = 1.0e-12,
) -> SolveResult:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    solve_indices = [idx[name] for name in FIXED_RESERVOIR_SOLVE_SPECIES]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pO2-pal", type=float, default=1.0)
    parser.add_argument("--pCO2-ppm", type=float, default=294.4)
    parser.add_argument("--gpp-scale", type=float, default=1.0)
    parser.add_argument("--r5-mode", choices=("paper", "table3-balanced"), default="table3-balanced")
    parser.add_argument("--closure-mode", choices=("none", "modern"), default="modern")
    parser.add_argument("--co2-photo-sink-mode", choices=("smow", "o2_source_water"), default="o2_source_water")
    args = parser.parse_args()

    y0 = scaled_table3_state(args.pO2_pal, args.pCO2_ppm, isotope_mode="printed")
    r5_m = 5.589807e18 if args.r5_mode == "table3-balanced" else None
    reactions = executable_reactions(
        r5_collision_partner_moles=r5_m,
        gpp_scale=args.gpp_scale,
        closure_mode=args.closure_mode,
        co2_photo_sink_mode=args.co2_photo_sink_mode,
    )
    result = solve_fixed_reservoir_isotopes(y0, reactions)
    print("Fixed-reservoir isotope diagnostic")
    print("----------------------------------")
    print(f"converged:  {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"norm:       {result.residual_norm:.6e}")
    print()
    for summary in isotope_summaries(result.y):
        print(
            f"{summary.label:10s} d18'={summary.delta18_prime:9.3f} per mil "
            f"d17'={summary.delta17_prime:9.3f} per mil "
            f"D17'={summary.cap_delta17:8.3f} per mil"
        )


if __name__ == "__main__":
    main()
