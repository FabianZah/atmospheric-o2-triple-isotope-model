"""Solve a coupled fast-stratosphere and CO2-isotope subsystem.

This diagnostic holds the major O2/CO2 reservoirs fixed, then solves:

- fast stratospheric O/O3/O(1D) species
- stratospheric CO18O/CO17O
- tropospheric CO18O/CO17O

It is the next step beyond solving fast stratosphere and tropospheric CO2
isotopologues separately.
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

from model_runner import isotope_summaries, largest_residuals, scaled_table3_state
from reactions import derivative
from solver_utils import throughput_scales
from table3_state import SPECIES_ORDER
from young_reactions import executable_reactions


SOLVE_SPECIES = [
    "O_strat",
    "O17_strat",
    "O18_strat",
    "O1D_strat",
    "O17_1D_strat",
    "O18_1D_strat",
    "O3_strat",
    "OO17O_strat",
    "OO18O_strat",
    "CO18O_strat",
    "CO17O_strat",
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


def solve_subsystem(y0: np.ndarray, reactions, max_iter: int = 50, tolerance: float = 1.0e-12) -> SolveResult:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    solve_indices = [idx[name] for name in SOLVE_SPECIES]
    x = np.log(np.maximum(y0[solve_indices], 1.0e-300))
    scales = throughput_scales(y0, reactions, SPECIES_ORDER, solve_indices)

    best_norm = np.inf
    for iteration in range(1, max_iter + 1):
        f0 = residual(x, y0, solve_indices, reactions, scales)
        norm0 = float(np.linalg.norm(f0, ord=2))
        best_norm = norm0
        if norm0 < tolerance:
            y = y0.copy()
            y[solve_indices] = np.exp(x)
            return SolveResult(y, True, iteration - 1, norm0)

        jac = np.empty((len(x), len(x)))
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
        if max_abs_step > 4.0:
            step *= 4.0 / max_abs_step

        accepted = False
        damping = 1.0
        while damping >= 1.0e-5:
            candidate = x + damping * step
            norm_candidate = float(np.linalg.norm(residual(candidate, y0, solve_indices, reactions, scales), ord=2))
            if norm_candidate < norm0:
                x = candidate
                best_norm = norm_candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break

    y = y0.copy()
    y[solve_indices] = np.exp(x)
    return SolveResult(y, False, iteration, best_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pO2-pal", type=float, default=1.0)
    parser.add_argument("--pCO2-ppm", type=float, default=294.4)
    parser.add_argument("--gpp-scale", type=float, default=1.0)
    parser.add_argument("--isotope-mode", choices=("printed", "rounded"), default="printed")
    parser.add_argument("--r5-mode", choices=("paper", "table3-balanced"), default="table3-balanced")
    parser.add_argument("--closure-mode", choices=("none", "modern"), default="modern")
    parser.add_argument("--co2-photo-sink-mode", choices=("smow", "o2_source_water"), default="o2_source_water")
    parser.add_argument("--co2-photo-sink-factor", type=float, default=1.0)
    parser.add_argument("--co2-closure-isotope-mode", choices=("rounded_table3", "printed_table3", "smow"), default="printed_table3")
    args = parser.parse_args()

    y0 = scaled_table3_state(args.pO2_pal, args.pCO2_ppm, isotope_mode=args.isotope_mode)
    r5_m = 5.589807e18 if args.r5_mode == "table3-balanced" else None
    reactions = executable_reactions(
        r5_collision_partner_moles=r5_m,
        gpp_scale=args.gpp_scale,
        closure_mode=args.closure_mode,
        co2_photo_sink_mode=args.co2_photo_sink_mode,
        co2_photo_sink_factor=args.co2_photo_sink_factor,
        co2_closure_isotope_mode=args.co2_closure_isotope_mode,
    )
    before = derivative(y0, reactions, SPECIES_ORDER)
    result = solve_subsystem(y0, reactions)
    after = derivative(result.y, reactions, SPECIES_ORDER)
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}

    print("Coupled isotope subsystem diagnostic")
    print("------------------------------------")
    print(f"converged:  {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"norm:       {result.residual_norm:.6e}")
    print()
    print("Solved species:")
    for name in SOLVE_SPECIES:
        i = idx[name]
        print(
            f"{name:14s} before={y0[i]:.6e} after={result.y[i]:.6e} "
            f"dydt_before={before[i]: .3e} dydt_after={after[i]: .3e}"
        )
    print()
    print("Isotope summaries after coupled solve:")
    for summary in isotope_summaries(result.y):
        print(
            f"{summary.label:10s} d18'={summary.delta18_prime:9.3f} per mil "
            f"d17'={summary.delta17_prime:9.3f} per mil "
            f"D17'={summary.cap_delta17:8.3f} per mil"
        )
    print()
    print("Largest relative residuals after coupled solve:")
    dydt = derivative(result.y, reactions, SPECIES_ORDER)
    for name, value, resid, rel in largest_residuals(result.y, dydt):
        print(f"{name:14s} y={value:.4e} dydt={resid:.4e} rel/yr={rel:.4e}")


if __name__ == "__main__":
    main()
