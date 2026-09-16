"""Solve the full atmospheric subset of the Young model rebuild.

This diagnostic solves all stratosphere and troposphere species together while
leaving biosphere/hydrosphere water placeholders fixed and excluding geosphere
bookkeeping reservoirs from the convergence target.
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
from solve_coupled_isotope_subsystem import solve_subsystem
from solver_utils import throughput_scales
from table3_state import SPECIES_ORDER
from young_reactions import executable_reactions


ATMOSPHERE_SPECIES = [
    "O_strat",
    "O17_strat",
    "O18_strat",
    "O1D_strat",
    "O17_1D_strat",
    "O18_1D_strat",
    "O2_strat",
    "O17O_strat",
    "O18O_strat",
    "CO2_strat",
    "CO17O_strat",
    "CO18O_strat",
    "O3_strat",
    "OO17O_strat",
    "OO18O_strat",
    "O2_trop",
    "O18O_trop",
    "O17O_trop",
    "CO2_trop",
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


def solve_full_atmosphere(y0: np.ndarray, reactions, max_iter: int = 60, tolerance: float = 1.0e-10) -> SolveResult:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    solve_indices = [idx[name] for name in ATMOSPHERE_SPECIES]
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
    parser.add_argument("--co2-closure-isotope-mode", choices=("rounded_table3", "printed_table3", "smow"), default="printed_table3")
    parser.add_argument("--staged-initial", action="store_true", help="Pre-balance fast stratosphere/CO2 isotope subsystem before full solve.")
    args = parser.parse_args()

    y0 = scaled_table3_state(args.pO2_pal, args.pCO2_ppm, isotope_mode="printed")
    r5_m = 5.589807e18 if args.r5_mode == "table3-balanced" else None
    reactions = executable_reactions(
        r5_collision_partner_moles=r5_m,
        gpp_scale=args.gpp_scale,
        closure_mode=args.closure_mode,
        co2_photo_sink_mode=args.co2_photo_sink_mode,
        co2_closure_isotope_mode=args.co2_closure_isotope_mode,
    )
    if args.staged_initial:
        staged = solve_subsystem(y0, reactions)
        y0 = staged.y
        print(f"staged coupled solve converged={staged.converged} norm={staged.residual_norm:.6e}")
        print()
    result = solve_full_atmosphere(y0, reactions)
    dydt = derivative(result.y, reactions, SPECIES_ORDER)
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}

    print("Full atmosphere steady-state diagnostic")
    print("---------------------------------------")
    print(f"converged:  {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"norm:       {result.residual_norm:.6e}")
    print()
    print("Major atmospheric reservoirs:")
    for name in ("O2_trop", "O2_strat", "CO2_trop", "CO2_strat", "O3_strat"):
        print(f"{name:10s} {result.y[idx[name]]:.6e} mol  dydt={dydt[idx[name]]:.6e} mol/yr")
    print()
    print("Isotope summaries after full-atmosphere solve:")
    for summary in isotope_summaries(result.y):
        print(
            f"{summary.label:10s} d18'={summary.delta18_prime:9.3f} per mil "
            f"d17'={summary.delta17_prime:9.3f} per mil "
            f"D17'={summary.cap_delta17:8.3f} per mil"
        )
    print()
    print("Largest relative residuals:")
    excluded = {"O_bio", "O18_bio", "O17_bio", "O_geo", "O18_geo", "O17_geo"}
    printed = 0
    for name, value, resid, rel in largest_residuals(result.y, dydt, n=len(SPECIES_ORDER)):
        if name in excluded:
            continue
        print(f"{name:14s} y={value:.4e} dydt={resid:.4e} rel/yr={rel:.4e}")
        printed += 1
        if printed >= 18:
            break


if __name__ == "__main__":
    main()
