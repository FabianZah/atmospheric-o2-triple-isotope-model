"""Solve the fast stratospheric chemistry subsystem at fixed reservoirs.

This is a diagnostic steady-state solver, not a replacement for Young's full
DLSODE integration. It holds the slow O2/CO2 reservoirs fixed and adjusts the
fast stratospheric species whose lifetimes make direct residual inspection hard.
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


FAST_SPECIES = [
    "O_strat",
    "O17_strat",
    "O18_strat",
    "O1D_strat",
    "O17_1D_strat",
    "O18_1D_strat",
    "O3_strat",
    "OO17O_strat",
    "OO18O_strat",
]


@dataclass(frozen=True)
class SolveResult:
    y: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float


def fast_residual(
    x: np.ndarray,
    y_base: np.ndarray,
    fast_indices: list[int],
    reactions,
    scales: np.ndarray,
) -> np.ndarray:
    y = y_base.copy()
    y[fast_indices] = np.exp(x)
    dydt = derivative(y, reactions, SPECIES_ORDER)
    return dydt[fast_indices] / scales


def solve_fast_species(
    y0: np.ndarray,
    reactions,
    max_iter: int = 40,
    tolerance: float = 1.0e-12,
) -> SolveResult:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    fast_indices = [idx[name] for name in FAST_SPECIES]
    x = np.log(np.maximum(y0[fast_indices], 1.0e-300))
    y_base = y0.copy()

    # Fixed scales based on gross local throughput prevent tiny or nearly
    # balanced species from dominating only because their net derivative is
    # small at the initial state.
    scales = throughput_scales(y0, reactions, SPECIES_ORDER, fast_indices)

    best_norm = np.inf
    for iteration in range(1, max_iter + 1):
        f0 = fast_residual(x, y_base, fast_indices, reactions, scales)
        norm0 = float(np.linalg.norm(f0, ord=2))
        best_norm = norm0
        if norm0 < tolerance:
            y = y_base.copy()
            y[fast_indices] = np.exp(x)
            return SolveResult(y, True, iteration - 1, norm0)

        jac = np.empty((len(x), len(x)))
        step_size = 1.0e-5
        for col in range(len(x)):
            xp = x.copy()
            xp[col] += step_size
            jac[:, col] = (fast_residual(xp, y_base, fast_indices, reactions, scales) - f0) / step_size

        try:
            step = np.linalg.solve(jac, -f0)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(jac, -f0, rcond=None)

        # Avoid wild excursions in log-space while still allowing large updates.
        max_abs_step = float(np.max(np.abs(step)))
        if max_abs_step > 4.0:
            step *= 4.0 / max_abs_step

        accepted = False
        damping = 1.0
        while damping >= 1.0e-4:
            candidate = x + damping * step
            f_candidate = fast_residual(candidate, y_base, fast_indices, reactions, scales)
            norm_candidate = float(np.linalg.norm(f_candidate, ord=2))
            if norm_candidate < norm0:
                x = candidate
                best_norm = norm_candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break

    y = y_base.copy()
    y[fast_indices] = np.exp(x)
    return SolveResult(y, False, iteration, best_norm)


def print_fast_table(y_before: np.ndarray, y_after: np.ndarray, reactions) -> None:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    before = derivative(y_before, reactions, SPECIES_ORDER)
    after = derivative(y_after, reactions, SPECIES_ORDER)
    print("Fast species:")
    for name in FAST_SPECIES:
        i = idx[name]
        print(
            f"{name:14s} before={y_before[i]:.6e} after={y_after[i]:.6e} "
            f"dydt_before={before[i]: .3e} dydt_after={after[i]: .3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pO2-pal", type=float, default=1.0)
    parser.add_argument("--pCO2-ppm", type=float, default=294.4)
    parser.add_argument("--gpp-scale", type=float, default=1.0)
    parser.add_argument("--isotope-mode", choices=("printed", "rounded"), default="printed")
    parser.add_argument("--r5-mode", choices=("paper", "table3-balanced"), default="table3-balanced")
    parser.add_argument("--closure-mode", choices=("none", "modern"), default="none")
    parser.add_argument("--co2-photo-sink-factor", type=float, default=1.0)
    parser.add_argument("--co2-photo-sink-mode", choices=("smow", "o2_source_water"), default="smow")
    parser.add_argument("--max-iter", type=int, default=40)
    args = parser.parse_args()

    y0 = scaled_table3_state(args.pO2_pal, args.pCO2_ppm, isotope_mode=args.isotope_mode)
    r5_m = 5.589807e18 if args.r5_mode == "table3-balanced" else None
    reactions = executable_reactions(
        r5_collision_partner_moles=r5_m,
        gpp_scale=args.gpp_scale,
        closure_mode=args.closure_mode,
        co2_photo_sink_factor=args.co2_photo_sink_factor,
        co2_photo_sink_mode=args.co2_photo_sink_mode,
    )
    result = solve_fast_species(y0, reactions, max_iter=args.max_iter)

    print("Fast stratosphere steady-state diagnostic")
    print("-----------------------------------------")
    print(f"converged:  {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"norm:       {result.residual_norm:.6e}")
    print()
    print_fast_table(y0, result.y, reactions)
    print()
    print("Isotope summaries after fast solve:")
    for summary in isotope_summaries(result.y):
        print(
            f"{summary.label:10s} d18'={summary.delta18_prime:9.3f} per mil "
            f"d17'={summary.delta17_prime:9.3f} per mil "
            f"D17'={summary.cap_delta17:8.3f} per mil"
        )
    print()
    print("Largest relative residuals after fast solve:")
    dydt = derivative(result.y, reactions, SPECIES_ORDER)
    for name, value, resid, rel in largest_residuals(result.y, dydt):
        print(f"{name:14s} y={value:.4e} dydt={resid:.4e} rel/yr={rel:.4e}")


if __name__ == "__main__":
    main()
