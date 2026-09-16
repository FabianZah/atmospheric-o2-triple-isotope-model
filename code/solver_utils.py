"""Numerical helper functions for diagnostic steady-state solvers."""

from __future__ import annotations

import numpy as np


def throughput_scales(y: np.ndarray, reactions, species_order: list[str], solve_indices: list[int]) -> np.ndarray:
    """Return per-species residual scales from local gross reaction throughput.

    Earlier solvers scaled residuals by the net derivative at the initial state.
    That is fragile for continuation/sensitivity tests: if a species begins
    near balance, an innocuous perturbation can be divided by an arbitrarily
    tiny number. Gross source/sink throughput is a more stable denominator for
    asking whether a species is close to steady state.
    """

    index = {name: i for i, name in enumerate(species_order)}
    solve_set = set(solve_indices)
    scales = np.zeros(len(solve_indices), dtype=float)
    scale_pos = {species_index: i for i, species_index in enumerate(solve_indices)}

    for reaction in reactions:
        dy = np.zeros_like(y, dtype=float)
        if hasattr(reaction, "apply_state"):
            reaction.apply_state(dy, y, index)
        else:
            reaction.apply(dy, reaction.rate(y, index), index)
        for species_index in solve_set:
            scales[scale_pos[species_index]] += abs(float(dy[species_index]))

    # Keep an absolute floor for species whose current reaction set gives no
    # throughput, and include the net derivative so highly one-sided budgets do
    # not receive an artificially tiny denominator.
    return np.maximum(scales, 1.0)
