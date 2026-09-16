"""Steady-state mass-balance alternatives for the Young reconstruction.

This module is intentionally separate from the ODE/Newton solvers. It asks a
more limited question: given a reaction context, what tropospheric O2 isotope
composition follows from direct source/loss mass balance?

For a singly substituted O2 isotopologue H in the troposphere:

    dH/dt = sources_H - loss_coeff_H * H = 0
    H = sources_H / loss_coeff_H

This first branch does not claim to recover Young's full 27 ODE model. It is a
transparent competing steady-state diagnostic that can be scored against the
published Young figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from calibrated_model import ModelConfig, build_reactions, run_model
from model_runner import isotope_summaries, scaled_table3_state
from reactions import derivative
from table3_state import SPECIES_ORDER


O2_HEAVY_SPECIES = ("O18O_trop", "O17O_trop")


@dataclass(frozen=True)
class SpeciesBalance:
    species: str
    source_mol_per_year: float
    loss_per_year: float
    solved_moles: float
    original_moles: float


@dataclass(frozen=True)
class MassBalanceResult:
    config: ModelConfig
    context: str
    y: np.ndarray
    balances: tuple[SpeciesBalance, ...]
    outputs: dict[str, float | str | bool]


def reaction_contribution(reaction: Any, y: np.ndarray, idx: dict[str, int], species: str) -> float:
    coeff = reaction.products.get(species, 0.0) - reaction.reactants.get(species, 0.0)
    if coeff == 0.0:
        return 0.0
    return coeff * reaction.rate(y, idx)


def linear_source_loss_for_species(
    y: np.ndarray,
    reactions: list[Any],
    species: str,
) -> tuple[float, float]:
    """Return source and first-order loss coefficient for one species.

    This is exact for the current tropospheric O2 heavy-isotopologue reactions,
    where losses are first-order in the target species and sources are
    independent of that target species.
    """

    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    source = 0.0
    loss = 0.0
    original = float(y[idx[species]])
    if original <= 0.0:
        raise ValueError(f"{species} must be positive for mass-balance decomposition")

    for reaction in reactions:
        contribution = reaction_contribution(reaction, y, idx, species)
        if contribution == 0.0:
            continue
        if species in reaction.reactants:
            # Contributions involving the target as a reactant are treated as
            # first-order losses. If a future reaction both consumes and
            # produces the same species, the net coefficient is still captured.
            loss += -contribution / original
        else:
            source += contribution
    return source, loss


def solve_tropospheric_o2_mass_balance(
    config: ModelConfig,
    *,
    context: str = "ode_context",
) -> MassBalanceResult:
    """Compute O2_trop heavy isotopologues from algebraic mass balance.

    Context choices:

    - `ode_context`: run the current model first, then recompute only
      tropospheric O18O/O17O by direct mass balance in that solved context.
    - `scaled_table3_context`: use the scaled Table-3 state directly, with no
      photochemical/isotope solve, then apply the same direct O2 balance.
    """

    if context == "ode_context":
        base = run_model(config).y.copy()
    elif context == "scaled_table3_context":
        base = scaled_table3_state(config.p_o2_pal, config.p_co2_ppm, isotope_mode="printed")
    else:
        raise ValueError("context must be 'ode_context' or 'scaled_table3_context'")

    reactions = build_reactions(config)
    y = base.copy()
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    balances: list[SpeciesBalance] = []
    for species in O2_HEAVY_SPECIES:
        source, loss = linear_source_loss_for_species(y, reactions, species)
        solved = source / loss if loss > 0.0 else np.nan
        balances.append(
            SpeciesBalance(
                species=species,
                source_mol_per_year=source,
                loss_per_year=loss,
                solved_moles=float(solved),
                original_moles=float(y[idx[species]]),
            )
        )
        y[idx[species]] = solved

    summaries = {summary.label: summary for summary in isotope_summaries(y)}
    dydt = derivative(y, reactions, SPECIES_ORDER)
    outputs = {
        "context": context,
        "O2_trop_D17O_permil": summaries["O2_trop"].cap_delta17,
        "O2_trop_d18_prime_permil": summaries["O2_trop"].delta18_prime,
        "O2_trop_d17_prime_permil": summaries["O2_trop"].delta17_prime,
        "CO2_trop_D17O_permil": summaries["CO2_trop"].cap_delta17,
        "CO2_strat_D17O_permil": summaries["CO2_strat"].cap_delta17,
        "O3_strat_D17O_permil": summaries["O3_strat"].cap_delta17,
        "O18O_trop_residual_mol_per_year": float(dydt[idx["O18O_trop"]]),
        "O17O_trop_residual_mol_per_year": float(dydt[idx["O17O_trop"]]),
    }
    return MassBalanceResult(
        config=replace(config),
        context=context,
        y=y,
        balances=tuple(balances),
        outputs=outputs,
    )
