"""Experimental processed high-altitude CO2 reservoir.

This prototype extends the explicit lower-stratospheric CO2 box with three
additional processed CO2 species:

- CO2_proc
- CO18O_proc
- CO17O_proc

The processed box receives a constant isotope-resolved O(1D)-derived source
and exports to the lower CO2 box. Major reservoirs are treated as fixed, as in
the existing Fig. 8 fixed-reservoir diagnostic. This file is intentionally not
wired into the public scenario presets yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from explicit_lower_co2_box_model import (
    EXTENDED_SPECIES_ORDER as LOWER_SPECIES_ORDER,
    LOWER_CO2_SPECIES,
    MODERN_CO2_STRAT_MOL,
    LowerBoxConfig,
    co2_summary,
    extended_initial_state,
    extended_reactions,
    summarize_extended_state,
)
from isotopes import R17_VSMOW, R18_VSMOW
from model_runner import set_singly_substituted_from_primes
from model_scenarios import ScenarioInput, config_from_scenario
from processed_altitude_reservoir import O1DIsotopeTransferSource
from reactions import Reaction, derivative
from solve_fixed_reservoir_isotopes import FIXED_RESERVOIR_SOLVE_SPECIES, SolveResult
from solver_utils import throughput_scales
from young_model_inventory import TABLE3_TARGETS


PROCESSED_CO2_SPECIES = ("CO2_proc", "CO18O_proc", "CO17O_proc")
PROCESSED_SPECIES_ORDER = [*LOWER_SPECIES_ORDER, *PROCESSED_CO2_SPECIES]


@dataclass(frozen=True)
class ProcessedBoxConfig:
    """Configuration for the processed high-altitude CO2 reservoir."""

    reservoir_fraction_of_lower: float = 0.01
    source_flux_mol_per_year: float = 1.485e14
    activation: float = 1.0
    transfer_efficiency: float = 1.0
    transfer_mode: str = "inherit_o1d_primes"
    route_to_lower: bool = True

    @property
    def turnover_rate_per_year(self) -> float:
        lower_major = MODERN_CO2_STRAT_MOL
        reservoir_mol = lower_major * self.reservoir_fraction_of_lower
        if reservoir_mol <= 0.0:
            raise ValueError("processed reservoir size must be positive")
        return self.source_flux_mol_per_year / reservoir_mol


def processed_signature(config: ProcessedBoxConfig):
    source = O1DIsotopeTransferSource(
        o1d_delta17_prime_permil=TABLE3_TARGETS["d17_O1D_permil"],
        o1d_delta18_prime_permil=TABLE3_TARGETS["d18_O1D_permil"],
        activation=config.activation,
        transfer_efficiency=config.transfer_efficiency,
        mode=config.transfer_mode,
    )
    return source.processed_signature()


def processed_initial_state(
    scenario: ScenarioInput,
    lower_box: LowerBoxConfig,
    processed_box: ProcessedBoxConfig,
) -> np.ndarray:
    config = config_from_scenario(scenario)
    lower_state = extended_initial_state(config.p_o2_pal, config.p_co2_ppm, lower_box)
    y = np.zeros(len(PROCESSED_SPECIES_ORDER), dtype=float)
    y[: len(LOWER_SPECIES_ORDER)] = lower_state
    idx = {name: i for i, name in enumerate(PROCESSED_SPECIES_ORDER)}
    signature = processed_signature(processed_box)
    co2_proc = (
        MODERN_CO2_STRAT_MOL
        * lower_box.lower_major_scale
        * processed_box.reservoir_fraction_of_lower
        * (config.p_co2_ppm / 294.4)
    )
    y[idx["CO2_proc"]] = co2_proc
    set_singly_substituted_from_primes(
        y,
        idx,
        "CO2_proc",
        "CO17O_proc",
        "CO18O_proc",
        signature.delta17_prime_permil,
        signature.delta18_prime_permil,
        1.0,
    )
    return y


def processed_reactions(
    scenario: ScenarioInput,
    lower_box: LowerBoxConfig,
    processed_box: ProcessedBoxConfig,
) -> list[Reaction]:
    config = config_from_scenario(scenario)
    reactions = extended_reactions(scenario, lower_box)
    scale = config.p_co2_ppm / 294.4
    source_flux = processed_box.source_flux_mol_per_year * scale * lower_box.lower_major_scale
    signature = processed_signature(processed_box)
    r17 = R17_VSMOW * float(np.exp(signature.delta17_prime_permil / 1000.0))
    r18 = R18_VSMOW * float(np.exp(signature.delta18_prime_permil / 1000.0))
    reservoir_mol = (
        MODERN_CO2_STRAT_MOL
        * lower_box.lower_major_scale
        * processed_box.reservoir_fraction_of_lower
        * scale
    )
    k_loss = source_flux / reservoir_mol
    co2_proc_loss_products = {"CO2_lower": 1.0} if processed_box.route_to_lower else {}
    co17o_proc_loss_products = {"CO17O_lower": 1.0} if processed_box.route_to_lower else {}
    co18o_proc_loss_products = {"CO18O_lower": 1.0} if processed_box.route_to_lower else {}
    route_note = "processed reservoir export to lower box" if processed_box.route_to_lower else "processed reservoir parallel export sink"
    return [
        *reactions,
        Reaction("P_CO2_proc_source", {}, {"CO2_proc": 1.0}, source_flux, "mol yr^-1", "processed CO2 major source"),
        Reaction(
            "P_CO17O_proc_source",
            {},
            {"CO17O_proc": 1.0},
            source_flux * r17,
            "mol yr^-1",
            "processed CO17O source from O(1D)-derived signature",
        ),
        Reaction(
            "P_CO18O_proc_source",
            {},
            {"CO18O_proc": 1.0},
            source_flux * r18,
            "mol yr^-1",
            "processed CO18O source from O(1D)-derived signature",
        ),
        Reaction("P_CO2_proc_loss", {"CO2_proc": 1.0}, co2_proc_loss_products, k_loss, "yr^-1", route_note),
        Reaction("P_CO17O_proc_loss", {"CO17O_proc": 1.0}, co17o_proc_loss_products, k_loss, "yr^-1", route_note),
        Reaction("P_CO18O_proc_loss", {"CO18O_proc": 1.0}, co18o_proc_loss_products, k_loss, "yr^-1", route_note),
    ]


def processed_export_flux_mol_per_year(scenario: ScenarioInput, lower_box: LowerBoxConfig, processed_box: ProcessedBoxConfig) -> float:
    """Return processed major CO2 export/source flux for diagnostics."""

    config = config_from_scenario(scenario)
    return processed_box.source_flux_mol_per_year * (config.p_co2_ppm / 294.4) * lower_box.lower_major_scale


def processed_solve_species() -> list[str]:
    return [*FIXED_RESERVOIR_SOLVE_SPECIES, "CO18O_lower", "CO17O_lower", "CO18O_proc", "CO17O_proc"]


def _residual(
    x: np.ndarray,
    y_base: np.ndarray,
    solve_indices: list[int],
    reactions: list[Reaction],
    scales: np.ndarray,
) -> np.ndarray:
    y = y_base.copy()
    y[solve_indices] = np.exp(x)
    dydt = derivative(y, reactions, PROCESSED_SPECIES_ORDER)
    return dydt[solve_indices] / scales


def solve_processed_box_fixed_reservoir(
    scenario: ScenarioInput,
    lower_box: LowerBoxConfig = LowerBoxConfig(),
    processed_box: ProcessedBoxConfig = ProcessedBoxConfig(),
    max_iter: int = 80,
    tolerance: float = 1.0e-12,
) -> tuple[SolveResult, np.ndarray, list[Reaction]]:
    y0 = processed_initial_state(scenario, lower_box, processed_box)
    reactions = processed_reactions(scenario, lower_box, processed_box)
    idx = {name: i for i, name in enumerate(PROCESSED_SPECIES_ORDER)}
    solve_indices = [idx[name] for name in processed_solve_species()]
    x = np.log(np.maximum(y0[solve_indices], 1.0e-300))
    scales = throughput_scales(y0, reactions, PROCESSED_SPECIES_ORDER, solve_indices)

    best_x = x.copy()
    best_norm = float(np.linalg.norm(_residual(x, y0, solve_indices, reactions, scales)))
    iteration = 0
    for iteration in range(1, max_iter + 1):
        f0 = _residual(x, y0, solve_indices, reactions, scales)
        norm0 = float(np.linalg.norm(f0))
        if norm0 < best_norm:
            best_norm = norm0
            best_x = x.copy()
        if norm0 < tolerance:
            y = y0.copy()
            y[solve_indices] = np.exp(x)
            return SolveResult(y, True, iteration - 1, norm0), y0, reactions

        jac = np.empty((len(x), len(x)), dtype=float)
        step_size = 1.0e-5
        for col in range(len(x)):
            xp = x.copy()
            xp[col] += step_size
            jac[:, col] = (_residual(xp, y0, solve_indices, reactions, scales) - f0) / step_size

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
            candidate_norm = float(np.linalg.norm(_residual(candidate, y0, solve_indices, reactions, scales)))
            if candidate_norm < norm0:
                x = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break

    y = y0.copy()
    y[solve_indices] = np.exp(best_x)
    return SolveResult(y, False, iteration, best_norm), y0, reactions


def summarize_processed_state(y: np.ndarray):
    summaries = summarize_extended_state(y[: len(LOWER_SPECIES_ORDER)])
    idx = {name: i for i, name in enumerate(PROCESSED_SPECIES_ORDER)}
    summaries["CO2_proc"] = co2_summary(
        "CO2_proc",
        y[idx["CO2_proc"]],
        y[idx["CO17O_proc"]],
        y[idx["CO18O_proc"]],
    )
    return summaries
