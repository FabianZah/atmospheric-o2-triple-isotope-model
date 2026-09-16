"""Experimental explicit lower-stratospheric CO2 isotope box.

This module is deliberately separate from the public 27-species scenario API.
It adds three explicit lower-stratosphere CO2 species:

- CO2_lower
- CO18O_lower
- CO17O_lower

The existing `CO2_strat` species is treated as the upper photochemical
production/exchange box. The lower box receives CO2 from the upper box and the
troposphere, then exports to the troposphere. This is a controlled prototype to
test whether a real lower export box can replace the high-pCO2 damping used in
the Young-reproduction branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibrated_model import build_reactions
from conservative_column_transport import AtmosphericLayer, ConservativeColumn, ExchangeInterface
from isotopes import R17_VSMOW, R18_VSMOW, cap_delta17_from_primes
from model_runner import IsotopeSummary, isotope_summaries, scaled_table3_state
from model_scenarios import ScenarioInput, config_from_scenario
from reactions import Reaction, derivative
from solve_fixed_reservoir_isotopes import FIXED_RESERVOIR_SOLVE_SPECIES, SolveResult
from solver_utils import throughput_scales
from table3_state import SPECIES_ORDER
from young_model_inventory import PARAMETERS


LOWER_CO2_SPECIES = ("CO2_lower", "CO18O_lower", "CO17O_lower")
EXTENDED_SPECIES_ORDER = [*SPECIES_ORDER, *LOWER_CO2_SPECIES]
MODERN_CO2_STRAT_MOL = 5.29e15
MODERN_CO2_TROP_MOL = 5.29e16


@dataclass(frozen=True)
class LowerBoxConfig:
    """Configuration for the lower CO2 box transport prototype."""

    upper_survival_fraction: float = 0.42
    lower_to_trop_rate_per_year: float = PARAMETERS["k_ST_per_year"]
    net_export_rate_per_year: float | None = None
    lower_major_scale: float = 1.0

    @property
    def effective_net_export_rate_per_year(self) -> float:
        """Return the one-way export rate used for flux diagnostics."""

        if self.net_export_rate_per_year is None:
            return self.lower_to_trop_rate_per_year
        return self.net_export_rate_per_year


def lower_box_transport_column(lower_box: LowerBoxConfig) -> ConservativeColumn:
    """Return the conservative three-layer carrier-air transport column.

    The upper and tropospheric air inventories retain Young's 1:10
    stratosphere/troposphere ratio. ``lower_major_scale`` sets the lower-box
    carrier inventory relative to Young's stratospheric inventory. The exchange
    fluxes reproduce the existing lower-box rate construction exactly.
    """

    upper_air = PARAMETERS["moles_stratosphere"]
    lower_air = PARAMETERS["moles_stratosphere"] * lower_box.lower_major_scale
    trop_air = PARAMETERS["moles_troposphere"]
    lower_trop_flux = lower_box.lower_to_trop_rate_per_year * lower_air
    f_upper = min(max(lower_box.upper_survival_fraction, 1.0e-6), 0.999)
    upper_lower_flux = f_upper / (1.0 - f_upper) * lower_trop_flux
    return ConservativeColumn(
        layers=(
            AtmosphericLayer("troposphere", trop_air),
            AtmosphericLayer("lower_stratosphere", lower_air),
            AtmosphericLayer("upper_stratosphere", upper_air),
        ),
        interfaces=(
            ExchangeInterface(
                "troposphere",
                "lower_stratosphere",
                lower_trop_flux,
                "Young et al. (2014) kST scale; explicit lower-box split",
            ),
            ExchangeInterface(
                "lower_stratosphere",
                "upper_stratosphere",
                upper_lower_flux,
                "Explicit lower-box source-partition diagnostic",
            ),
        ),
    )


def _prime_delta(heavy: float, major: float, reference_ratio: float) -> float:
    ratio = max(heavy, 1.0e-300) / max(major, 1.0e-300)
    return 1000.0 * float(np.log(max(ratio / reference_ratio, 1.0e-300)))


def co2_summary(label: str, major: float, n17: float, n18: float) -> IsotopeSummary:
    d17p = _prime_delta(n17, major, R17_VSMOW)
    d18p = _prime_delta(n18, major, R18_VSMOW)
    return IsotopeSummary(label, d18p, d17p, cap_delta17_from_primes(d17p, d18p))


def extended_initial_state(
    p_o2_pal: float,
    p_co2_ppm: float,
    lower_box: LowerBoxConfig,
) -> np.ndarray:
    """Return an extended initial state with a lower CO2 box."""

    base = scaled_table3_state(p_o2_pal, p_co2_ppm, isotope_mode="printed")
    idx_base = {name: i for i, name in enumerate(SPECIES_ORDER)}
    idx_ext = {name: i for i, name in enumerate(EXTENDED_SPECIES_ORDER)}
    y = np.zeros(len(EXTENDED_SPECIES_ORDER), dtype=float)
    y[: len(SPECIES_ORDER)] = base

    lower_major = MODERN_CO2_STRAT_MOL * lower_box.lower_major_scale * (p_co2_ppm / 294.4)
    strat = co2_summary(
        "CO2_strat",
        base[idx_base["CO2_strat"]],
        base[idx_base["CO17O_strat"]],
        base[idx_base["CO18O_strat"]],
    )
    trop = co2_summary(
        "CO2_trop",
        base[idx_base["CO2_trop"]],
        base[idx_base["CO17O_trop"]],
        base[idx_base["CO18O_trop"]],
    )
    f_upper = lower_box.upper_survival_fraction
    lower_d18p = trop.delta18_prime + f_upper * (strat.delta18_prime - trop.delta18_prime)
    lower_d17p = trop.delta17_prime + f_upper * (strat.delta17_prime - trop.delta17_prime)
    y[idx_ext["CO2_lower"]] = lower_major
    y[idx_ext["CO18O_lower"]] = lower_major * R18_VSMOW * float(np.exp(lower_d18p / 1000.0))
    y[idx_ext["CO17O_lower"]] = lower_major * R17_VSMOW * float(np.exp(lower_d17p / 1000.0))
    return y


def explicit_lower_box_transport_reactions(
    p_co2_ppm: float,
    lower_box: LowerBoxConfig,
) -> list[Reaction]:
    """Return upper/lower/troposphere CO2 transport reactions.

    `upper_survival_fraction` defines the fraction of the lower-box isotope
    source flux derived from the upper photochemical box. Major reservoirs are
    balanced by using bidirectional upper/lower and lower/troposphere exchange:

    - upper <-> lower fluxes are equal
    - lower <-> troposphere fluxes are equal
    - F_upper_lower / (F_upper_lower + F_trop_lower) = f_upper
    """

    if p_co2_ppm <= 0.0:
        raise ValueError("pCO2 must be positive")
    column = lower_box_transport_column(lower_box)
    rate_rows = column.interface_rate_constants_per_year()
    trop_lower = rate_rows[0]
    lower_upper = rate_rows[1]
    k_trop_lower = float(trop_lower["first_to_second_per_year"])
    k_lower_trop = float(trop_lower["second_to_first_per_year"])
    k_lower_upper = float(lower_upper["first_to_second_per_year"])
    k_upper_lower = float(lower_upper["second_to_first_per_year"])
    pairs = (
        ("CO2_strat", "CO2_lower", "CO2_trop"),
        ("CO18O_strat", "CO18O_lower", "CO18O_trop"),
        ("CO17O_strat", "CO17O_lower", "CO17O_trop"),
    )
    reactions: list[Reaction] = []
    for upper, lower, trop in pairs:
        reactions.append(
            Reaction(
                f"k_UL_{upper}",
                {upper: 1.0},
                {lower: 1.0},
                k_upper_lower,
                "yr^-1",
                "upper stratosphere -> lower stratosphere CO2 transport",
            )
        )
        reactions.append(
            Reaction(
                f"k_LU_{lower}",
                {lower: 1.0},
                {upper: 1.0},
                k_lower_upper,
                "yr^-1",
                "lower stratosphere -> upper stratosphere CO2 transport",
            )
        )
        reactions.append(
            Reaction(
                f"k_TL_{trop}",
                {trop: 1.0},
                {lower: 1.0},
                k_trop_lower,
                "yr^-1",
                "troposphere -> lower stratosphere CO2 mixing",
            )
        )
        reactions.append(
            Reaction(
                f"k_LT_{lower}",
                {lower: 1.0},
                {trop: 1.0},
                k_lower_trop,
                "yr^-1",
                "lower stratosphere -> troposphere CO2 export",
            )
        )
    return reactions


def replace_co2_transport_with_lower_box(
    reactions: list[Reaction],
    p_co2_ppm: float,
    lower_box: LowerBoxConfig,
) -> list[Reaction]:
    """Remove one-box CO2 ST/TS reactions and add explicit lower-box transport."""

    removed = {
        "k_ST_CO2_strat",
        "k_ST_CO18O_strat",
        "k_ST_CO17O_strat",
        "k_TS_CO2_trop",
        "k_TS_CO18O_trop",
        "k_TS_CO17O_trop",
    }
    return [
        *(reaction for reaction in reactions if reaction.key not in removed),
        *explicit_lower_box_transport_reactions(p_co2_ppm, lower_box),
    ]


def extended_reactions(scenario: ScenarioInput, lower_box: LowerBoxConfig) -> list[Reaction]:
    config = config_from_scenario(scenario)
    reactions = build_reactions(config)
    return replace_co2_transport_with_lower_box(reactions, config.p_co2_ppm, lower_box)


def extended_fixed_solve_species() -> list[str]:
    return [*FIXED_RESERVOIR_SOLVE_SPECIES, "CO18O_lower", "CO17O_lower"]


def _residual(
    x: np.ndarray,
    y_base: np.ndarray,
    solve_indices: list[int],
    reactions: list[Reaction],
    scales: np.ndarray,
) -> np.ndarray:
    y = y_base.copy()
    y[solve_indices] = np.exp(x)
    dydt = derivative(y, reactions, EXTENDED_SPECIES_ORDER)
    return dydt[solve_indices] / scales


def solve_explicit_lower_box_fixed_reservoir(
    scenario: ScenarioInput,
    lower_box: LowerBoxConfig = LowerBoxConfig(),
    max_iter: int = 80,
    tolerance: float = 1.0e-12,
) -> tuple[SolveResult, np.ndarray, list[Reaction]]:
    """Solve the fixed-reservoir isotope subsystem with explicit lower CO2."""

    config = config_from_scenario(scenario)
    y0 = extended_initial_state(config.p_o2_pal, config.p_co2_ppm, lower_box)
    reactions = extended_reactions(scenario, lower_box)
    idx = {name: i for i, name in enumerate(EXTENDED_SPECIES_ORDER)}
    solve_indices = [idx[name] for name in extended_fixed_solve_species()]
    x = np.log(np.maximum(y0[solve_indices], 1.0e-300))
    scales = throughput_scales(y0, reactions, EXTENDED_SPECIES_ORDER, solve_indices)

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


def summarize_extended_state(y: np.ndarray) -> dict[str, IsotopeSummary]:
    """Return isotope summaries including the explicit lower CO2 box."""

    idx = {name: i for i, name in enumerate(EXTENDED_SPECIES_ORDER)}
    summaries = {summary.label: summary for summary in isotope_summaries(y[: len(SPECIES_ORDER)])}
    summaries["CO2_lower"] = co2_summary(
        "CO2_lower",
        y[idx["CO2_lower"]],
        y[idx["CO17O_lower"]],
        y[idx["CO18O_lower"]],
    )
    return summaries
