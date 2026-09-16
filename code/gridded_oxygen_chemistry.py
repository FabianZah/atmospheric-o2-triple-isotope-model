"""Cell-local triple-oxygen chemistry for a finite-volume atmosphere.

Rate coefficients use conventional concentration units: ``s-1`` for
photolysis, ``cm3 molecule-1 s-1`` for bimolecular reactions, and ``cm6
molecule-2 s-1`` for termolecular reactions. The evaluator converts cell
inventories to number density, evaluates mass action locally, and converts the
result back to mol yr-1. No Young box-volume factor is reused on the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


AVOGADRO_PER_MOL = 6.02214076e23
BOLTZMANN_J_PER_K = 1.380649e-23
SECONDS_PER_YEAR = 365.25 * 86_400.0


ATMOSPHERIC_OXYGEN_SPECIES = (
    "O",
    "O17",
    "O18",
    "O1D",
    "O17_1D",
    "O18_1D",
    "O2",
    "O17O",
    "O18O",
    "CO2",
    "CO17O",
    "CO18O",
    "O3",
    "OO17O",
    "OO18O",
)


ATMOSPHERIC_ATOM_COUNTS: dict[str, dict[str, float]] = {
    "O": {"16O": 1.0},
    "O17": {"17O": 1.0},
    "O18": {"18O": 1.0},
    "O1D": {"16O": 1.0},
    "O17_1D": {"17O": 1.0},
    "O18_1D": {"18O": 1.0},
    "O2": {"16O": 2.0},
    "O17O": {"16O": 1.0, "17O": 1.0},
    "O18O": {"16O": 1.0, "18O": 1.0},
    "CO2": {"C": 1.0, "16O": 2.0},
    "CO17O": {"C": 1.0, "16O": 1.0, "17O": 1.0},
    "CO18O": {"C": 1.0, "16O": 1.0, "18O": 1.0},
    "O3": {"16O": 3.0},
    "OO17O": {"16O": 2.0, "17O": 1.0},
    "OO18O": {"16O": 2.0, "18O": 1.0},
}


@dataclass(frozen=True)
class ElementaryGridReaction:
    """One atom-balanced local reaction with a scalar or cellwise coefficient."""

    key: str
    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate_coefficient: float | np.ndarray
    source: str

    def __post_init__(self) -> None:
        if not self.key or not self.source:
            raise ValueError("grid reactions require a key and provenance")
        if not self.reactants:
            raise ValueError("local mass-action reactions require reactants")
        for side in (self.reactants, self.products):
            if any(value <= 0 or int(value) != value for value in side.values()):
                raise ValueError("reaction stoichiometry must use positive integers")
        coefficient = np.asarray(self.rate_coefficient, dtype=float)
        if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError("rate coefficients must be finite and non-negative")

    @property
    def kinetic_order(self) -> int:
        return int(sum(self.reactants.values()))

    @property
    def expected_coefficient_units(self) -> str:
        order = self.kinetic_order
        if order == 1:
            return "s-1"
        return f"cm{3 * (order - 1)} molecule-{order - 1} s-1"


def reaction_atom_residual(
    reaction: ElementaryGridReaction,
    atom_counts: Mapping[str, Mapping[str, float]] = ATMOSPHERIC_ATOM_COUNTS,
) -> dict[str, float]:
    """Return products minus reactants for each tracked atom."""

    residual: dict[str, float] = {}
    for sign, side in ((-1.0, reaction.reactants), (1.0, reaction.products)):
        for species, coefficient in side.items():
            if species not in atom_counts:
                raise ValueError(f"missing atom-count record for {species}")
            for atom, count in atom_counts[species].items():
                residual[atom] = residual.get(atom, 0.0) + sign * coefficient * count
    return {atom: value for atom, value in residual.items() if value != 0.0}


def validate_atom_balanced_reactions(
    reactions: tuple[ElementaryGridReaction, ...],
    atom_counts: Mapping[str, Mapping[str, float]] = ATMOSPHERIC_ATOM_COUNTS,
) -> None:
    """Reject any local reaction that creates or destroys a tracked atom."""

    failures = {
        reaction.key: reaction_atom_residual(reaction, atom_counts)
        for reaction in reactions
        if reaction_atom_residual(reaction, atom_counts)
    }
    if failures:
        raise ValueError(f"grid reactions violate atom balance: {failures}")


def local_reaction_tendency_mol_per_year(
    tracer_moles: np.ndarray,
    *,
    species_names: tuple[str, ...],
    air_moles: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    reactions: tuple[ElementaryGridReaction, ...],
    atom_counts: Mapping[str, Mapping[str, float]] = ATMOSPHERIC_ATOM_COUNTS,
    validate_atom_balance: bool = True,
    return_throughput: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate all local reactions independently in every atmospheric cell."""

    species = tuple(species_names)
    index = {name: position for position, name in enumerate(species)}
    if len(index) != len(species):
        raise ValueError("species names must be unique")
    inventory = np.asarray(tracer_moles, dtype=float)
    air = np.asarray(air_moles, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    expected = (len(species), len(air))
    if inventory.shape != expected:
        raise ValueError(f"species inventory must have shape {expected}")
    if pressure.shape != air.shape or temperature.shape != air.shape:
        raise ValueError("pressure and temperature must have one value per cell")
    if (
        not np.all(np.isfinite(inventory))
        or np.any(inventory < 0.0)
        or not np.all(np.isfinite(air))
        or np.any(air <= 0.0)
        or not np.all(np.isfinite(pressure))
        or np.any(pressure <= 0.0)
        or not np.all(np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("cell state must be finite, positive, and non-negative in inventory")
    if validate_atom_balance:
        validate_atom_balanced_reactions(reactions, atom_counts)

    missing = sorted(
        {
            name
            for reaction in reactions
            for name in (*reaction.reactants.keys(), *reaction.products.keys())
            if name not in index
        }
    )
    if missing:
        raise ValueError(f"reaction species are absent from the grid state: {missing}")

    total_number_density_cm3 = pressure / (
        BOLTZMANN_J_PER_K * temperature
    ) / 1.0e6
    cell_volume_cm3 = air * AVOGADRO_PER_MOL / total_number_density_cm3
    species_number_density_cm3 = (
        inventory / air[None, :]
    ) * total_number_density_cm3[None, :]
    tendency = np.zeros_like(inventory)
    throughput = np.zeros_like(inventory)
    for reaction in reactions:
        coefficient = np.asarray(reaction.rate_coefficient, dtype=float)
        try:
            coefficient = np.broadcast_to(coefficient, air.shape)
        except ValueError as error:
            raise ValueError(
                f"rate coefficient for {reaction.key} cannot broadcast to cells"
            ) from error
        event_density_cm3_s = coefficient.copy()
        for name, power in reaction.reactants.items():
            event_density_cm3_s *= species_number_density_cm3[index[name]] ** power
        event_mol_per_year = (
            event_density_cm3_s
            * cell_volume_cm3
            / AVOGADRO_PER_MOL
            * SECONDS_PER_YEAR
        )
        for name, coefficient_stoichiometric in reaction.reactants.items():
            tendency[index[name]] -= coefficient_stoichiometric * event_mol_per_year
            throughput[index[name]] += coefficient_stoichiometric * event_mol_per_year
        for name, coefficient_stoichiometric in reaction.products.items():
            tendency[index[name]] += coefficient_stoichiometric * event_mol_per_year
            throughput[index[name]] += coefficient_stoichiometric * event_mol_per_year
    if return_throughput:
        return tendency, throughput
    return tendency


def local_reaction_event_rate_mol_per_year(
    tracer_moles: np.ndarray,
    *,
    species_names: tuple[str, ...],
    air_moles: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    reaction: ElementaryGridReaction,
) -> np.ndarray:
    """Evaluate one reaction's gross event rate in each atmospheric cell.

    This differs from a species tendency when a reactant is regenerated, as in
    catalytic O(1D)-CO2 quenching. It is therefore the appropriate quantity for
    reaction-fate and exposure ledgers.
    """

    species = tuple(species_names)
    index = {name: position for position, name in enumerate(species)}
    if len(index) != len(species):
        raise ValueError("species names must be unique")
    inventory = np.asarray(tracer_moles, dtype=float)
    air = np.asarray(air_moles, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    expected = (len(species), len(air))
    if inventory.shape != expected:
        raise ValueError(f"species inventory must have shape {expected}")
    if pressure.shape != air.shape or temperature.shape != air.shape:
        raise ValueError("pressure and temperature must have one value per cell")
    if (
        not np.all(np.isfinite(inventory))
        or np.any(inventory < 0.0)
        or not np.all(np.isfinite(air))
        or np.any(air <= 0.0)
        or not np.all(np.isfinite(pressure))
        or np.any(pressure <= 0.0)
        or not np.all(np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("cell state must be finite, positive, and non-negative in inventory")
    missing = sorted(set(reaction.reactants).difference(index))
    if missing:
        raise ValueError(f"reaction species are absent from the grid state: {missing}")

    total_number_density_cm3 = pressure / (
        BOLTZMANN_J_PER_K * temperature
    ) / 1.0e6
    cell_volume_cm3 = air * AVOGADRO_PER_MOL / total_number_density_cm3
    species_number_density_cm3 = (
        inventory / air[None, :]
    ) * total_number_density_cm3[None, :]
    try:
        event_density_cm3_s = np.broadcast_to(
            np.asarray(reaction.rate_coefficient, dtype=float), air.shape
        ).copy()
    except ValueError as error:
        raise ValueError(
            f"rate coefficient for {reaction.key} cannot broadcast to cells"
        ) from error
    for name, power in reaction.reactants.items():
        event_density_cm3_s *= species_number_density_cm3[index[name]] ** power
    return (
        event_density_cm3_s
        * cell_volume_cm3
        / AVOGADRO_PER_MOL
        * SECONDS_PER_YEAR
    )



def bind_local_reaction_operator(
    *,
    species_names: tuple[str, ...],
    air_moles: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    reactions: tuple[ElementaryGridReaction, ...],
    atom_counts: Mapping[str, Mapping[str, float]] = ATMOSPHERIC_ATOM_COUNTS,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Bind static grid fields into a chemistry callback for transport coupling."""
    species = tuple(species_names)
    air = np.asarray(air_moles, dtype=float).copy()
    pressure = np.asarray(pressure_pa, dtype=float).copy()
    temperature = np.asarray(temperature_k, dtype=float).copy()
    if air.ndim != 1 or pressure.shape != air.shape or temperature.shape != air.shape:
        raise ValueError("bound chemistry fields must be one-dimensional and cell aligned")
    validate_atom_balanced_reactions(reactions, atom_counts)
    reaction_species = {
        name
        for reaction in reactions
        for name in (*reaction.reactants.keys(), *reaction.products.keys())
    }
    missing = sorted(reaction_species.difference(species))
    if missing:
        raise ValueError(f"reaction species are absent from the bound grid state: {missing}")

    def chemistry(_time_years: float, tracer_moles: np.ndarray) -> np.ndarray:
        return local_reaction_tendency_mol_per_year(
            tracer_moles,
            species_names=species,
            air_moles=air,
            pressure_pa=pressure,
            temperature_k=temperature,
            reactions=reactions,
            atom_counts=atom_counts,
            validate_atom_balance=False,
        )

    return chemistry


def bind_local_reaction_throughput_operator(
    *,
    species_names: tuple[str, ...],
    air_moles: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    reactions: tuple[ElementaryGridReaction, ...],
    atom_counts: Mapping[str, Mapping[str, float]] = ATMOSPHERIC_ATOM_COUNTS,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Bind the gross local source-plus-sink throughput for residual scaling."""

    species = tuple(species_names)
    air = np.asarray(air_moles, dtype=float).copy()
    pressure = np.asarray(pressure_pa, dtype=float).copy()
    temperature = np.asarray(temperature_k, dtype=float).copy()
    if air.ndim != 1 or pressure.shape != air.shape or temperature.shape != air.shape:
        raise ValueError("bound chemistry fields must be one-dimensional and cell aligned")
    validate_atom_balanced_reactions(reactions, atom_counts)

    def throughput(_time_years: float, tracer_moles: np.ndarray) -> np.ndarray:
        _, gross = local_reaction_tendency_mol_per_year(
            tracer_moles,
            species_names=species,
            air_moles=air,
            pressure_pa=pressure,
            temperature_k=temperature,
            reactions=reactions,
            atom_counts=atom_counts,
            validate_atom_balance=False,
            return_throughput=True,
        )
        return gross

    return throughput
