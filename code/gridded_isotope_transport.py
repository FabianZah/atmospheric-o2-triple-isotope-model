"""Conservative multispecies transport for the latitude-pressure model.

This module is the chemistry/transport coupling boundary. Atmospheric
transport acts on species inventories with one shared conservative operator;
cell-local chemistry and external boundary fluxes are supplied separately.
That separation prevents a chemistry adjustment from silently changing the
circulation and makes elemental and isotope budgets directly auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


LocalChemistry = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SpeciesTendencyComponents:
    """Named terms in one coupled species-inventory derivative."""

    transport_mol_per_year: np.ndarray
    chemistry_mol_per_year: np.ndarray
    boundary_mol_per_year: np.ndarray

    @property
    def total_mol_per_year(self) -> np.ndarray:
        return (
            np.asarray(self.transport_mol_per_year, dtype=float)
            + np.asarray(self.chemistry_mol_per_year, dtype=float)
            + np.asarray(self.boundary_mol_per_year, dtype=float)
        )


@dataclass(frozen=True)
class GriddedSpeciesSystem:
    """One conservative cell network shared by an arbitrary species list.

    ``inventory_transport_matrix_per_year`` follows ``dN/dt = A @ N`` for one
    species inventory. Its columns must sum to zero, which guarantees that
    transport conserves every species independently.
    """

    species_names: tuple[str, ...]
    air_moles: np.ndarray
    inventory_transport_matrix_per_year: np.ndarray
    source: str
    conservation_relative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        species = tuple(self.species_names)
        air = np.asarray(self.air_moles, dtype=float)
        matrix = np.asarray(self.inventory_transport_matrix_per_year, dtype=float)
        if not species or len(species) != len(set(species)):
            raise ValueError("species names must be non-empty and unique")
        if air.ndim != 1 or not np.all(np.isfinite(air)) or np.any(air <= 0.0):
            raise ValueError("cell air inventories must be finite and positive")
        if matrix.shape != (len(air), len(air)) or not np.all(np.isfinite(matrix)):
            raise ValueError("transport matrix must be finite and square on the cell grid")
        if self.conservation_relative_tolerance < 0.0:
            raise ValueError("conservation tolerance must be non-negative")
        matrix_scale = max(float(np.max(np.sum(np.abs(matrix), axis=0))), 1.0)
        residual = float(np.max(np.abs(np.sum(matrix, axis=0))))
        if residual > self.conservation_relative_tolerance * matrix_scale:
            raise ValueError(
                "transport matrix does not conserve inventory; "
                f"relative column residual={residual / matrix_scale:.6g}"
            )
        air_tendency = matrix @ air
        air_scale = max(float(np.max(np.abs(matrix) @ air)), 1.0)
        air_residual = float(np.max(np.abs(air_tendency)))
        if air_residual > self.conservation_relative_tolerance * air_scale:
            raise ValueError(
                "transport matrix does not preserve a uniform mixing ratio; "
                f"relative air-state residual={air_residual / air_scale:.6g}"
            )
        if not self.source:
            raise ValueError("gridded species transport requires provenance")

    @property
    def cell_count(self) -> int:
        return len(np.asarray(self.air_moles))

    @property
    def species_count(self) -> int:
        return len(self.species_names)

    def validate_inventory(self, tracer_moles: np.ndarray) -> np.ndarray:
        inventory = np.asarray(tracer_moles, dtype=float)
        expected = (self.species_count, self.cell_count)
        if inventory.shape != expected:
            raise ValueError(f"species inventory must have shape {expected}")
        if not np.all(np.isfinite(inventory)):
            raise ValueError("species inventory must be finite")
        return inventory

    def mixing_ratios(self, tracer_moles: np.ndarray) -> np.ndarray:
        inventory = self.validate_inventory(tracer_moles)
        return inventory / np.asarray(self.air_moles, dtype=float)[None, :]

    def transport_derivative_mol_per_year(self, tracer_moles: np.ndarray) -> np.ndarray:
        inventory = self.validate_inventory(tracer_moles)
        matrix = np.asarray(self.inventory_transport_matrix_per_year, dtype=float)
        return inventory @ matrix.T

    def derivative_mol_per_year(
        self,
        time_years: float,
        tracer_moles: np.ndarray,
        *,
        local_chemistry: LocalChemistry | None = None,
        boundary_flux_mol_per_year: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return transport + local chemistry + explicit boundary fluxes."""

        return self.tendency_components_mol_per_year(
            time_years,
            tracer_moles,
            local_chemistry=local_chemistry,
            boundary_flux_mol_per_year=boundary_flux_mol_per_year,
        ).total_mol_per_year

    def tendency_components_mol_per_year(
        self,
        time_years: float,
        tracer_moles: np.ndarray,
        *,
        local_chemistry: LocalChemistry | None = None,
        boundary_flux_mol_per_year: np.ndarray | None = None,
    ) -> SpeciesTendencyComponents:
        """Return auditable transport, chemistry, and boundary terms."""

        inventory = self.validate_inventory(tracer_moles)
        transport = self.transport_derivative_mol_per_year(inventory)
        chemistry = np.zeros_like(inventory)
        if local_chemistry is not None:
            chemistry = np.asarray(local_chemistry(float(time_years), inventory), dtype=float)
            if chemistry.shape != inventory.shape or not np.all(np.isfinite(chemistry)):
                raise ValueError("local chemistry must return one finite tendency per species and cell")
        boundary = np.zeros_like(inventory)
        if boundary_flux_mol_per_year is not None:
            boundary = np.asarray(boundary_flux_mol_per_year, dtype=float)
            if boundary.shape != inventory.shape or not np.all(np.isfinite(boundary)):
                raise ValueError("boundary flux must match the species inventory shape")
        return SpeciesTendencyComponents(
            transport_mol_per_year=transport,
            chemistry_mol_per_year=chemistry,
            boundary_mol_per_year=boundary,
        )

    def species_conservation_residual_mol_per_year(
        self,
        tracer_moles: np.ndarray,
    ) -> np.ndarray:
        """Return global transport tendency for each species; all values are zero."""

        return np.sum(self.transport_derivative_mol_per_year(tracer_moles), axis=1)

    def atom_inventory_moles(
        self,
        tracer_moles: np.ndarray,
        atom_counts: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        """Aggregate named atoms or isotope classes from molecular inventories."""

        inventory = self.validate_inventory(tracer_moles)
        totals: dict[str, float] = {}
        for species_index, species in enumerate(self.species_names):
            if species not in atom_counts:
                raise ValueError(f"missing atom-count record for {species}")
            species_total = float(np.sum(inventory[species_index]))
            for atom, count in atom_counts[species].items():
                if count < 0.0:
                    raise ValueError("atom counts must be non-negative")
                totals[atom] = totals.get(atom, 0.0) + float(count) * species_total
        return totals

    def atom_tendency_mol_per_year(
        self,
        species_tendency_mol_per_year: np.ndarray,
        atom_counts: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        """Aggregate a species tendency into global elemental/isotope budgets."""

        tendency = self.validate_inventory(species_tendency_mol_per_year)
        totals: dict[str, float] = {}
        for species_index, species in enumerate(self.species_names):
            if species not in atom_counts:
                raise ValueError(f"missing atom-count record for {species}")
            species_total = float(np.sum(tendency[species_index]))
            for atom, count in atom_counts[species].items():
                if count < 0.0:
                    raise ValueError("atom counts must be non-negative")
                totals[atom] = totals.get(atom, 0.0) + float(count) * species_total
        return totals

    def component_atom_budget_mol_per_year(
        self,
        components: SpeciesTendencyComponents,
        atom_counts: Mapping[str, Mapping[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Report internal and externally imposed atom tendencies separately."""

        return {
            "transport": self.atom_tendency_mol_per_year(
                components.transport_mol_per_year, atom_counts
            ),
            "chemistry": self.atom_tendency_mol_per_year(
                components.chemistry_mol_per_year, atom_counts
            ),
            "boundary": self.atom_tendency_mol_per_year(
                components.boundary_mol_per_year, atom_counts
            ),
            "total": self.atom_tendency_mol_per_year(
                components.total_mol_per_year, atom_counts
            ),
        }


def gridded_species_system_from_transport(
    transport: object,
    species_names: tuple[str, ...],
    *,
    source: str,
) -> GriddedSpeciesSystem:
    """Adapt a validated passive transport grid to multispecies transport."""

    air = np.asarray(transport.air_moles, dtype=float).ravel()
    matrix = np.asarray(transport.inventory_transport_matrix_per_year(), dtype=float)
    return GriddedSpeciesSystem(
        species_names=species_names,
        air_moles=air,
        inventory_transport_matrix_per_year=matrix,
        source=source,
    )
