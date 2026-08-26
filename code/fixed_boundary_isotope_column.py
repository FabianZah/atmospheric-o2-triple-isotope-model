"""Positive fixed-boundary solver for an isotope-resolved atmospheric column.

The fast atmospheric chemistry is solved in log inventory, while selected
reservoir entries remain prescribed. Their compensating flux is reported
explicitly instead of being discarded. This lets a vertically resolved fast
column provide source terms to a slower global O2/CO2 budget without pretending
that the column itself predicts the bulk atmospheric reservoirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix

from gridded_isotope_transport import (
    GriddedSpeciesSystem,
    LocalChemistry,
    SpeciesTendencyComponents,
)


@dataclass(frozen=True)
class FixedBoundaryColumnResult:
    """Final state and numerical diagnostics from one column integration."""

    inventory_moles: np.ndarray
    elapsed_years: float
    converged_to_steady: bool
    maximum_free_log_tendency_per_year: float
    maximum_free_throughput_scaled_residual: float
    solver_success: bool
    solver_message: str
    function_evaluations: int
    jacobian_evaluations: int


@dataclass(frozen=True)
class FixedBoundaryIsotopeColumn:
    """Coupled transport/chemistry system with explicit prescribed entries."""

    species_system: GriddedSpeciesSystem
    local_chemistry: LocalChemistry | None
    fixed_mask: np.ndarray
    prescribed_inventory_moles: np.ndarray
    source: str
    local_chemistry_throughput: LocalChemistry | None = None

    def __post_init__(self) -> None:
        prescribed = self.species_system.validate_inventory(
            self.prescribed_inventory_moles
        )
        mask = np.asarray(self.fixed_mask, dtype=bool)
        if mask.shape != prescribed.shape:
            raise ValueError("fixed mask must match the species inventory shape")
        if np.any(prescribed <= 0.0):
            raise ValueError("prescribed and initial inventories must be positive")
        if not np.any(mask) or np.all(mask):
            raise ValueError("column must contain both fixed and freely solved entries")
        if not self.source:
            raise ValueError("fixed-boundary column requires provenance")

    @property
    def free_flat_indices(self) -> np.ndarray:
        return np.flatnonzero(~np.asarray(self.fixed_mask, dtype=bool).ravel())

    @property
    def free_count(self) -> int:
        return len(self.free_flat_indices)

    def log_free_state(self, inventory_moles: np.ndarray | None = None) -> np.ndarray:
        inventory = (
            np.asarray(self.prescribed_inventory_moles, dtype=float)
            if inventory_moles is None
            else self.species_system.validate_inventory(inventory_moles)
        )
        free = inventory.ravel()[self.free_flat_indices]
        if np.any(free <= 0.0):
            raise ValueError("free inventories must be positive before log transformation")
        return np.log(free)

    def inventory_from_log_free(self, log_free_inventory: np.ndarray) -> np.ndarray:
        log_free = np.asarray(log_free_inventory, dtype=float)
        if log_free.shape != (self.free_count,) or not np.all(np.isfinite(log_free)):
            raise ValueError("log free state must contain one finite value per free entry")
        inventory = np.asarray(self.prescribed_inventory_moles, dtype=float).copy()
        inventory.ravel()[self.free_flat_indices] = np.exp(log_free)
        if not np.all(np.isfinite(inventory)):
            raise FloatingPointError("log-state transformation overflowed")
        return inventory

    def internal_components(
        self,
        time_years: float,
        inventory_moles: np.ndarray,
    ) -> SpeciesTendencyComponents:
        return self.species_system.tendency_components_mol_per_year(
            time_years,
            inventory_moles,
            local_chemistry=self.local_chemistry,
        )

    def free_log_derivative(
        self,
        time_years: float,
        log_free_inventory: np.ndarray,
    ) -> np.ndarray:
        inventory = self.inventory_from_log_free(log_free_inventory)
        tendency = self.internal_components(
            time_years, inventory
        ).total_mol_per_year.ravel()[self.free_flat_indices]
        abundance = inventory.ravel()[self.free_flat_indices]
        return tendency / abundance

    def required_reservoir_flux_mol_per_year(
        self,
        time_years: float,
        inventory_moles: np.ndarray,
    ) -> np.ndarray:
        """Return the external flux required to hold every prescribed entry."""

        internal = self.internal_components(
            time_years, inventory_moles
        ).total_mol_per_year
        closure = np.zeros_like(internal)
        mask = np.asarray(self.fixed_mask, dtype=bool)
        closure[mask] = -internal[mask]
        return closure

    def internal_throughput_mol_per_year(
        self,
        time_years: float,
        inventory_moles: np.ndarray,
    ) -> np.ndarray:
        """Return gross transport plus chemistry source-and-sink throughput."""

        inventory = self.species_system.validate_inventory(inventory_moles)
        transport_matrix = np.asarray(
            self.species_system.inventory_transport_matrix_per_year, dtype=float
        )
        throughput = inventory @ np.abs(transport_matrix).T
        if self.local_chemistry_throughput is not None:
            chemistry = np.asarray(
                self.local_chemistry_throughput(float(time_years), inventory),
                dtype=float,
            )
            if chemistry.shape != inventory.shape or np.any(chemistry < 0.0):
                raise ValueError("chemistry throughput must be non-negative and cell aligned")
            throughput = throughput + chemistry
        return throughput

    def maximum_free_throughput_scaled_residual(
        self,
        time_years: float,
        inventory_moles: np.ndarray,
    ) -> float:
        internal = self.internal_components(
            time_years, inventory_moles
        ).total_mol_per_year.ravel()[self.free_flat_indices]
        throughput = self.internal_throughput_mol_per_year(
            time_years, inventory_moles
        ).ravel()[self.free_flat_indices]
        scale = np.maximum(throughput, np.finfo(float).tiny)
        return float(np.max(np.abs(internal) / scale))

    def reservoir_atom_flux_mol_per_year(
        self,
        time_years: float,
        inventory_moles: np.ndarray,
        atom_counts: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        closure = self.required_reservoir_flux_mol_per_year(
            time_years, inventory_moles
        )
        return self.species_system.atom_tendency_mol_per_year(closure, atom_counts)

    def jacobian_sparsity(self) -> csr_matrix:
        """Conservative upper bound for the free log-state Jacobian pattern."""

        species_count = self.species_system.species_count
        cell_count = self.species_system.cell_count
        free_flat = self.free_flat_indices
        free_position = {int(flat): index for index, flat in enumerate(free_flat)}
        pattern = lil_matrix((self.free_count, self.free_count), dtype=np.int8)
        transport = np.asarray(
            self.species_system.inventory_transport_matrix_per_year, dtype=float
        )
        for row_position, flat_row in enumerate(free_flat):
            species_index, cell_index = divmod(int(flat_row), cell_count)
            # Every local reaction may couple any pair of species in one cell.
            for dependency_species in range(species_count):
                flat_column = dependency_species * cell_count + cell_index
                column_position = free_position.get(flat_column)
                if column_position is not None:
                    pattern[row_position, column_position] = 1
            # Transport couples one species between cells wherever A is nonzero.
            for dependency_cell in np.flatnonzero(transport[cell_index]):
                flat_column = species_index * cell_count + int(dependency_cell)
                column_position = free_position.get(flat_column)
                if column_position is not None:
                    pattern[row_position, column_position] = 1
        return pattern.tocsr()


def solve_fixed_boundary_column(
    column: FixedBoundaryIsotopeColumn,
    *,
    maximum_years: float,
    steady_log_tendency_tolerance_per_year: float = 1.0e-8,
    steady_throughput_relative_tolerance: float = 1.0e-8,
    relative_tolerance: float = 1.0e-7,
    absolute_log_tolerance: float = 1.0e-9,
    maximum_step_years: float = np.inf,
) -> FixedBoundaryColumnResult:
    """Integrate the positive free state and report, but do not force, steady state."""

    if not np.isfinite(maximum_years) or maximum_years <= 0.0:
        raise ValueError("maximum_years must be finite and positive")
    if steady_log_tendency_tolerance_per_year <= 0.0:
        raise ValueError("steady tolerance must be positive")
    if steady_throughput_relative_tolerance <= 0.0:
        raise ValueError("throughput-relative steady tolerance must be positive")
    if relative_tolerance <= 0.0 or absolute_log_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if maximum_step_years <= 0.0:
        raise ValueError("maximum step must be positive")

    initial = column.log_free_state()
    solution = solve_ivp(
        column.free_log_derivative,
        (0.0, float(maximum_years)),
        initial,
        method="BDF",
        jac_sparsity=column.jacobian_sparsity(),
        rtol=relative_tolerance,
        atol=absolute_log_tolerance,
        max_step=float(maximum_step_years),
    )
    solved_states = np.asarray(solution.y, dtype=float)
    if solved_states.ndim != 2 or solved_states.shape[1] == 0:
        final_log = initial
    else:
        final_log = solved_states[:, -1]
    reached_time = float(solution.t[-1]) if len(solution.t) else 0.0
    final_inventory = column.inventory_from_log_free(final_log)
    maximum_tendency = float(
        np.max(np.abs(column.free_log_derivative(reached_time, final_log)))
    )
    maximum_scaled_residual = column.maximum_free_throughput_scaled_residual(
        reached_time, final_inventory
    )
    return FixedBoundaryColumnResult(
        inventory_moles=final_inventory,
        elapsed_years=reached_time,
        converged_to_steady=(
            bool(solution.success)
            and maximum_scaled_residual <= steady_throughput_relative_tolerance
        ),
        maximum_free_log_tendency_per_year=maximum_tendency,
        maximum_free_throughput_scaled_residual=maximum_scaled_residual,
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
        function_evaluations=int(solution.nfev),
        jacobian_evaluations=int(solution.njev),
    )
