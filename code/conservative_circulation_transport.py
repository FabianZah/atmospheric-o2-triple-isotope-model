"""Mass-conserving directed transport on an atmospheric cell network.

Eddy diffusion and residual circulation are physically different operators.
The former exchanges air bidirectionally across an interface. The latter moves
air around a closed overturning circulation. This module implements the
directed part without prescribing a grid, circulation strength, or latitude
boundary.

For fixed cell air inventories, every cell must have equal total inflow and
outflow. The constructor rejects a divergent air-flux field instead of silently
balancing it. A uniform tracer mixing ratio is therefore stationary, and every
tracer inventory is conserved to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from conservative_column_transport import AtmosphericLayer


class MatrixTransportOperator(Protocol):
    """Minimal interface shared by conservative transport operators."""

    def transport_matrix_per_year(self) -> np.ndarray: ...


@dataclass(frozen=True)
class DirectedAirFlux:
    """A gross one-way air flux from one atmospheric cell to another."""

    source_layer: str
    target_layer: str
    air_flux_mol_per_year: float
    source: str

    def __post_init__(self) -> None:
        if self.source_layer == self.target_layer:
            raise ValueError("a directed air flux must connect two different cells")
        if self.air_flux_mol_per_year < 0.0:
            raise ValueError("directed air flux must be non-negative")
        if not self.source:
            raise ValueError("directed air flux requires a provenance source")


@dataclass(frozen=True)
class ConservativeCirculationNetwork:
    """Directed transport among fixed-inventory atmospheric cells."""

    layers: tuple[AtmosphericLayer, ...]
    fluxes: tuple[DirectedAirFlux, ...]
    stationarity_relative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        names = [layer.name for layer in self.layers]
        if not names:
            raise ValueError("circulation network requires at least one cell")
        if len(names) != len(set(names)):
            raise ValueError("cell names must be unique")
        if self.stationarity_relative_tolerance < 0.0:
            raise ValueError("stationarity tolerance must be non-negative")

        known = set(names)
        for flux in self.fluxes:
            if flux.source_layer not in known or flux.target_layer not in known:
                raise ValueError("all directed-flux endpoints must name an existing cell")

        residual = self.air_mass_tendency_mol_per_year()
        throughput = max(
            sum(flux.air_flux_mol_per_year for flux in self.fluxes),
            1.0,
        )
        relative_residual = float(np.max(np.abs(residual))) / throughput
        if relative_residual > self.stationarity_relative_tolerance:
            by_cell = {
                layer.name: float(residual[index])
                for index, layer in enumerate(self.layers)
                if residual[index] != 0.0
            }
            raise ValueError(
                "directed air fluxes are not stationary for fixed cell inventories; "
                f"relative residual={relative_residual:.6g}, by_cell={by_cell}"
            )

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    @property
    def air_moles(self) -> np.ndarray:
        return np.asarray([layer.air_moles for layer in self.layers], dtype=float)

    def air_mass_tendency_mol_per_year(self) -> np.ndarray:
        """Return inflow minus outflow for each atmospheric cell."""

        index = {layer.name: position for position, layer in enumerate(self.layers)}
        tendency = np.zeros(len(self.layers), dtype=float)
        for flux in self.fluxes:
            amount = flux.air_flux_mol_per_year
            tendency[index[flux.source_layer]] -= amount
            tendency[index[flux.target_layer]] += amount
        return tendency

    def transport_matrix_per_year(self) -> np.ndarray:
        """Return matrix A such that dN/dt = A @ N for tracer inventories."""

        index = {layer.name: position for position, layer in enumerate(self.layers)}
        air = self.air_moles
        matrix = np.zeros((len(self.layers), len(self.layers)), dtype=float)
        for flux in self.fluxes:
            source = index[flux.source_layer]
            target = index[flux.target_layer]
            rate = flux.air_flux_mol_per_year / air[source]
            matrix[source, source] -= rate
            matrix[target, source] += rate
        return matrix

    def derivative(self, tracer_moles: np.ndarray) -> np.ndarray:
        tracer = np.asarray(tracer_moles, dtype=float)
        if tracer.shape[-1] != len(self.layers):
            raise ValueError("last tracer dimension must equal the number of cells")
        return tracer @ self.transport_matrix_per_year().T

    def mixing_ratios(self, tracer_moles: np.ndarray) -> np.ndarray:
        return np.asarray(tracer_moles, dtype=float) / self.air_moles

    def conservation_residual(self, tracer_moles: np.ndarray) -> np.ndarray:
        """Return summed tracer tendency; zero is exact network conservation."""

        return np.sum(self.derivative(tracer_moles), axis=-1)

    def flux_rate_constants_per_year(self) -> tuple[dict[str, float | str], ...]:
        index = {layer.name: position for position, layer in enumerate(self.layers)}
        air = self.air_moles
        return tuple(
            {
                "source_layer": flux.source_layer,
                "target_layer": flux.target_layer,
                "source_to_target_per_year": (
                    flux.air_flux_mol_per_year / air[index[flux.source_layer]]
                ),
                "air_flux_mol_per_year": flux.air_flux_mol_per_year,
                "source": flux.source,
            }
            for flux in self.fluxes
        )


def combined_transport_matrix_per_year(
    *operators: MatrixTransportOperator,
) -> np.ndarray:
    """Add conservative transport operators defined on the same ordered cells."""

    if not operators:
        raise ValueError("at least one transport operator is required")
    matrices = [
        np.asarray(operator.transport_matrix_per_year(), dtype=float)
        for operator in operators
    ]
    shape = matrices[0].shape
    if shape[0] != shape[1]:
        raise ValueError("transport matrices must be square")
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("all transport matrices must have the same shape")
    return np.sum(matrices, axis=0)


def latitude_vertical_cell_name(latitude_index: int, vertical_index: int) -> str:
    return f"latitude_{latitude_index:02d}_level_{vertical_index:02d}"


def circulation_network_from_streamfunction(
    air_moles: np.ndarray,
    streamfunction_mol_per_year: np.ndarray,
    source: str,
    boundary_relative_tolerance: float = 1.0e-12,
) -> ConservativeCirculationNetwork:
    """Convert a cell-corner mass streamfunction to a closed flux network.

    Arrays use ``(latitude, vertical)`` order. ``air_moles`` is cell-centred
    with shape ``(n_latitude, n_vertical)``. The streamfunction is defined on
    cell corners and therefore has shape ``(n_latitude + 1, n_vertical + 1)``.
    A constant streamfunction around the outer boundary enforces zero normal
    air flux.

    Positive meridional flux points from lower to higher latitude index.
    Positive vertical flux points from lower to higher vertical index.
    """

    if not source:
        raise ValueError("streamfunction transport requires a provenance source")
    if boundary_relative_tolerance < 0.0:
        raise ValueError("boundary tolerance must be non-negative")

    air = np.asarray(air_moles, dtype=float)
    streamfunction = np.asarray(streamfunction_mol_per_year, dtype=float)
    if air.ndim != 2 or min(air.shape) < 1:
        raise ValueError("air inventories must be a non-empty 2-D array")
    if streamfunction.shape != (air.shape[0] + 1, air.shape[1] + 1):
        raise ValueError(
            "streamfunction must contain one more edge than air_moles in each dimension"
        )
    if not np.all(np.isfinite(air)) or not np.all(air > 0.0):
        raise ValueError("all cell air inventories must be finite and positive")
    if not np.all(np.isfinite(streamfunction)):
        raise ValueError("streamfunction values must be finite")

    perimeter = np.concatenate(
        (
            streamfunction[0, :],
            streamfunction[-1, :],
            streamfunction[1:-1, 0],
            streamfunction[1:-1, -1],
        )
    )
    perimeter_scale = max(float(np.max(np.abs(streamfunction))), 1.0)
    if float(np.ptp(perimeter)) > boundary_relative_tolerance * perimeter_scale:
        raise ValueError(
            "streamfunction must be constant around the outer boundary for a closed domain"
        )

    n_latitude, n_vertical = air.shape
    layers = tuple(
        AtmosphericLayer(
            latitude_vertical_cell_name(latitude_index, vertical_index),
            float(air[latitude_index, vertical_index]),
        )
        for latitude_index in range(n_latitude)
        for vertical_index in range(n_vertical)
    )

    fluxes: list[DirectedAirFlux] = []

    def append_signed_flux(
        first_layer: str,
        second_layer: str,
        signed_first_to_second: float,
    ) -> None:
        if signed_first_to_second > 0.0:
            source_layer, target_layer = first_layer, second_layer
            amount = signed_first_to_second
        elif signed_first_to_second < 0.0:
            source_layer, target_layer = second_layer, first_layer
            amount = -signed_first_to_second
        else:
            return
        fluxes.append(
            DirectedAirFlux(
                source_layer=source_layer,
                target_layer=target_layer,
                air_flux_mol_per_year=float(amount),
                source=source,
            )
        )

    # Meridional fluxes across internal latitude edges.
    for latitude_edge in range(1, n_latitude):
        for vertical_index in range(n_vertical):
            south = latitude_vertical_cell_name(
                latitude_edge - 1,
                vertical_index,
            )
            north = latitude_vertical_cell_name(
                latitude_edge,
                vertical_index,
            )
            south_to_north = (
                streamfunction[latitude_edge, vertical_index + 1]
                - streamfunction[latitude_edge, vertical_index]
            )
            append_signed_flux(south, north, float(south_to_north))

    # Vertical fluxes across internal level edges.
    for latitude_index in range(n_latitude):
        for vertical_edge in range(1, n_vertical):
            lower = latitude_vertical_cell_name(
                latitude_index,
                vertical_edge - 1,
            )
            upper = latitude_vertical_cell_name(
                latitude_index,
                vertical_edge,
            )
            lower_to_upper = (
                streamfunction[latitude_index, vertical_edge]
                - streamfunction[latitude_index + 1, vertical_edge]
            )
            append_signed_flux(lower, upper, float(lower_to_upper))

    return ConservativeCirculationNetwork(
        layers=layers,
        fluxes=tuple(fluxes),
    )


def circulation_network_with_lower_reservoir_row(
    atmospheric_air_moles: np.ndarray,
    lower_reservoir_air_moles: np.ndarray,
    streamfunction_mol_per_year: np.ndarray,
    source: str,
    closed_boundary_value_mol_per_year: float = 0.0,
    boundary_relative_tolerance: float = 1.0e-12,
) -> ConservativeCirculationNetwork:
    """Close an open lower boundary with latitude-resolved reservoir cells.

    ``streamfunction_mol_per_year`` describes the atmospheric cells and is open
    along its first vertical edge. Its two polar edges and upper edge must equal
    ``closed_boundary_value_mol_per_year``. One lower-atmosphere reservoir cell
    is prepended in each latitude band, and a constant streamfunction boundary
    is placed below that row. Finite differences then represent upwelling,
    descent, and meridional return flow as one closed circulation.

    No exchange magnitude is fitted or balanced by this function.
    """

    atmospheric_air = np.asarray(atmospheric_air_moles, dtype=float)
    lower_air = np.asarray(lower_reservoir_air_moles, dtype=float)
    streamfunction = np.asarray(streamfunction_mol_per_year, dtype=float)
    if atmospheric_air.ndim != 2:
        raise ValueError("atmospheric air inventories must be a 2-D array")
    if lower_air.shape != (atmospheric_air.shape[0],):
        raise ValueError("lower reservoir requires one air inventory per latitude band")
    if not np.all(np.isfinite(lower_air)) or not np.all(lower_air > 0.0):
        raise ValueError("lower-reservoir air inventories must be finite and positive")
    if streamfunction.shape != (
        atmospheric_air.shape[0] + 1,
        atmospheric_air.shape[1] + 1,
    ):
        raise ValueError(
            "streamfunction must contain one more edge than atmospheric cells "
            "in each dimension"
        )

    scale = max(float(np.max(np.abs(streamfunction))), 1.0)
    tolerance = boundary_relative_tolerance * scale
    closed_value = float(closed_boundary_value_mol_per_year)
    if (
        np.max(np.abs(streamfunction[0, :] - closed_value)) > tolerance
        or np.max(np.abs(streamfunction[-1, :] - closed_value)) > tolerance
        or np.max(np.abs(streamfunction[:, -1] - closed_value)) > tolerance
    ):
        raise ValueError(
            "polar and upper streamfunction boundaries must be closed before "
            "adding a lower reservoir row"
        )

    expanded_air = np.concatenate(
        (lower_air[:, None], atmospheric_air),
        axis=1,
    )
    lower_boundary = np.full(
        (streamfunction.shape[0], 1),
        closed_value,
        dtype=float,
    )
    expanded_streamfunction = np.concatenate(
        (lower_boundary, streamfunction),
        axis=1,
    )
    return circulation_network_from_streamfunction(
        expanded_air,
        expanded_streamfunction,
        source=source,
        boundary_relative_tolerance=boundary_relative_tolerance,
    )
