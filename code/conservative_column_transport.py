"""Mass-conserving transport operator for a layered atmosphere.

The operator transports tracer inventories, in moles, between fixed atmospheric
air-mass layers. Each interface is represented by one gross bidirectional air
exchange flux. Using the same gross flux in both directions preserves every
tracer exactly and leaves the prescribed layer air inventories stationary.

This module contains no photochemistry and no fitted layer fractions. It is the
transport foundation for the explicit multi-layer bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from young_model_inventory import PARAMETERS


@dataclass(frozen=True)
class AtmosphericLayer:
    name: str
    air_moles: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must not be empty")
        if self.air_moles <= 0.0:
            raise ValueError("layer air inventory must be positive")


@dataclass(frozen=True)
class ExchangeInterface:
    first_layer: str
    second_layer: str
    gross_air_flux_mol_per_year: float
    source: str

    def __post_init__(self) -> None:
        if self.first_layer == self.second_layer:
            raise ValueError("an exchange interface must connect two different layers")
        if self.gross_air_flux_mol_per_year < 0.0:
            raise ValueError("gross air exchange flux must be non-negative")
        if not self.source:
            raise ValueError("transport interface requires a provenance source")


@dataclass(frozen=True)
class ConservativeColumn:
    layers: tuple[AtmosphericLayer, ...]
    interfaces: tuple[ExchangeInterface, ...]

    def __post_init__(self) -> None:
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("layer names must be unique")
        known = set(names)
        for interface in self.interfaces:
            if interface.first_layer not in known or interface.second_layer not in known:
                raise ValueError("all interface endpoints must name an existing layer")

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    @property
    def air_moles(self) -> np.ndarray:
        return np.asarray([layer.air_moles for layer in self.layers], dtype=float)

    def transport_matrix_per_year(self) -> np.ndarray:
        """Return matrix A such that dN/dt = A @ N for tracer inventories."""

        index = {layer.name: i for i, layer in enumerate(self.layers)}
        air = self.air_moles
        matrix = np.zeros((len(self.layers), len(self.layers)), dtype=float)
        for interface in self.interfaces:
            first = index[interface.first_layer]
            second = index[interface.second_layer]
            flux = interface.gross_air_flux_mol_per_year
            first_to_second = flux / air[first]
            second_to_first = flux / air[second]
            matrix[first, first] -= first_to_second
            matrix[second, first] += first_to_second
            matrix[second, second] -= second_to_first
            matrix[first, second] += second_to_first
        return matrix

    def derivative(self, tracer_moles: np.ndarray) -> np.ndarray:
        tracer = np.asarray(tracer_moles, dtype=float)
        if tracer.shape[-1] != len(self.layers):
            raise ValueError("last tracer dimension must equal the number of layers")
        return tracer @ self.transport_matrix_per_year().T

    def mixing_ratios(self, tracer_moles: np.ndarray) -> np.ndarray:
        return np.asarray(tracer_moles, dtype=float) / self.air_moles

    def conservation_residual(self, tracer_moles: np.ndarray) -> np.ndarray:
        """Return summed tracer tendency; zero is exact column conservation."""

        return np.sum(self.derivative(tracer_moles), axis=-1)

    def interface_rate_constants_per_year(self) -> tuple[dict[str, float | str], ...]:
        index = {layer.name: i for i, layer in enumerate(self.layers)}
        air = self.air_moles
        rows = []
        for interface in self.interfaces:
            first = index[interface.first_layer]
            second = index[interface.second_layer]
            rows.append(
                {
                    "first_layer": interface.first_layer,
                    "second_layer": interface.second_layer,
                    "first_to_second_per_year": interface.gross_air_flux_mol_per_year / air[first],
                    "second_to_first_per_year": interface.gross_air_flux_mol_per_year / air[second],
                    "gross_air_flux_mol_per_year": interface.gross_air_flux_mol_per_year,
                    "source": interface.source,
                }
            )
        return tuple(rows)


def young_two_layer_column() -> ConservativeColumn:
    """Return the exact Young et al. stratosphere-troposphere exchange pair."""

    stratosphere = PARAMETERS["moles_stratosphere"]
    troposphere = PARAMETERS["moles_troposphere"]
    downward_flux = PARAMETERS["k_ST_per_year"] * stratosphere
    upward_flux = PARAMETERS["k_TS_per_year"] * troposphere
    if not np.isclose(downward_flux, upward_flux, rtol=1.0e-12, atol=0.0):
        raise ValueError("Young kST/kTS values do not define a stationary gross exchange flux")
    return ConservativeColumn(
        layers=(
            AtmosphericLayer("troposphere", troposphere),
            AtmosphericLayer("stratosphere", stratosphere),
        ),
        interfaces=(
            ExchangeInterface(
                "troposphere",
                "stratosphere",
                downward_flux,
                "Young et al. (2014) Section 3.3; Appenzeller and Holton (1996)",
            ),
        ),
    )
