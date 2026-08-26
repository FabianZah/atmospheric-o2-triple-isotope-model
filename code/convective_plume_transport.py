"""Conservative bulk-plume transport for passive atmospheric tracers.

The formulation follows the passive-tracer terms in ECMWF IFS Part IV,
Equations 6.1, 6.3, and 6.14. Updraught and downdraught plume mass fluxes are
defined on vertical interfaces. Layer detrainment is supplied explicitly and
entrainment is diagnosed from plume mass continuity.

This module is independent of ERA5 storage conventions. All air fluxes use
mol air yr-1 after an external adapter has converted native mass-flux and
detrainment fields. The same linear operator can therefore be applied to every
CO2 isotopologue without creating an isotope effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conservative_column_transport import AtmosphericLayer


IFS_CONVECTION_SOURCE = (
    "ECMWF IFS Documentation Part IV, Chapter 6, Equations 6.1, 6.3, and 6.14"
)


def _finite_nonnegative(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


@dataclass(frozen=True)
class ConvectivePlumeColumn:
    """One fixed-air column with entraining updraught and downdraught plumes.

    Layers and interfaces are ordered from bottom to top. Downdraught flux is
    stored as a non-negative downward magnitude. Updraught flux at the upper
    boundary and downdraught flux at the lower boundary must vanish. A plume
    entering at the other boundary is initialized with the adjacent layer's
    composition, representing launch from within the modeled column.

    The finite-volume plume step mixes entrained environmental air into the
    incoming plume before detrainment. Compensating environmental motion uses
    upwind composition at each interface. This is a monotone discretization of
    the printed IFS flux-divergence equations, not an eddy diffusivity.
    """

    layers: tuple[AtmosphericLayer, ...]
    updraught_interface_flux_mol_per_year: np.ndarray
    downdraught_interface_flux_mol_per_year: np.ndarray
    updraught_detrainment_mol_per_year: np.ndarray
    downdraught_detrainment_mol_per_year: np.ndarray
    source: str = IFS_CONVECTION_SOURCE
    continuity_relative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("a convective plume column requires at least one layer")
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("convective plume layer names must be unique")
        if not self.source:
            raise ValueError("convective plume transport requires a provenance source")
        if self.continuity_relative_tolerance < 0.0:
            raise ValueError("continuity tolerance must be non-negative")

        layer_count = len(self.layers)
        updraught = _finite_nonnegative(
            "updraught interface flux",
            self.updraught_interface_flux_mol_per_year,
        )
        downdraught = _finite_nonnegative(
            "downdraught interface flux",
            self.downdraught_interface_flux_mol_per_year,
        )
        up_detrainment = _finite_nonnegative(
            "updraught detrainment",
            self.updraught_detrainment_mol_per_year,
        )
        down_detrainment = _finite_nonnegative(
            "downdraught detrainment",
            self.downdraught_detrainment_mol_per_year,
        )
        if updraught.shape != (layer_count + 1,):
            raise ValueError("updraught flux requires one value per vertical interface")
        if downdraught.shape != (layer_count + 1,):
            raise ValueError("downdraught flux requires one value per vertical interface")
        if up_detrainment.shape != (layer_count,):
            raise ValueError("updraught detrainment requires one value per layer")
        if down_detrainment.shape != (layer_count,):
            raise ValueError("downdraught detrainment requires one value per layer")

        scale = max(
            float(np.max(updraught)),
            float(np.max(downdraught)),
            float(np.max(up_detrainment)),
            float(np.max(down_detrainment)),
            1.0,
        )
        tolerance = self.continuity_relative_tolerance * scale
        if updraught[-1] > tolerance:
            raise ValueError("updraught flux must terminate at the upper boundary")
        if downdraught[0] > tolerance:
            raise ValueError("downdraught flux must terminate at the lower boundary")

        up_entrainment = updraught[1:] - updraught[:-1] + up_detrainment
        down_entrainment = (
            downdraught[:-1] - downdraught[1:] + down_detrainment
        )
        if np.min(up_entrainment) < -tolerance:
            raise ValueError(
                "updraught mass-flux divergence and detrainment imply negative entrainment"
            )
        if np.min(down_entrainment) < -tolerance:
            raise ValueError(
                "downdraught mass-flux divergence and detrainment imply negative entrainment"
            )

        object.__setattr__(
            self,
            "updraught_interface_flux_mol_per_year",
            updraught.copy(),
        )
        object.__setattr__(
            self,
            "downdraught_interface_flux_mol_per_year",
            downdraught.copy(),
        )
        object.__setattr__(
            self,
            "updraught_detrainment_mol_per_year",
            up_detrainment.copy(),
        )
        object.__setattr__(
            self,
            "downdraught_detrainment_mol_per_year",
            down_detrainment.copy(),
        )

        # Construct once during validation so a non-monotone discretization is
        # rejected before it reaches an isotope calculation.
        inventory_matrix = self._build_inventory_transport_matrix()
        off_diagonal = inventory_matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        matrix_scale = max(float(np.max(np.abs(inventory_matrix))), 1.0)
        matrix_tolerance = self.continuity_relative_tolerance * matrix_scale
        if np.min(off_diagonal) < -matrix_tolerance:
            raise ValueError("convective plume discretization is not positivity preserving")
        if np.max(np.diag(inventory_matrix)) > matrix_tolerance:
            raise ValueError("convective plume diagonal loss rates must be non-positive")
        if np.max(np.abs(np.sum(inventory_matrix, axis=0))) > matrix_tolerance:
            raise ValueError("convective plume operator does not conserve tracer inventory")
        if np.max(np.abs(inventory_matrix @ self.air_moles)) > (
            matrix_tolerance * max(float(np.max(self.air_moles)), 1.0)
        ):
            raise ValueError("convective plume operator does not preserve uniform composition")

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    @property
    def air_moles(self) -> np.ndarray:
        return np.asarray([layer.air_moles for layer in self.layers], dtype=float)

    @property
    def updraught_entrainment_mol_per_year(self) -> np.ndarray:
        entrainment = (
            self.updraught_interface_flux_mol_per_year[1:]
            - self.updraught_interface_flux_mol_per_year[:-1]
            + self.updraught_detrainment_mol_per_year
        )
        scale = max(
            float(np.max(self.updraught_interface_flux_mol_per_year)),
            float(np.max(self.updraught_detrainment_mol_per_year)),
            1.0,
        )
        tolerance = self.continuity_relative_tolerance * scale
        return np.where(
            (entrainment < 0.0) & (entrainment >= -tolerance),
            0.0,
            entrainment,
        )

    @property
    def downdraught_entrainment_mol_per_year(self) -> np.ndarray:
        entrainment = (
            self.downdraught_interface_flux_mol_per_year[:-1]
            - self.downdraught_interface_flux_mol_per_year[1:]
            + self.downdraught_detrainment_mol_per_year
        )
        scale = max(
            float(np.max(self.downdraught_interface_flux_mol_per_year)),
            float(np.max(self.downdraught_detrainment_mol_per_year)),
            1.0,
        )
        tolerance = self.continuity_relative_tolerance * scale
        return np.where(
            (entrainment < 0.0) & (entrainment >= -tolerance),
            0.0,
            entrainment,
        )

    def _plume_interface_compositions(
        self,
        environmental_mixing_ratio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        environmental = np.asarray(environmental_mixing_ratio, dtype=float)
        layer_count = len(self.layers)
        if environmental.shape != (layer_count,):
            raise ValueError("environmental mixing ratio must match the plume layers")
        if not np.all(np.isfinite(environmental)):
            raise ValueError("environmental mixing ratio must be finite")

        updraught = self.updraught_interface_flux_mol_per_year
        downdraught = self.downdraught_interface_flux_mol_per_year
        up_entrainment = self.updraught_entrainment_mol_per_year
        down_entrainment = self.downdraught_entrainment_mol_per_year

        up_composition = np.zeros(layer_count + 1, dtype=float)
        up_composition[0] = environmental[0]
        for layer_index in range(layer_count):
            mixed_air = updraught[layer_index] + up_entrainment[layer_index]
            if mixed_air > 0.0:
                up_composition[layer_index + 1] = (
                    updraught[layer_index] * up_composition[layer_index]
                    + up_entrainment[layer_index] * environmental[layer_index]
                ) / mixed_air
            else:
                up_composition[layer_index + 1] = environmental[layer_index]

        down_composition = np.zeros(layer_count + 1, dtype=float)
        down_composition[-1] = environmental[-1]
        for layer_index in range(layer_count - 1, -1, -1):
            mixed_air = downdraught[layer_index + 1] + down_entrainment[layer_index]
            if mixed_air > 0.0:
                down_composition[layer_index] = (
                    downdraught[layer_index + 1]
                    * down_composition[layer_index + 1]
                    + down_entrainment[layer_index] * environmental[layer_index]
                ) / mixed_air
            else:
                down_composition[layer_index] = environmental[layer_index]
        return up_composition, down_composition

    def inventory_tendency_from_mixing_ratio(
        self,
        environmental_mixing_ratio: np.ndarray,
    ) -> np.ndarray:
        """Return tracer mol yr-1 from the IFS convective flux divergence."""

        environmental = np.asarray(environmental_mixing_ratio, dtype=float)
        up_composition, down_composition = self._plume_interface_compositions(
            environmental
        )
        layer_count = len(self.layers)
        upward_anomaly_flux = np.zeros(layer_count + 1, dtype=float)
        for interface_index in range(1, layer_count):
            upward_anomaly_flux[interface_index] = (
                self.updraught_interface_flux_mol_per_year[interface_index]
                * (up_composition[interface_index] - environmental[interface_index])
                - self.downdraught_interface_flux_mol_per_year[interface_index]
                * (
                    down_composition[interface_index]
                    - environmental[interface_index - 1]
                )
            )
        return upward_anomaly_flux[:-1] - upward_anomaly_flux[1:]

    def _build_inventory_transport_matrix(self) -> np.ndarray:
        layer_count = len(self.layers)
        mixing_ratio_flux_matrix = np.column_stack(
            [
                self.inventory_tendency_from_mixing_ratio(
                    np.eye(layer_count, dtype=float)[:, source_index]
                )
                for source_index in range(layer_count)
            ]
        )
        return mixing_ratio_flux_matrix / self.air_moles[None, :]

    def transport_matrix_per_year(self) -> np.ndarray:
        """Return matrix A such that dN/dt = A @ N for tracer inventories."""

        return self._build_inventory_transport_matrix()

    def mixing_ratio_transport_matrix_per_year(self) -> np.ndarray:
        inventory = self.transport_matrix_per_year()
        air = self.air_moles
        return (inventory * air[None, :]) / air[:, None]

    def derivative(self, tracer_moles: np.ndarray) -> np.ndarray:
        tracer = np.asarray(tracer_moles, dtype=float)
        if tracer.shape[-1] != len(self.layers):
            raise ValueError("last tracer dimension must equal the number of layers")
        return tracer @ self.transport_matrix_per_year().T

    def conservation_residual(self, tracer_moles: np.ndarray) -> np.ndarray:
        return np.sum(self.derivative(tracer_moles), axis=-1)


@dataclass(frozen=True)
class ConvectivePlumeGrid:
    """Independent latitude columns assembled in latitude-major order."""

    columns: tuple[ConvectivePlumeColumn, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("a convective plume grid requires at least one column")
        layer_count = len(self.columns[0].layers)
        if any(len(column.layers) != layer_count for column in self.columns):
            raise ValueError("all convective plume columns must share a vertical size")
        names = [name for column in self.columns for name in column.layer_names]
        if len(names) != len(set(names)):
            raise ValueError("convective plume grid layer names must be globally unique")

    @property
    def air_moles(self) -> np.ndarray:
        return np.concatenate([column.air_moles for column in self.columns])

    def transport_matrix_per_year(self) -> np.ndarray:
        sizes = [len(column.layers) for column in self.columns]
        total = sum(sizes)
        matrix = np.zeros((total, total), dtype=float)
        start = 0
        for column, size in zip(self.columns, sizes, strict=True):
            matrix[start : start + size, start : start + size] = (
                column.transport_matrix_per_year()
            )
            start += size
        return matrix
