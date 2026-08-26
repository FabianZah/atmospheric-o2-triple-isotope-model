"""Conservative meridional eddy diffusion for a latitude-pressure grid.

Morgan et al. (2004), Equation A1, treats Kyy separately from residual
circulation and Kzz. This module implements that finite-volume operator but
does not prescribe Kyy. A caller must supply a non-negative, sourced field on
the internal latitude faces and vertical cell centres.
"""

from __future__ import annotations

import numpy as np

from conservative_circulation_transport import latitude_vertical_cell_name
from conservative_column_transport import (
    AtmosphericLayer,
    ConservativeColumn,
    ExchangeInterface,
)
from meridional_transport_reference import (
    SECONDS_PER_YEAR,
    STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    STANDARD_GRAVITY_M_PER_S2,
)


def meridional_eddy_diffusion_operator(
    *,
    air_moles: np.ndarray,
    latitude_edges_degrees: np.ndarray,
    pressure_edges_pa: np.ndarray,
    kyy_m2_per_s: np.ndarray,
    source: str,
) -> ConservativeColumn:
    """Return conservative Kyy exchange on internal latitude faces.

    ``kyy_m2_per_s`` has shape ``(n_latitude - 1, n_vertical)``. Each value is
    located on one internal latitude face and at one pressure-cell centre.
    Hydrostatic integration of the meridional face gives the gross exchange

        Kyy * 2 pi cos(latitude) * pressure_thickness
        ------------------------------------------------.
             gravity * dry_air_molar_mass * delta_latitude

    Planetary radius cancels between zonal face circumference and meridional
    centre spacing. This is the pressure-coordinate finite-volume form of the
    meridional diffusion term in Morgan et al. (2004), Equation A1.
    """

    if not source:
        raise ValueError("Kyy transport requires explicit provenance")
    air = np.asarray(air_moles, dtype=float)
    latitude_edges = np.asarray(latitude_edges_degrees, dtype=float)
    pressure_edges = np.asarray(pressure_edges_pa, dtype=float)
    kyy = np.asarray(kyy_m2_per_s, dtype=float)
    if air.ndim != 2 or min(air.shape) < 1:
        raise ValueError("air inventory must be a non-empty latitude-vertical grid")
    if latitude_edges.shape != (air.shape[0] + 1,):
        raise ValueError("latitude edges do not match the air grid")
    if pressure_edges.shape != (air.shape[1] + 1,):
        raise ValueError("pressure edges do not match the air grid")
    if kyy.shape != (air.shape[0] - 1, air.shape[1]):
        raise ValueError("Kyy must have one value per internal latitude face and cell")
    if not np.all(np.isfinite(kyy)) or np.any(kyy < 0.0):
        raise ValueError("Kyy must be finite and non-negative")
    if not np.all(np.diff(latitude_edges) > 0.0):
        raise ValueError("latitude edges must increase south to north")
    if not np.all(np.diff(pressure_edges) < 0.0):
        raise ValueError("pressure edges must decrease upward")

    latitude_centers = 0.5 * (latitude_edges[:-1] + latitude_edges[1:])
    center_spacing_radians = np.deg2rad(np.diff(latitude_centers))
    interface_latitude_radians = np.deg2rad(latitude_edges[1:-1])
    pressure_thickness_pa = -np.diff(pressure_edges)
    layers = tuple(
        AtmosphericLayer(
            latitude_vertical_cell_name(latitude_index, vertical_index),
            float(air[latitude_index, vertical_index]),
        )
        for latitude_index in range(air.shape[0])
        for vertical_index in range(air.shape[1])
    )
    interfaces: list[ExchangeInterface] = []
    for latitude_face in range(air.shape[0] - 1):
        cosine = np.cos(interface_latitude_radians[latitude_face])
        for vertical_index in range(air.shape[1]):
            gross_air_flux = (
                kyy[latitude_face, vertical_index]
                * 2.0
                * np.pi
                * cosine
                * pressure_thickness_pa[vertical_index]
                / (
                    STANDARD_GRAVITY_M_PER_S2
                    * STANDARD_DRY_AIR_MOLAR_MASS_KG_PER_MOL
                    * center_spacing_radians[latitude_face]
                )
                * SECONDS_PER_YEAR
            )
            interfaces.append(
                ExchangeInterface(
                    first_layer=latitude_vertical_cell_name(
                        latitude_face,
                        vertical_index,
                    ),
                    second_layer=latitude_vertical_cell_name(
                        latitude_face + 1,
                        vertical_index,
                    ),
                    gross_air_flux_mol_per_year=float(gross_air_flux),
                    source=source,
                )
            )
    return ConservativeColumn(layers=layers, interfaces=tuple(interfaces))
