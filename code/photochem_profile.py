"""Adapters for versioned Photochem atmospheric profiles.

Photochem is GPL-3.0 licensed. This project does not redistribute its example
files. The loader accepts a locally downloaded file and verifies the pinned
v0.6.7 ModernEarth profile by SHA-256 when requested.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from conservative_column_transport import (
    AtmosphericLayer,
    ConservativeColumn,
    ExchangeInterface,
)
from vertical_column_profile import ValidatedVerticalProfile, VerticalCell


AVOGADRO_PER_MOL = 6.02214076e23
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
EARTH_RADIUS_CM = 6.371e8

PHOTOCHEM_VERSION = "0.6.7"
PHOTOCHEM_RELEASE_URL = (
    "https://github.com/Nicholaswogan/photochem/releases/tag/v0.6.7"
)
PHOTOCHEM_MODERN_EARTH_RELATIVE_PATH = "examples/ModernEarth/atmosphere.txt"
PHOTOCHEM_MODERN_EARTH_SHA256 = (
    "d2a3d6cfef512ade9f1e6dc80217f05e1c0f116f2b77205cbdef789c8699b2be"
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shell_volume_cm3(
    lower_altitude_km: float,
    upper_altitude_km: float,
    planetary_radius_cm: float,
) -> float:
    lower_radius = planetary_radius_cm + lower_altitude_km * 1.0e5
    upper_radius = planetary_radius_cm + upper_altitude_km * 1.0e5
    return (4.0 * np.pi / 3.0) * (upper_radius**3 - lower_radius**3)


def load_photochem_v067_modern_earth_profile(
    atmosphere_path: str | Path,
    *,
    verify_checksum: bool = True,
    planetary_radius_cm: float = EARTH_RADIUS_CM,
) -> ValidatedVerticalProfile:
    """Load the pinned 100-layer Photochem v0.6.7 ModernEarth profile.

    Grid edges come from ``examples/ModernEarth/settings.yaml``: 0 to 100 km
    with 100 layers. The atmosphere file supplies layer-center altitude,
    pressure, number density, temperature, and eddy diffusivity.
    """

    path = Path(atmosphere_path)
    if verify_checksum:
        actual = file_sha256(path)
        if actual != PHOTOCHEM_MODERN_EARTH_SHA256:
            raise ValueError(
                "Photochem ModernEarth profile checksum mismatch: "
                f"expected {PHOTOCHEM_MODERN_EARTH_SHA256}, found {actual}"
            )

    data = np.genfromtxt(
        path,
        names=True,
        usecols=("alt", "press", "den", "temp", "eddy"),
        dtype=float,
        encoding="ascii",
    )
    if data.shape != (100,):
        raise ValueError(f"expected 100 Photochem layers, found {data.shape}")

    edges_km = np.linspace(0.0, 100.0, 101)
    expected_centers_km = 0.5 * (edges_km[:-1] + edges_km[1:])
    if not np.allclose(data["alt"], expected_centers_km, rtol=0.0, atol=1.0e-12):
        raise ValueError("Photochem altitude centers do not match the pinned settings grid")
    if not np.all(np.diff(data["press"]) < 0.0):
        raise ValueError("Photochem pressure must decrease upward")

    cells = []
    for index in range(100):
        lower = float(edges_km[index])
        upper = float(edges_km[index + 1])
        number_density = float(data["den"][index])
        air_moles = (
            number_density
            * _shell_volume_cm3(lower, upper, planetary_radius_cm)
            / AVOGADRO_PER_MOL
        )
        cells.append(
            VerticalCell(
                lower_altitude_km=lower,
                upper_altitude_km=upper,
                air_moles=air_moles,
                number_density_molecules_cm3=number_density,
                temperature_k=float(data["temp"][index]),
                pressure_center_bar=float(data["press"][index]),
                eddy_diffusivity_cm2_per_s=float(data["eddy"][index]),
            )
        )

    source = (
        f"Photochem v{PHOTOCHEM_VERSION} {PHOTOCHEM_MODERN_EARTH_RELATIVE_PATH}; "
        f"SHA-256 {PHOTOCHEM_MODERN_EARTH_SHA256}; Wogan et al. (2025)"
    )
    return ValidatedVerticalProfile(
        name="Photochem v0.6.7 ModernEarth",
        cells=tuple(cells),
        atmospheric_state_source=source,
        eddy_diffusivity_source=(
            f"{source}; modern-Earth Kzz profile described by Wogan et al. (2025)"
        ),
    )


def eddy_diffusion_column(
    profile: ValidatedVerticalProfile,
    *,
    planetary_radius_cm: float = EARTH_RADIUS_CM,
) -> ConservativeColumn:
    """Map Kzz to conservative bidirectional mixing-ratio exchange.

    At each interface, the gross air exchange is

        Kzz_interface * n_interface * area_interface / center_spacing,

    using the geometric means for Kzz and number density used at Photochem
    interfaces. Molecular diffusion, thermal diffusion, and mean vertical
    advection are intentionally excluded from this first passive-tracer test.
    """

    layers = tuple(
        AtmosphericLayer(
            name=f"z_{cell.lower_altitude_km:05.1f}_{cell.upper_altitude_km:05.1f}_km",
            air_moles=cell.air_moles,
        )
        for cell in profile.cells
    )
    interfaces = []
    for lower_index, (lower, upper) in enumerate(zip(profile.cells[:-1], profile.cells[1:])):
        interface_altitude_km = lower.upper_altitude_km
        area_cm2 = 4.0 * np.pi * (
            planetary_radius_cm + interface_altitude_km * 1.0e5
        ) ** 2
        center_spacing_cm = (
            0.5 * (upper.lower_altitude_km + upper.upper_altitude_km)
            - 0.5 * (lower.lower_altitude_km + lower.upper_altitude_km)
        ) * 1.0e5
        kzz_interface = np.sqrt(
            lower.eddy_diffusivity_cm2_per_s * upper.eddy_diffusivity_cm2_per_s
        )
        number_density_interface = np.sqrt(
            lower.number_density_molecules_cm3
            * upper.number_density_molecules_cm3
        )
        gross_moles_per_year = (
            kzz_interface
            * number_density_interface
            * area_cm2
            / center_spacing_cm
            * SECONDS_PER_YEAR
            / AVOGADRO_PER_MOL
        )
        interfaces.append(
            ExchangeInterface(
                first_layer=layers[lower_index].name,
                second_layer=layers[lower_index + 1].name,
                gross_air_flux_mol_per_year=float(gross_moles_per_year),
                source=(
                    f"{profile.eddy_diffusivity_source}; geometric interface "
                    "means follow Photochem v0.6.7 diffusion_coefficients_evo"
                ),
            )
        )
    return ConservativeColumn(layers=layers, interfaces=tuple(interfaces))
