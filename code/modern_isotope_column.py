"""Build the fixed-boundary Modern Earth triple-oxygen chemistry column."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log1p
from pathlib import Path

import numpy as np

from fixed_boundary_isotope_column import FixedBoundaryIsotopeColumn
from gridded_isotope_transport import GriddedSpeciesSystem
from gridded_oxygen_chemistry import (
    ATMOSPHERIC_OXYGEN_SPECIES,
    bind_local_reaction_operator,
    bind_local_reaction_throughput_operator,
)
from isotopes import R17_VSMOW, R18_VSMOW
from photochem_profile import (
    eddy_diffusion_column,
    load_photochem_v067_modern_earth_profile,
)
from photochem_two_stream import load_two_stream_modern_earth_photolysis
from modern_photolysis import load_photolysis_profile
from r1_r7_grid_chemistry import (
    full_young_r1_r7_grid_reactions,
    modern_r1_r7_rate_fields,
)
from vertical_column_profile import ValidatedVerticalProfile


@dataclass(frozen=True)
class PrimeIsotopeComposition:
    """One oxygen-isotope composition in logarithmic per-mil notation."""

    delta18_prime_permil: float
    cap_delta17_prime_permil: float
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta18_prime_permil):
            raise ValueError("delta18-prime must be finite")
        if not np.isfinite(self.cap_delta17_prime_permil):
            raise ValueError("Delta-prime-17O must be finite")
        if not self.source:
            raise ValueError("isotope composition requires provenance")

    @classmethod
    def from_delta18_and_cap_delta17(
        cls,
        *,
        delta18_permil: float,
        cap_delta17_prime_permil: float,
        source: str,
    ) -> "PrimeIsotopeComposition":
        """Convert conventional delta18O to the solver's logarithmic notation."""

        if not np.isfinite(delta18_permil) or delta18_permil <= -1000.0:
            raise ValueError("delta18 must be finite and greater than -1000 per mil")
        return cls(
            delta18_prime_permil=1000.0 * log1p(delta18_permil / 1000.0),
            cap_delta17_prime_permil=cap_delta17_prime_permil,
            source=source,
        )

    @property
    def delta17_prime_permil(self) -> float:
        return self.cap_delta17_prime_permil + 0.528 * self.delta18_prime_permil

    @property
    def ratio17(self) -> float:
        return R17_VSMOW * exp(self.delta17_prime_permil / 1000.0)

    @property
    def ratio18(self) -> float:
        return R18_VSMOW * exp(self.delta18_prime_permil / 1000.0)


@dataclass(frozen=True)
class ModernColumnBuild:
    column: FixedBoundaryIsotopeColumn
    profile: ValidatedVerticalProfile
    altitude_km: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    number_density_molecules_cm3: np.ndarray
    reaction_count: int
    photolysis_source: str
    o2_boundary_source: str
    co2_boundary_source: str
    omitted_o2_o1d_maximum_ratio: float
    prescribed_parent_species: bool
    transport_scale: float = 1.0
    co2_parent_source: str = ""


def modern_reference_isotope_compositions() -> tuple[
    PrimeIsotopeComposition, PrimeIsotopeComposition
]:
    """Return the source-backed O2 and upper-tropospheric CO2 boundaries."""

    o2 = PrimeIsotopeComposition(
        delta18_prime_permil=23.600,
        cap_delta17_prime_permil=-0.432,
        source=(
            "Barkan and Luz (2011) delta18-prime as quoted by Young et al. "
            "(2014); Pack (2021) Delta-prime-17O"
        ),
    )
    co2 = PrimeIsotopeComposition.from_delta18_and_cap_delta17(
        delta18_permil=41.78933333333333,
        cap_delta17_prime_permil=-0.2186969696969697,
        source=(
            "Adnew et al. (2025) Supplement Table S2 mean for n=33 CARIBIC "
            "upper-tropospheric samples selected by N2O >= 313.5 ppb; "
            "delta18O=41.7893 per mil VSMOW converted to delta18-prime; "
            "Delta-prime-17O=-0.218697 per mil, lambda_ref=0.528"
        ),
    )
    return o2, co2


def ozone_constrained_column_from_chapman_state(
    build: ModernColumnBuild,
    chapman_inventory_moles: np.ndarray,
) -> FixedBoundaryIsotopeColumn:
    """Apply Photochem O3 abundance while preserving solved isotope ratios.

    The input state must come from the free-parent R1-R7 column on the same
    grid. O3, OO17O, and OO18O are then prescribed together, so no
    isotope-inconsistent parent-only closure is introduced.
    """

    if build.prescribed_parent_species:
        raise ValueError("ozone constraint requires a free-parent Chapman build")
    column = build.column
    inventory = column.species_system.validate_inventory(chapman_inventory_moles)
    species_index = {
        name: index for index, name in enumerate(column.species_system.species_names)
    }
    o3_index = species_index["O3"]
    o17_index = species_index["OO17O"]
    o18_index = species_index["OO18O"]
    ratio17 = inventory[o17_index] / inventory[o3_index]
    ratio18 = inventory[o18_index] / inventory[o3_index]
    target_o3 = np.asarray(column.prescribed_inventory_moles[o3_index], dtype=float)
    constrained_inventory = inventory.copy()
    constrained_inventory[o3_index] = target_o3
    constrained_inventory[o17_index] = target_o3 * ratio17
    constrained_inventory[o18_index] = target_o3 * ratio18
    fixed = np.asarray(column.fixed_mask, dtype=bool).copy()
    fixed[[o3_index, o17_index, o18_index], :] = True
    return replace(
        column,
        fixed_mask=fixed,
        prescribed_inventory_moles=constrained_inventory,
        source=(
            f"{column.source}; Photochem O3 abundance constrained with "
            "free-Chapman R1-R7 isotope ratios"
        ),
    )


def _subprofile(
    profile: ValidatedVerticalProfile,
    lower_altitude_km: float,
    upper_altitude_km: float,
) -> tuple[ValidatedVerticalProfile, np.ndarray]:
    selected = profile.whole_cell_indices(lower_altitude_km, upper_altitude_km)
    cells = tuple(profile.cells[int(index)] for index in selected)
    return (
        ValidatedVerticalProfile(
            name=f"{profile.name} {lower_altitude_km:g}-{upper_altitude_km:g} km",
            cells=cells,
            atmospheric_state_source=profile.atmospheric_state_source,
            eddy_diffusivity_source=profile.eddy_diffusivity_source,
        ),
        selected,
    )


def _rare_mixing_ratio(
    parent_mixing_ratio: np.ndarray,
    composition: PrimeIsotopeComposition,
    *,
    isotope: int,
    represented_sites: float,
) -> np.ndarray:
    ratio = composition.ratio17 if isotope == 17 else composition.ratio18
    return represented_sites * np.asarray(parent_mixing_ratio, dtype=float) * ratio


def build_modern_fixed_boundary_column(
    external_data_directory: Path,
    *,
    o2_composition: PrimeIsotopeComposition,
    co2_composition: PrimeIsotopeComposition,
    lower_altitude_km: float = 10.0,
    upper_altitude_km: float = 73.0,
    ozone_quantum_yield_convention: str = "pinned",
    photolysis_cache_path: Path | None = None,
    prescribe_parent_species: bool = True,
    transport_scale: float = 1.0,
    co2_parent_mixing_ratio: float | None = None,
    co2_parent_source: str | None = None,
) -> ModernColumnBuild:
    """Construct, but do not solve, the source-backed modern column.

    O2, O17O, O18O, and major C16O2 are prescribed in every cell. By default,
    parent O, O(1D), and O3 are also prescribed from the validated Photochem
    background because R1-R7 do not contain its HOx/NOx/ClOx chemistry. Rare
    atomic, ozone, and CO2 isotopologues remain free above the lower boundary.
    """

    if not np.isfinite(transport_scale) or transport_scale <= 0.0:
        raise ValueError("transport scale must be finite and positive")
    if co2_parent_mixing_ratio is not None:
        if (
            not np.isfinite(co2_parent_mixing_ratio)
            or co2_parent_mixing_ratio <= 0.0
            or co2_parent_mixing_ratio >= 1.0
        ):
            raise ValueError("CO2 parent mixing ratio must lie between zero and one")
        if not co2_parent_source:
            raise ValueError("an overridden CO2 parent abundance requires provenance")
    external = Path(external_data_directory)
    full_profile = load_photochem_v067_modern_earth_profile(
        external / "photochem-v0.6.7-modern-earth" / "atmosphere.txt"
    )
    profile, selected = _subprofile(
        full_profile, lower_altitude_km, upper_altitude_km
    )
    transport = eddy_diffusion_column(profile)
    system = GriddedSpeciesSystem(
        species_names=ATMOSPHERIC_OXYGEN_SPECIES,
        air_moles=transport.air_moles,
        inventory_transport_matrix_per_year=(
            transport_scale * transport.transport_matrix_per_year()
        ),
        source=(
            f"{profile.eddy_diffusivity_source}; fixed-boundary photochemical "
            f"column; Kzz scale={transport_scale:g}; Kzz is not interpreted "
            "as complete Brewer-Dobson transport"
        ),
    )

    atmosphere_path = (
        external
        / "photochem-v0.6.7-source"
        / "photochem-0.6.7"
        / "examples"
        / "ModernEarth"
        / "atmosphere.txt"
    )
    atmosphere = np.genfromtxt(atmosphere_path, names=True)
    altitude = np.asarray(atmosphere["alt"][selected], dtype=float)
    pressure = np.asarray(atmosphere["press"][selected], dtype=float) * 1.0e5
    temperature = np.asarray(atmosphere["temp"][selected], dtype=float)
    number_density = np.asarray(atmosphere["den"][selected], dtype=float)
    parent_co2 = (
        np.asarray(atmosphere["CO2"][selected], dtype=float)
        if co2_parent_mixing_ratio is None
        else np.full(len(selected), float(co2_parent_mixing_ratio), dtype=float)
    )
    resolved_co2_parent_source = (
        f"{profile.atmospheric_state_source} CO2 profile"
        if co2_parent_mixing_ratio is None
        else str(co2_parent_source)
    )
    rate_fields = modern_r1_r7_rate_fields(
        temperature,
        number_density,
        o2_mixing_ratio=atmosphere["O2"][selected],
        n2_mixing_ratio=atmosphere["N2"][selected],
        co2_mixing_ratio=parent_co2,
    )
    if photolysis_cache_path is None:
        photolysis_profile = load_two_stream_modern_earth_photolysis(
            external,
            ozone_quantum_yield_convention=ozone_quantum_yield_convention,
        )
    else:
        photolysis_profile = load_photolysis_profile(photolysis_cache_path)
    photolysis = photolysis_profile.interpolate(altitude)
    reactions = full_young_r1_r7_grid_reactions(rate_fields, photolysis)
    chemistry = bind_local_reaction_operator(
        species_names=system.species_names,
        air_moles=system.air_moles,
        pressure_pa=pressure,
        temperature_k=temperature,
        reactions=reactions,
    )
    chemistry_throughput = bind_local_reaction_throughput_operator(
        species_names=system.species_names,
        air_moles=system.air_moles,
        pressure_pa=pressure,
        temperature_k=temperature,
        reactions=reactions,
    )

    parent = {
        "O": np.asarray(atmosphere["O"][selected], dtype=float),
        "O1D": np.asarray(atmosphere["O1D"][selected], dtype=float),
        "O2": np.asarray(atmosphere["O2"][selected], dtype=float),
        "CO2": parent_co2,
        "O3": np.asarray(atmosphere["O3"][selected], dtype=float),
    }
    tiny = np.finfo(float).tiny
    mixing_ratio = {
        "O": np.maximum(parent["O"], tiny),
        "O17": np.maximum(parent["O"], tiny) * o2_composition.ratio17,
        "O18": np.maximum(parent["O"], tiny) * o2_composition.ratio18,
        "O1D": np.maximum(parent["O1D"], tiny),
        "O17_1D": np.maximum(parent["O1D"], tiny) * o2_composition.ratio17,
        "O18_1D": np.maximum(parent["O1D"], tiny) * o2_composition.ratio18,
        "O2": parent["O2"],
        "O17O": _rare_mixing_ratio(
            parent["O2"], o2_composition, isotope=17, represented_sites=2.0
        ),
        "O18O": _rare_mixing_ratio(
            parent["O2"], o2_composition, isotope=18, represented_sites=2.0
        ),
        "CO2": parent["CO2"],
        "CO17O": _rare_mixing_ratio(
            parent["CO2"], co2_composition, isotope=17, represented_sites=1.0
        ),
        "CO18O": _rare_mixing_ratio(
            parent["CO2"], co2_composition, isotope=18, represented_sites=1.0
        ),
        "O3": np.maximum(parent["O3"], tiny),
        "OO17O": _rare_mixing_ratio(
            np.maximum(parent["O3"], tiny),
            o2_composition,
            isotope=17,
            represented_sites=3.0,
        ),
        "OO18O": _rare_mixing_ratio(
            np.maximum(parent["O3"], tiny),
            o2_composition,
            isotope=18,
            represented_sites=3.0,
        ),
    }
    inventory = np.asarray(
        [mixing_ratio[name] * system.air_moles for name in system.species_names]
    )
    fixed = np.zeros_like(inventory, dtype=bool)
    fixed[:, 0] = True
    species_index = {name: index for index, name in enumerate(system.species_names)}
    for name in ("O2", "O17O", "O18O", "CO2"):
        fixed[species_index[name], :] = True
    if prescribe_parent_species:
        for name in ("O", "O1D", "O3"):
            fixed[species_index[name], :] = True

    o1d = np.asarray(photolysis_profile.j_o2_to_o_o1d_per_s, dtype=float)
    parent_o = np.asarray(photolysis_profile.j_o2_to_o_o_per_s, dtype=float)
    domain = (
        photolysis_profile.altitude_km >= lower_altitude_km
    ) & (photolysis_profile.altitude_km <= upper_altitude_km)
    omitted_ratio = float(
        np.max(np.divide(o1d[domain], parent_o[domain], out=np.zeros_like(o1d[domain]), where=parent_o[domain] > 0.0))
    )
    column = FixedBoundaryIsotopeColumn(
        species_system=system,
        local_chemistry=chemistry,
        fixed_mask=fixed,
        prescribed_inventory_moles=inventory,
        source=(
            f"{profile.atmospheric_state_source}; {photolysis_profile.source}; "
            f"O2 boundary: {o2_composition.source}; CO2 isotope boundary: "
            f"{co2_composition.source}; CO2 parent abundance: "
            f"{resolved_co2_parent_source}"
        ),
        local_chemistry_throughput=chemistry_throughput,
    )
    return ModernColumnBuild(
        column=column,
        profile=profile,
        altitude_km=altitude,
        pressure_pa=pressure,
        temperature_k=temperature,
        number_density_molecules_cm3=number_density,
        reaction_count=len(reactions),
        photolysis_source=photolysis_profile.source,
        o2_boundary_source=o2_composition.source,
        co2_boundary_source=co2_composition.source,
        omitted_o2_o1d_maximum_ratio=omitted_ratio,
        prescribed_parent_species=bool(prescribe_parent_species),
        transport_scale=float(transport_scale),
        co2_parent_source=resolved_co2_parent_source,
    )
