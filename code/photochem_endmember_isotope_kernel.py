"""Drive the rare-isotope R1-R7 kernel with native Photochem end members.

The parent atmosphere is prescribed from one converged Photochem 0.6.7
artifact.  Only rare O, O(1D), O3, and CO2 isotopologues are transported and
solved.  Globally mixed O2 isotopologues remain fixed so the fast column can
provide a conservative R7 forcing to the separate slow global O2 reservoir.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from fixed_boundary_isotope_column import FixedBoundaryIsotopeColumn
from gridded_isotope_transport import GriddedSpeciesSystem
from gridded_oxygen_chemistry import (
    ATMOSPHERIC_OXYGEN_SPECIES,
    ElementaryGridReaction,
    bind_local_reaction_operator,
    bind_local_reaction_throughput_operator,
)
from modern_isotope_reference import PrimeIsotopeComposition
from photochem_profile import eddy_diffusion_column
from r1_r7_grid_chemistry import (
    full_young_r1_r7_grid_reactions,
    modern_r1_r7_rate_fields,
)
from vertical_column_profile import ValidatedVerticalProfile, VerticalCell


AVOGADRO_PER_MOL = 6.02214076e23


@dataclass(frozen=True)
class PhotochemEndmemberIsotopeBuild:
    """One fixed-parent isotope kernel and its source-resolved coordinates."""

    column: FixedBoundaryIsotopeColumn
    profile: ValidatedVerticalProfile
    altitude_km: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    number_density_molecules_cm3: np.ndarray
    parent_mixing_ratios: dict[str, np.ndarray]
    reactions: tuple[ElementaryGridReaction, ...]
    r7_reactions: tuple[ElementaryGridReaction, ...]
    pO2_PAL: float
    pCO2_ppm: float
    a_mif: float
    artifact_sha256: str
    omitted_o2_o1d_maximum_ratio: float
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_edges_from_centers(altitude_km: np.ndarray) -> np.ndarray:
    altitude = np.asarray(altitude_km, dtype=float)
    if altitude.ndim != 1 or altitude.size < 2:
        raise ValueError("end-member altitude must be a one-dimensional grid")
    if not np.all(np.isfinite(altitude)) or np.any(np.diff(altitude) <= 0.0):
        raise ValueError("end-member altitude must be finite and increasing")
    edges = np.empty(altitude.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (altitude[:-1] + altitude[1:])
    edges[0] = altitude[0] - 0.5 * (altitude[1] - altitude[0])
    edges[-1] = altitude[-1] + 0.5 * (altitude[-1] - altitude[-2])
    if edges[0] < 0.0 and not np.isclose(edges[0], 0.0, atol=1.0e-10):
        raise ValueError("end-member grid extrapolates below the surface")
    edges[0] = max(0.0, edges[0])
    return edges


def _shell_volume_cm3(lower_km: float, upper_km: float, radius_cm: float) -> float:
    inner = radius_cm + lower_km * 1.0e5
    outer = radius_cm + upper_km * 1.0e5
    return float((4.0 * np.pi / 3.0) * (outer**3 - inner**3))


def _nonnegative_parent(name: str, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"native {name} mixing ratio must be finite")
    tolerance = 1.0e-12 * max(float(np.max(np.abs(raw))), np.finfo(float).tiny)
    if np.any(raw < -tolerance):
        raise ValueError(f"native {name} mixing ratio contains a significant negative")
    return np.maximum(raw, np.finfo(float).tiny)


def _rare_mixing_ratio(
    parent: np.ndarray,
    composition: PrimeIsotopeComposition,
    *,
    isotope: int,
    represented_sites: float,
) -> np.ndarray:
    ratio = composition.ratio17 if isotope == 17 else composition.ratio18
    return represented_sites * np.asarray(parent, dtype=float) * ratio


def _native_photolysis_fields(
    equations: tuple[str, ...], rates: np.ndarray, selected: np.ndarray
) -> tuple[dict[str, np.ndarray], float]:
    lookup = {equation: index for index, equation in enumerate(equations)}
    if len(lookup) != len(equations):
        raise ValueError("native photolysis reaction identities must be unique")
    required = {
        "j_r1_o2_to_o_o_per_s": "O2 + hv => O + O",
        "j_o2_to_o_o1d_per_s": "O2 + hv => O + O1D",
        "j_r3a_o3_to_o2_o_per_s": "O3 + hv => O + O2",
        "j_r3f_o3_to_o2_o1d_per_s": "O3 + hv => O1D + O2",
    }
    missing = sorted(set(required.values()).difference(lookup))
    if missing:
        raise KeyError(f"native artifact lacks photolysis channels: {missing}")
    fields = {
        key: np.asarray(rates[selected, lookup[equation]], dtype=float)
        for key, equation in required.items()
    }
    if any(np.any(value < 0.0) or not np.all(np.isfinite(value)) for value in fields.values()):
        raise ValueError("native photolysis fields must be finite and non-negative")
    parent = fields["j_r1_o2_to_o_o_per_s"]
    omitted = fields["j_o2_to_o_o1d_per_s"]
    maximum_ratio = float(
        np.max(np.divide(omitted, parent, out=np.zeros_like(omitted), where=parent > 0.0))
    )
    return fields, maximum_ratio


def _fixed_parent_ratio_corrected_operators(
    *,
    system: GriddedSpeciesSystem,
    chemistry,
    throughput,
):
    """Preserve isotope-ratio dynamics when parent profiles are prescribed.

    If a parent inventory N is held fixed, directly integrating only its rare
    inventory N* omits the denominator term in d(N*/N)/dt.  The correction
    ``-(N*/N) dN/dt`` is therefore applied using the complete internal parent
    tendency (transport plus chemistry).  It represents isotope-neutral
    compensation of the prescribed parent profile and contains no adjustable
    coefficient.
    """

    index = {name: position for position, name in enumerate(system.species_names)}
    families = {
        "O": ("O17", "O18"),
        "O1D": ("O17_1D", "O18_1D"),
        "CO2": ("CO17O", "CO18O"),
        "O3": ("OO17O", "OO18O"),
    }
    transport_matrix = np.asarray(
        system.inventory_transport_matrix_per_year, dtype=float
    )

    def correction(time_years: float, inventory_moles: np.ndarray) -> np.ndarray:
        inventory = system.validate_inventory(inventory_moles)
        chemical = np.asarray(chemistry(time_years, inventory), dtype=float)
        transported = inventory @ transport_matrix.T
        parent_internal = chemical + transported
        result = np.zeros_like(inventory)
        for parent, rare_names in families.items():
            parent_index = index[parent]
            parent_inventory = inventory[parent_index]
            for rare in rare_names:
                rare_index = index[rare]
                result[rare_index] = -(
                    inventory[rare_index]
                    / parent_inventory
                    * parent_internal[parent_index]
                )
        return result

    def corrected_chemistry(time_years: float, inventory_moles: np.ndarray) -> np.ndarray:
        base = np.asarray(chemistry(time_years, inventory_moles), dtype=float)
        return base + correction(time_years, inventory_moles)

    def corrected_throughput(time_years: float, inventory_moles: np.ndarray) -> np.ndarray:
        base = np.asarray(throughput(time_years, inventory_moles), dtype=float)
        return base + np.abs(correction(time_years, inventory_moles))

    return corrected_chemistry, corrected_throughput


def build_photochem_endmember_isotope_kernel(
    artifact_path: str | Path,
    *,
    o2_composition: PrimeIsotopeComposition,
    co2_composition: PrimeIsotopeComposition,
    lower_altitude_km: float = 10.0,
    upper_altitude_km: float = 73.0,
    transport_scale: float = 1.0,
    a_mif: float = 1.065,
) -> PhotochemEndmemberIsotopeBuild:
    """Build a fixed-parent isotope column from one native end-member artifact."""

    if not np.isfinite(transport_scale) or transport_scale <= 0.0:
        raise ValueError("transport scale must be finite and positive")
    if not np.isfinite(a_mif) or a_mif <= 0.0:
        raise ValueError("a_mif must be finite and positive")
    path = Path(artifact_path).resolve()
    with np.load(path, allow_pickle=False) as data:
        if not bool(data["converged"][0]):
            raise ValueError(f"Photochem end member is not converged: {path}")
        altitude_all = np.asarray(data["altitude_km"], dtype=float)
        species_names = tuple(str(value) for value in data["species_names"])
        mixing_all = np.asarray(data["mixing_ratios"], dtype=float)
        density_all = np.asarray(data["number_density_cm3"], dtype=float)
        pressure_all = np.asarray(data["pressure_dyn_cm2"], dtype=float)
        temperature_all = np.asarray(data["temperature_k"], dtype=float)
        eddy_all = np.asarray(data["eddy_diffusion_cm2_s"], dtype=float)
        radius_cm = float(data["planet_radius_cm"][0])
        equations = tuple(str(value) for value in data["photolysis_reaction_equations"])
        photolysis_all = np.asarray(data["photolysis_rates_per_s"], dtype=float)
        po2_pal = float(data["pO2_PAL"][0])
        pco2_ppm = float(data["pCO2_ppm"][0])
        version = str(data["photochem_version"][0])

    layer_count = altitude_all.size
    if mixing_all.shape != (len(species_names), layer_count):
        raise ValueError("native mixing-ratio matrix is not cell aligned")
    for values, label in (
        (density_all, "number density"),
        (pressure_all, "pressure"),
        (temperature_all, "temperature"),
        (eddy_all, "eddy diffusion"),
    ):
        if values.shape != altitude_all.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"native {label} must be finite and cell aligned")
    if np.any(density_all <= 0.0) or np.any(pressure_all <= 0.0) or np.any(temperature_all <= 0.0) or np.any(eddy_all < 0.0):
        raise ValueError("native physical fields are outside their allowed domain")

    edges = _cell_edges_from_centers(altitude_all)
    selected = np.flatnonzero(
        (edges[:-1] >= lower_altitude_km - 1.0e-12)
        & (edges[1:] <= upper_altitude_km + 1.0e-12)
    )
    if selected.size < 2:
        raise ValueError("requested isotope-kernel domain contains too few cells")
    if not np.isclose(edges[selected[0]], lower_altitude_km, atol=1.0e-10) or not np.isclose(edges[selected[-1] + 1], upper_altitude_km, atol=1.0e-10):
        raise ValueError("isotope-kernel bounds must coincide with native cell edges")

    index = {name: position for position, name in enumerate(species_names)}
    missing = sorted({"O", "O1D", "O2", "CO2", "O3", "N2"}.difference(index))
    if missing:
        raise KeyError(f"native artifact lacks parent species: {missing}")
    parent = {
        name: _nonnegative_parent(name, mixing_all[index[name], selected])
        for name in ("O", "O1D", "O2", "CO2", "O3", "N2")
    }
    altitude = altitude_all[selected]
    density = density_all[selected]
    pressure_pa = 0.1 * pressure_all[selected]
    temperature = temperature_all[selected]
    cells = tuple(
        VerticalCell(
            lower_altitude_km=float(edges[cell]),
            upper_altitude_km=float(edges[cell + 1]),
            air_moles=(
                float(density_all[cell])
                * _shell_volume_cm3(float(edges[cell]), float(edges[cell + 1]), radius_cm)
                / AVOGADRO_PER_MOL
            ),
            number_density_molecules_cm3=float(density_all[cell]),
            temperature_k=float(temperature_all[cell]),
            pressure_center_bar=float(pressure_all[cell]) / 1.0e6,
            eddy_diffusivity_cm2_per_s=float(eddy_all[cell]),
        )
        for cell in selected
    )
    checksum = _sha256(path)
    provenance = (
        f"native Photochem {version} end member {path.name}; SHA-256 {checksum}; "
        f"pO2={po2_pal:g} PAL; pCO2={pco2_ppm:g} ppm"
    )
    profile = ValidatedVerticalProfile(
        name=f"Photochem end member {po2_pal:g} PAL, {pco2_ppm:g} ppm",
        cells=cells,
        atmospheric_state_source=provenance,
        eddy_diffusivity_source=f"{provenance}; native Kzz",
    )
    transport = eddy_diffusion_column(profile, planetary_radius_cm=radius_cm)
    system = GriddedSpeciesSystem(
        species_names=ATMOSPHERIC_OXYGEN_SPECIES,
        air_moles=transport.air_moles,
        inventory_transport_matrix_per_year=(
            transport_scale * transport.transport_matrix_per_year()
        ),
        source=f"{provenance}; finite-volume Kzz transport scale={transport_scale:g}",
    )
    rate_fields = modern_r1_r7_rate_fields(
        temperature,
        density,
        o2_mixing_ratio=parent["O2"],
        n2_mixing_ratio=parent["N2"],
        co2_mixing_ratio=parent["CO2"],
    )
    photolysis, omitted_ratio = _native_photolysis_fields(
        equations, photolysis_all, selected
    )
    reactions = full_young_r1_r7_grid_reactions(
        rate_fields, photolysis, a_mif=a_mif
    )
    chemistry = bind_local_reaction_operator(
        species_names=system.species_names,
        air_moles=system.air_moles,
        pressure_pa=pressure_pa,
        temperature_k=temperature,
        reactions=reactions,
    )
    throughput = bind_local_reaction_throughput_operator(
        species_names=system.species_names,
        air_moles=system.air_moles,
        pressure_pa=pressure_pa,
        temperature_k=temperature,
        reactions=reactions,
    )
    chemistry, throughput = _fixed_parent_ratio_corrected_operators(
        system=system,
        chemistry=chemistry,
        throughput=throughput,
    )

    tiny = np.finfo(float).tiny
    mixing_ratio = {
        "O": parent["O"],
        "O17": parent["O"] * o2_composition.ratio17,
        "O18": parent["O"] * o2_composition.ratio18,
        "O1D": parent["O1D"],
        "O17_1D": parent["O1D"] * o2_composition.ratio17,
        "O18_1D": parent["O1D"] * o2_composition.ratio18,
        "O2": parent["O2"],
        "O17O": _rare_mixing_ratio(parent["O2"], o2_composition, isotope=17, represented_sites=2.0),
        "O18O": _rare_mixing_ratio(parent["O2"], o2_composition, isotope=18, represented_sites=2.0),
        "CO2": parent["CO2"],
        "CO17O": _rare_mixing_ratio(parent["CO2"], co2_composition, isotope=17, represented_sites=1.0),
        "CO18O": _rare_mixing_ratio(parent["CO2"], co2_composition, isotope=18, represented_sites=1.0),
        "O3": parent["O3"],
        "OO17O": _rare_mixing_ratio(parent["O3"], o2_composition, isotope=17, represented_sites=3.0),
        "OO18O": _rare_mixing_ratio(parent["O3"], o2_composition, isotope=18, represented_sites=3.0),
    }
    inventory = np.asarray(
        [np.maximum(mixing_ratio[name], tiny) * system.air_moles for name in system.species_names]
    )
    fixed = np.zeros_like(inventory, dtype=bool)
    fixed[:, 0] = True
    isotope_index = {name: position for position, name in enumerate(system.species_names)}
    for name in ("O", "O1D", "O2", "O17O", "O18O", "CO2", "O3"):
        fixed[isotope_index[name], :] = True
    column = FixedBoundaryIsotopeColumn(
        species_system=system,
        local_chemistry=chemistry,
        fixed_mask=fixed,
        prescribed_inventory_moles=inventory,
        source=(
            f"{provenance}; O2 isotopes: {o2_composition.source}; CO2 lower-boundary "
            f"isotopes: {co2_composition.source}; source-derived Sander et al. (2006) "
            f"R7(T); Young ozone-formation a_MIF={a_mif:g}; exact fixed-parent "
            "isotope-ratio correction; no fitted "
            "throughput multiplier"
        ),
        local_chemistry_throughput=throughput,
    )
    return PhotochemEndmemberIsotopeBuild(
        column=column,
        profile=profile,
        altitude_km=altitude,
        pressure_pa=pressure_pa,
        temperature_k=temperature,
        number_density_molecules_cm3=density,
        parent_mixing_ratios=parent,
        reactions=reactions,
        r7_reactions=tuple(reaction for reaction in reactions if reaction.key.startswith("R7")),
        pO2_PAL=po2_pal,
        pCO2_ppm=pco2_ppm,
        a_mif=float(a_mif),
        artifact_sha256=checksum,
        omitted_o2_o1d_maximum_ratio=omitted_ratio,
        source=column.source,
    )
