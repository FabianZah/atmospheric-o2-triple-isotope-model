"""Reproducible vertical O2/O3 photolysis reference profiles.

The implemented direct-beam profile uses pinned Photochem 0.6.7 Modern Earth
inputs and matching photochem_clima_data 0.3.0 cross sections. It is a
transparent benchmark, not a replacement for Photochem's two-stream solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np


PLANCK_J_S = 6.62607015e-34
LIGHT_M_PER_S = 299_792_458.0
PINNED_SHA256 = {
    "H2O.h5": "37b799676e6d17d0c03fac4e9aeb7be8f296bbb3df29fa7a5414676765515ddf",
    "H2O2.h5": "75fd222543ba2456b8ce30db506c48fdad1a5cbc6f197a81d74fc142900a1448",
    "HO2.h5": "95da9c11586e721e797ae8877ed4e57e323bebbd45eb47f700a272beda58f2cb",
    "HNO3.h5": "e23bc05ccb168c31ecce6a62c7752bd80ccde1bd6a9f84edc81dfa7b7b3dfb4f",
    "NO3.h5": "d3625e2b35837dedbd47812ed7638bc202fd71dc31b4e6b0e9da5f9ab0495764",
    "NO3_Sander2006_Table4_16.csv": "938e6d6cab6187e6c081ddb04c45b7936cfb9b9c71661cba2edcbf7756a9795f",
    "HOCl.h5": "c010899fbca37dabd756e1e7dc04b5a826185e874aaa69fcc598db9870228cad",
    "NO2.h5": "7efb95f6511f4f601f04981eee4fbc88d979f0642fd34db1fb7188e52a214c7b",
    "O2.h5": "5ec49e334be8af964d3e2defdb4347f6d64032c3b3cb837bd4683c65d3541e68",
    "O3.h5": "29b3ea6ff30d757679b82476c1338412171140f95ae085ae766688f444a6a7bb",
    "OH.h5": "06357e854899fc03dd15beabbebe54dbe3154c903411ef5f85bf836d698130d7",
    "CH3OH.h5": "61860faa67e44f22ef57ea889031d21c26ee31004689c7dfc2b382997b2705fe",
    "Sun_now.txt": "0257f12135a27022b935f2af10e1fa480907987782b0f96580f755cfcf92b7de",
    "atmosphere.txt": "d2a3d6cfef512ade9f1e6dc80217f05e1c0f116f2b77205cbdef789c8699b2be",
}
PHOTOLYSIS_PROFILE_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class PhotolysisProfile:
    altitude_km: np.ndarray
    j_o2_to_o_o_per_s: np.ndarray
    j_o2_to_o_o1d_per_s: np.ndarray
    j_o3_to_o2_o_per_s: np.ndarray
    j_o3_to_o2_o1d_per_s: np.ndarray
    solar_zenith_degrees: float
    diurnal_factor: float
    method: str
    source: str

    def __post_init__(self) -> None:
        altitude = np.asarray(self.altitude_km, dtype=float)
        fields = (
            self.j_o2_to_o_o_per_s,
            self.j_o2_to_o_o1d_per_s,
            self.j_o3_to_o2_o_per_s,
            self.j_o3_to_o2_o1d_per_s,
        )
        if (
            altitude.ndim != 1
            or len(altitude) < 2
            or not np.all(np.diff(altitude) > 0.0)
        ):
            raise ValueError("altitude must be a strictly increasing vector")
        if any(np.asarray(field).shape != altitude.shape for field in fields):
            raise ValueError("photolysis fields must match altitude")
        if any(
            np.any(np.asarray(field) < 0.0) or not np.all(np.isfinite(field))
            for field in fields
        ):
            raise ValueError("photolysis frequencies must be finite and non-negative")

    def interpolate(self, altitude_km: np.ndarray) -> dict[str, np.ndarray]:
        altitude = np.asarray(altitude_km, dtype=float)
        return {
            "j_r1_o2_to_o_o_per_s": np.interp(
                altitude, self.altitude_km, self.j_o2_to_o_o_per_s
            ),
            "j_o2_to_o_o1d_per_s": np.interp(
                altitude, self.altitude_km, self.j_o2_to_o_o1d_per_s
            ),
            "j_r3a_o3_to_o2_o_per_s": np.interp(
                altitude, self.altitude_km, self.j_o3_to_o2_o_per_s
            ),
            "j_r3f_o3_to_o2_o1d_per_s": np.interp(
                altitude, self.altitude_km, self.j_o3_to_o2_o1d_per_s
            ),
        }


def save_photolysis_profile(
    profile: PhotolysisProfile,
    path: str | Path,
) -> None:
    """Serialize one derived profile without pickles or hidden model state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray(PHOTOLYSIS_PROFILE_CACHE_SCHEMA, dtype=np.int64),
        altitude_km=np.asarray(profile.altitude_km, dtype=float),
        j_o2_to_o_o_per_s=np.asarray(profile.j_o2_to_o_o_per_s, dtype=float),
        j_o2_to_o_o1d_per_s=np.asarray(profile.j_o2_to_o_o1d_per_s, dtype=float),
        j_o3_to_o2_o_per_s=np.asarray(profile.j_o3_to_o2_o_per_s, dtype=float),
        j_o3_to_o2_o1d_per_s=np.asarray(profile.j_o3_to_o2_o1d_per_s, dtype=float),
        solar_zenith_degrees=np.asarray(profile.solar_zenith_degrees, dtype=float),
        diurnal_factor=np.asarray(profile.diurnal_factor, dtype=float),
        method=np.asarray(profile.method),
        source=np.asarray(profile.source),
    )


def load_photolysis_profile(path: str | Path) -> PhotolysisProfile:
    """Load and validate a profile written by :func:`save_photolysis_profile`."""

    with np.load(Path(path), allow_pickle=False) as cached:
        schema = int(np.asarray(cached["schema_version"]).item())
        if schema != PHOTOLYSIS_PROFILE_CACHE_SCHEMA:
            raise ValueError(
                f"unsupported photolysis cache schema {schema}; "
                f"expected {PHOTOLYSIS_PROFILE_CACHE_SCHEMA}"
            )
        return PhotolysisProfile(
            altitude_km=np.asarray(cached["altitude_km"], dtype=float),
            j_o2_to_o_o_per_s=np.asarray(cached["j_o2_to_o_o_per_s"], dtype=float),
            j_o2_to_o_o1d_per_s=np.asarray(cached["j_o2_to_o_o1d_per_s"], dtype=float),
            j_o3_to_o2_o_per_s=np.asarray(cached["j_o3_to_o2_o_per_s"], dtype=float),
            j_o3_to_o2_o1d_per_s=np.asarray(cached["j_o3_to_o2_o1d_per_s"], dtype=float),
            solar_zenith_degrees=float(
                np.asarray(cached["solar_zenith_degrees"]).item()
            ),
            diurnal_factor=float(np.asarray(cached["diurnal_factor"]).item()),
            method=str(np.asarray(cached["method"]).item()),
            source=str(np.asarray(cached["source"]).item()),
        )


def _verify(path: Path) -> None:
    actual = sha256(path.read_bytes()).hexdigest()
    expected = PINNED_SHA256.get(path.name)
    if expected is None or actual != expected:
        raise ValueError(f"checksum mismatch for {path.name}: {actual}")


def _load_solar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    wavelength = table[:, 0]
    edges = np.empty(len(wavelength) + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    photons = (
        table[:, 1]
        * 1.0e-3
        * wavelength
        * 1.0e-9
        / (PLANCK_J_S * LIGHT_M_PER_S)
        / 1.0e4
    )
    return wavelength, np.diff(edges), photons


def _load_xs(
    path: Path, wavelength: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path) as data:
        native = np.asarray(data["wavelengths"][:], dtype=float)
        absorption = np.interp(
            wavelength, native, data["photoabsorption"][:], left=0.0, right=0.0
        )
        dissociation = np.interp(
            wavelength, native, data["photodissociation"][:], left=0.0, right=0.0
        )
        group = data["photodissociation-qy"]
        qy_wavelength = np.asarray(group["wavelengths"][:], dtype=float)
        channels = {}
        for key in group.keys():
            if key == "wavelengths":
                continue
            values = np.asarray(group[key][:], dtype=float)
            qy = np.interp(
                wavelength, qy_wavelength, values, left=values[0], right=values[-1]
            )
            channels[key] = dissociation * qy
    return absorption, channels


def matsumi_kawasaki_o1d_quantum_yield(
    wavelength_nm: np.ndarray | float,
    temperature_k: np.ndarray | float,
    *,
    young_cutoff_convention: bool = True,
) -> np.ndarray:
    """O3 -> O(1D) yield from Matsumi and Kawasaki (2003), Eq. 13.

    The recommendation is 0.90 below 306 nm, Eq. 13 from 306 to 328 nm,
    and 0.08 from 328 to 340 nm. Young et al. (2014) explicitly set the
    yield to zero above 340 nm; disabling ``young_cutoff_convention`` retains
    the 0.08 spin-forbidden tail to the 411 nm energetic threshold.
    """
    wavelength = np.asarray(wavelength_nm, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength must be finite and positive")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature must be finite and positive")
    wavelength, temperature = np.broadcast_arrays(wavelength, temperature)

    x1, x2, x3 = 304.225, 314.957, 310.737
    width1, width2, width3 = 5.576, 6.601, 2.187
    a1, a2, a3 = 0.8036, 8.9061, 0.1192
    q1 = np.ones_like(temperature)
    q2 = np.exp(-825.518 / (0.695 * temperature))
    transition = (
        q1 / (q1 + q2) * a1 * np.exp(-(((x1 - wavelength) / width1) ** 4))
        + q2
        / (q1 + q2)
        * a2
        * (temperature / 300.0) ** 2
        * np.exp(-(((x2 - wavelength) / width2) ** 2))
        + a3
        * (temperature / 300.0) ** 1.5
        * np.exp(-(((x3 - wavelength) / width3) ** 2))
        + 0.0765
    )
    upper_limit = 340.0 if young_cutoff_convention else 411.0
    yield_o1d = np.where(
        wavelength < 306.0,
        0.90,
        np.where(
            wavelength <= 328.0,
            transition,
            np.where(wavelength <= upper_limit, 0.08, 0.0),
        ),
    )
    return np.clip(yield_o1d, 0.0, 1.0)


def load_direct_beam_modern_earth_photolysis(
    external_data_directory: Path,
    *,
    solar_zenith_degrees: float = 60.0,
    diurnal_factor: float = 0.5,
    verify_checksums: bool = True,
) -> PhotolysisProfile:
    """Compute a 1-km profile with O2/O3 absorption and no scattering."""

    external = Path(external_data_directory)
    source = (
        external
        / "photochem-v0.6.7-source"
        / "photochem-0.6.7"
        / "examples"
        / "ModernEarth"
    )
    xs = external / "photochem-data-0.3.0" / "xsections"
    atmosphere_path = source / "atmosphere.txt"
    solar_path = source / "Sun_now.txt"
    o2_path = xs / "O2.h5"
    o3_path = xs / "O3.h5"
    paths = (atmosphere_path, solar_path, o2_path, o3_path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing pinned photolysis inputs: {missing}")
    if verify_checksums:
        for path in paths:
            _verify(path)
    if not 0.0 <= solar_zenith_degrees < 90.0 or not 0.0 < diurnal_factor <= 1.0:
        raise ValueError("invalid solar zenith angle or diurnal factor")

    atmosphere = np.genfromtxt(atmosphere_path, names=True)
    altitude = np.asarray(atmosphere["alt"], dtype=float)
    wavelength, width, photon_flux = _load_solar(solar_path)
    absorption_o2, channels_o2 = _load_xs(o2_path, wavelength)
    absorption_o3, channels_o3 = _load_xs(o3_path, wavelength)
    dz_cm = float(np.diff(altitude)[0] * 1.0e5)
    density_o2 = atmosphere["den"] * atmosphere["O2"]
    density_o3 = atmosphere["den"] * atmosphere["O3"]
    column_o2 = (
        np.flip(np.cumsum(np.flip(density_o2 * dz_cm))) - 0.5 * density_o2 * dz_cm
    )
    column_o3 = (
        np.flip(np.cumsum(np.flip(density_o3 * dz_cm))) - 0.5 * density_o3 * dz_cm
    )
    optical_depth = (
        column_o2[:, None] * absorption_o2 + column_o3[:, None] * absorption_o3
    )
    actinic = diurnal_factor * np.exp(
        -optical_depth / np.cos(np.deg2rad(solar_zenith_degrees))
    )
    photons_per_bin = photon_flux * width

    def integrate(channel: np.ndarray) -> np.ndarray:
        return np.sum(actinic * photons_per_bin * channel, axis=1)

    return PhotolysisProfile(
        altitude,
        integrate(channels_o2["O2 + hv => O + O"]),
        integrate(channels_o2["O2 + hv => O + O1D"]),
        integrate(channels_o3["O3 + hv => O + O2"]),
        integrate(channels_o3["O3 + hv => O1D + O2"]),
        float(solar_zenith_degrees),
        float(diurnal_factor),
        "direct-beam Beer-Lambert; O2/O3 absorption; no multiple scattering",
        "Photochem 0.6.7 ModernEarth plus photochem_clima_data 0.3.0",
    )
