"""Photochem 0.6.7 two-stream photolysis reference implemented in Python.

The algebra ports ``photochem_radtran.f90``. This diagnostic includes O2/O3
absorption and gas Rayleigh scattering, but not particles or minor absorbers.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from modern_photolysis import (
    PINNED_SHA256,
    PhotolysisProfile,
    _load_solar,
    _load_xs,
    _verify,
    matsumi_kawasaki_o1d_quantum_yield,
)


def rayleigh_vardavas_cm2(wavelength_nm, *, a, b, delta):
    wavelength_um = np.asarray(wavelength_nm, dtype=float) * 1.0e-3
    depolarization = (6.0 + 3.0 * delta) / (6.0 - 7.0 * delta)
    return (
        4.577e-21
        * depolarization
        * (a * (1.0 + b / wavelength_um**2)) ** 2
        / wavelength_um**4
    )


def two_stream_mean_intensity(
    tau, single_scattering_albedo, *, solar_zenith_cosine, surface_albedo
):
    """Port Photochem's two_stream for top-to-surface layer arrays."""
    tau = np.asarray(tau, dtype=float).copy()
    w0 = np.asarray(single_scattering_albedo, dtype=float).copy()
    if tau.ndim != 1 or w0.shape != tau.shape or np.any(tau <= 0.0):
        raise ValueError("tau and albedo must be positive layer vectors")
    if np.any(w0 < 0.0) or np.any(w0 >= 1.0):
        raise ValueError("single-scattering albedo must be in [0, 1)")
    u0 = float(solar_zenith_cosine)
    if not 0.0 < u0 <= 1.0 or not 0.0 <= surface_albedo <= 1.0:
        raise ValueError("invalid solar geometry or surface albedo")

    n = len(tau)
    root3 = np.sqrt(3.0)
    gam1 = root3 * (2.0 - w0) / 2.0
    gam2 = root3 * w0 / 2.0
    gam3 = np.full(n, 0.5)
    gam4 = np.full(n, 0.5)
    u1 = 1.0 / root3
    lam = np.sqrt(np.maximum(gam1**2 - gam2**2, 0.0))
    cap_gam = gam2 / (gam1 + lam)
    exponential = np.exp(-lam * tau)
    e1 = 1.0 + cap_gam * exponential
    e2 = 1.0 - cap_gam * exponential
    e3 = cap_gam + exponential
    e4 = cap_gam - exponential
    cumulative = np.concatenate(([0.0], np.cumsum(tau)))
    direct = np.empty(n + 1)
    direct[0] = u0
    cp0, cpb, cm0, cmb = (np.empty(n) for _ in range(4))
    for i in range(n):
        facp = w0[i] * ((gam1[i] - 1.0 / u0) * gam3[i] + gam4[i] * gam2[i])
        facm = w0[i] * ((gam1[i] + 1.0 / u0) * gam4[i] + gam2[i] * gam3[i])
        et0 = np.exp(-cumulative[i] / u0)
        etb = np.exp(-cumulative[i + 1] / u0)
        denominator = lam[i] ** 2 - 1.0 / u0**2
        if abs(denominator) < 1.0e-14:
            denominator = -1.0e-14
        direct[i + 1] = u0 * etb
        cp0[i], cpb[i] = et0 * facp / denominator, etb * facp / denominator
        cm0[i], cmb[i] = et0 * facm / denominator, etb * facm / denominator

    lower, diagonal, upper, rhs = (np.zeros(2 * n) for _ in range(4))
    diagonal[0], upper[0], rhs[0] = e1[0], -e2[0], -cm0[0]
    for i in range(n - 1):
        odd = 2 * i + 2
        lower[odd] = e2[i] * e3[i] - e4[i] * e1[i]
        diagonal[odd] = e1[i] * e1[i + 1] - e3[i] * e3[i + 1]
        upper[odd] = e3[i] * e4[i + 1] - e1[i] * e2[i + 1]
        rhs[odd] = e3[i] * (cp0[i + 1] - cpb[i]) + e1[i] * (cmb[i] - cm0[i + 1])
        even = 2 * i + 1
        lower[even] = e2[i + 1] * e1[i] - e3[i] * e4[i + 1]
        diagonal[even] = e2[i] * e2[i + 1] - e4[i] * e4[i + 1]
        upper[even] = e1[i + 1] * e4[i + 1] - e2[i + 1] * e3[i + 1]
        rhs[even] = e2[i + 1] * (cp0[i + 1] - cpb[i]) - e4[i + 1] * (
            cm0[i + 1] - cmb[i]
        )
    lower[-1] = e1[-1] - surface_albedo * e3[-1]
    diagonal[-1] = e2[-1] - surface_albedo * e4[-1]
    rhs[-1] = surface_albedo * direct[-1] - cpb[-1] + surface_albedo * cmb[-1]

    upper[0], rhs[0] = upper[0] / diagonal[0], rhs[0] / diagonal[0]
    for i in range(1, 2 * n - 1):
        pivot = diagonal[i] - lower[i] * upper[i - 1]
        upper[i] /= pivot
        rhs[i] = (rhs[i] - lower[i] * rhs[i - 1]) / pivot
    rhs[-1] = (rhs[-1] - lower[-1] * rhs[-2]) / (diagonal[-1] - lower[-1] * upper[-2])
    for i in range(2 * n - 2, -1, -1):
        rhs[i] -= upper[i] * rhs[i + 1]

    y1, y2 = rhs[0::2], rhs[1::2]
    mean = np.empty(n + 1)
    mean[0] = (y1[0] * e3[0] - y2[0] * e4[0] + cp0[0]) / u1 + direct[0] / u0
    for i in range(n):
        mean[i + 1] = (
            y1[i] * (e1[i] + e3[i]) + y2[i] * (e2[i] + e4[i]) + cpb[i] + cmb[i]
        ) / u1 + direct[i + 1] / u0
    return np.abs(mean)


_TWO_STREAM_SPECTRAL_CACHE = {}


@dataclass(frozen=True)
class TracePhotolysisProfile:
    altitude_km: np.ndarray
    rate_per_s: np.ndarray
    species: str
    reaction: str
    solar_zenith_degrees: float
    diurnal_factor: float
    method: str
    source: str

    def __post_init__(self) -> None:
        altitude = np.asarray(self.altitude_km, dtype=float)
        rate = np.asarray(self.rate_per_s, dtype=float)
        if (
            altitude.ndim != 1
            or rate.shape != altitude.shape
            or not np.all(np.diff(altitude) > 0.0)
            or np.any(rate < 0.0)
            or not np.all(np.isfinite(rate))
        ):
            raise ValueError("trace photolysis profile must be finite and grid aligned")

    def interpolate(self, altitude_km: np.ndarray) -> np.ndarray:
        altitude = np.asarray(altitude_km, dtype=float)
        return np.interp(
            altitude,
            self.altitude_km,
            self.rate_per_s,
            left=float(self.rate_per_s[0]),
            right=float(self.rate_per_s[-1]),
        )


def _cached_two_stream_spectral_state(
    atmosphere_path: Path,
    solar_path: Path,
    o2_path: Path,
    o3_path: Path,
    rayleigh_path: Path,
    additional_absorber_directory: Path | None = None,
    *,
    solar_zenith_degrees: float,
    surface_albedo: float,
):
    """Cache geometry-dependent radiative transfer within one Python process."""
    key = (
        str(atmosphere_path.resolve()),
        str(solar_path.resolve()),
        str(o2_path.resolve()),
        str(o3_path.resolve()),
        (
            None
            if additional_absorber_directory is None
            else str(additional_absorber_directory.resolve())
        ),
        float(solar_zenith_degrees),
        float(surface_albedo),
    )
    cached = _TWO_STREAM_SPECTRAL_CACHE.get(key)
    if cached is not None:
        return cached

    atmosphere = np.genfromtxt(atmosphere_path, names=True)
    altitude = np.asarray(atmosphere["alt"], dtype=float)
    wavelength, width, photon_flux = _load_solar(solar_path)
    absorption_o2, channels_o2 = _load_xs(o2_path, wavelength)
    absorption_o3, channels_o3 = _load_xs(o3_path, wavelength)
    rayleigh_data = yaml.safe_load(rayleigh_path.read_text(encoding="utf-8"))
    dz_cm = float(np.diff(altitude)[0] * 1.0e5)
    density = np.asarray(atmosphere["den"], dtype=float)
    absorption_layer = (
        density[:, None] * atmosphere["O2"][:, None] * absorption_o2
        + density[:, None] * atmosphere["O3"][:, None] * absorption_o3
    ) * dz_cm
    if additional_absorber_directory is not None:
        absorber_directory = Path(additional_absorber_directory)
        for species in atmosphere.dtype.names or ():
            if species in {"alt", "den", "temp", "O2", "O3"}:
                continue
            cross_section_path = absorber_directory / f"{species}.h5"
            if not cross_section_path.is_file():
                continue
            absorption, _channels = _load_xs(cross_section_path, wavelength)
            absorption_layer += (
                density[:, None]
                * np.asarray(atmosphere[species], dtype=float)[:, None]
                * absorption[None, :]
                * dz_cm
            )
    scattering_layer = np.zeros_like(absorption_layer)
    for species, record in rayleigh_data.items():
        if species not in atmosphere.dtype.names:
            continue
        parameters = record["data"]
        cross_section = rayleigh_vardavas_cm2(
            wavelength,
            a=float(parameters["A"]),
            b=float(parameters["B"]),
            delta=float(parameters["Delta"]),
        )
        scattering_layer += (
            density[:, None] * atmosphere[species][:, None] * cross_section * dz_cm
        )

    relevant = (
        (np.sum(absorption_layer, axis=0) > 1.0e-16)
        | (np.sum(scattering_layer, axis=0) > 1.0e-16)
    )
    u0 = float(np.cos(np.deg2rad(solar_zenith_degrees)))
    actinic = np.zeros_like(absorption_layer)
    for spectral_index in np.flatnonzero(relevant):
        total_top_down = np.maximum(
            (absorption_layer[:, spectral_index] + scattering_layer[:, spectral_index])[
                ::-1
            ],
            1.0e-300,
        )
        scattering_top_down = scattering_layer[::-1, spectral_index]
        w0 = np.minimum(0.99999, scattering_top_down / total_top_down)
        edges = two_stream_mean_intensity(
            total_top_down,
            w0,
            solar_zenith_cosine=u0,
            surface_albedo=surface_albedo,
        )
        actinic[:, spectral_index] = np.sqrt(edges[:-1] * edges[1:])[::-1]

    cached = (
        atmosphere,
        altitude,
        wavelength,
        width,
        photon_flux,
        channels_o2,
        channels_o3,
        actinic,
    )
    _TWO_STREAM_SPECTRAL_CACHE[key] = cached
    return cached


def load_two_stream_modern_earth_photolysis(
    external_data_directory: Path,
    *,
    solar_zenith_degrees: float = 60.0,
    diurnal_factor: float = 0.5,
    surface_albedo: float = 0.25,
    verify_checksums: bool = True,
    ozone_quantum_yield_convention: str = "pinned",
    ozone_quantum_yield_temperature_k: float | None = None,
) -> PhotolysisProfile:
    """Compute O2/O3 photolysis with Photochem's two-stream equations."""
    external = Path(external_data_directory)
    source = (
        external
        / "photochem-v0.6.7-source"
        / "photochem-0.6.7"
        / "examples"
        / "ModernEarth"
    )
    data = external / "photochem-data-0.3.0"
    atmosphere_path = source / "atmosphere.txt"
    solar_path = source / "Sun_now.txt"
    o2_path = data / "xsections" / "O2.h5"
    o3_path = data / "xsections" / "O3.h5"
    if verify_checksums:
        for path in (atmosphere_path, solar_path, o2_path, o3_path):
            _verify(path)

    valid_yield_conventions = {"pinned", "matsumi_kawasaki_2003_young_cutoff"}
    if ozone_quantum_yield_convention not in valid_yield_conventions:
        raise ValueError(
            "ozone_quantum_yield_convention must be one of "
            f"{sorted(valid_yield_conventions)}"
        )
    if ozone_quantum_yield_temperature_k is not None and (
        not np.isfinite(ozone_quantum_yield_temperature_k)
        or ozone_quantum_yield_temperature_k <= 0.0
    ):
        raise ValueError("ozone quantum-yield temperature must be positive")
    (
        atmosphere,
        altitude,
        wavelength,
        width,
        photon_flux,
        channels_o2,
        channels_o3,
        actinic,
    ) = _cached_two_stream_spectral_state(
        atmosphere_path,
        solar_path,
        o2_path,
        o3_path,
        data / "rayleigh" / "rayleigh.yaml",
        solar_zenith_degrees=solar_zenith_degrees,
        surface_albedo=surface_albedo,
    )

    photons_per_bin = photon_flux * width * diurnal_factor

    def integrate(channel):
        return np.sum(actinic * photons_per_bin * channel, axis=1)

    if ozone_quantum_yield_convention == "pinned":
        ozone_o_channel = channels_o3["O3 + hv => O + O2"]
        ozone_o1d_channel = channels_o3["O3 + hv => O1D + O2"]
        yield_source = "photochem_clima_data 0.3.0"
    else:
        ozone_dissociation = (
            channels_o3["O3 + hv => O + O2"] + channels_o3["O3 + hv => O1D + O2"]
        )
        if ozone_quantum_yield_temperature_k is None:
            yield_temperature = np.asarray(atmosphere["temp"], dtype=float)
        else:
            yield_temperature = np.full_like(
                altitude, float(ozone_quantum_yield_temperature_k)
            )
        yield_o1d = matsumi_kawasaki_o1d_quantum_yield(
            wavelength[None, :],
            yield_temperature[:, None],
            young_cutoff_convention=True,
        )
        ozone_o1d_channel = ozone_dissociation[None, :] * yield_o1d
        ozone_o_channel = ozone_dissociation[None, :] * (1.0 - yield_o1d)
        yield_source = (
            "Matsumi and Kawasaki (2003), Eq. 13 and Table 3; "
            "Young et al. (2014) >340 nm cutoff"
        )

    return PhotolysisProfile(
        altitude,
        integrate(channels_o2["O2 + hv => O + O"]),
        integrate(channels_o2["O2 + hv => O + O1D"]),
        integrate(ozone_o_channel),
        integrate(ozone_o1d_channel),
        float(solar_zenith_degrees),
        float(diurnal_factor),
        "Photochem 0.6.7 delta-Eddington two-stream port; gas Rayleigh; no particles",
        "Photochem 0.6.7 source plus " + yield_source,
    )


def load_two_stream_trace_photolysis(
    external_data_directory: Path,
    *,
    species: str,
    reaction: str,
    solar_zenith_degrees: float = 60.0,
    diurnal_factor: float = 0.5,
    surface_albedo: float = 0.25,
    verify_checksums: bool = True,
    background_opacity_convention: str = "o2_o3_rayleigh",
) -> TracePhotolysisProfile:
    """Evaluate an optically thin trace-species channel in the O2/O3 field."""
    if not species or not reaction:
        raise ValueError("trace photolysis requires species and reaction names")
    if not 0.0 < diurnal_factor <= 1.0:
        raise ValueError("diurnal factor must be in (0, 1]")
    valid_opacity = {"o2_o3_rayleigh", "all_gases_rayleigh"}
    if background_opacity_convention not in valid_opacity:
        raise ValueError(
            "background_opacity_convention must be one of "
            f"{sorted(valid_opacity)}"
        )

    external = Path(external_data_directory)
    source = (
        external
        / "photochem-v0.6.7-source"
        / "photochem-0.6.7"
        / "examples"
        / "ModernEarth"
    )
    data = external / "photochem-data-0.3.0"
    atmosphere_path = source / "atmosphere.txt"
    solar_path = source / "Sun_now.txt"
    o2_path = data / "xsections" / "O2.h5"
    o3_path = data / "xsections" / "O3.h5"
    trace_path = data / "xsections" / f"{species}.h5"
    paths = (atmosphere_path, solar_path, o2_path, o3_path, trace_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned trace-photolysis inputs: {missing}")
    if verify_checksums:
        for path in paths:
            _verify(path)

    (
        _atmosphere,
        altitude,
        wavelength,
        width,
        photon_flux,
        _channels_o2,
        _channels_o3,
        actinic,
    ) = _cached_two_stream_spectral_state(
        atmosphere_path,
        solar_path,
        o2_path,
        o3_path,
        data / "rayleigh" / "rayleigh.yaml",
        (
            data / "xsections"
            if background_opacity_convention == "all_gases_rayleigh"
            else None
        ),
        solar_zenith_degrees=solar_zenith_degrees,
        surface_albedo=surface_albedo,
    )
    _absorption, channels = _load_xs(trace_path, wavelength)
    if reaction not in channels:
        raise KeyError(
            f"photolysis channel {reaction!r} is absent from {trace_path.name}"
        )
    photons_per_bin = photon_flux * width * float(diurnal_factor)
    rate = np.sum(actinic * photons_per_bin * channels[reaction], axis=1)
    return TracePhotolysisProfile(
        altitude_km=altitude,
        rate_per_s=rate,
        species=species,
        reaction=reaction,
        solar_zenith_degrees=float(solar_zenith_degrees),
        diurnal_factor=float(diurnal_factor),
        method=(
            "Photochem 0.6.7 delta-Eddington two-stream port; trace absorber "
            + (
                "uses all available pinned gas absorption plus Rayleigh"
                if background_opacity_convention == "all_gases_rayleigh"
                else "does not alter the O2/O3/Rayleigh radiation field"
            )
        ),
        source=(
            f"photochem_clima_data 0.3.0 {trace_path.name}; SHA-256 "
            f"{PINNED_SHA256[trace_path.name] if verify_checksums else 'not checked'}"
        ),
    )


def sander_2006_no3_quantum_yields(
    wavelength_nm: np.ndarray,
    temperature_k: np.ndarray,
    table_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate Sander et al. (2006) Table 4-16 NO3 product yields.

    The output arrays have shape ``(temperature, wavelength)``. Sander et al.
    prescribe phi(NO2 + O)=1 and phi(NO + O2)=0 below 585 nm, and zero for
    both channels above 640 nm. Values are linearly interpolated in wavelength
    and temperature between the tabulated 190, 230, and 298 K evaluations.
    Temperatures outside that measured interval use the nearest table limit.
    """

    wavelength = np.asarray(wavelength_nm, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    if wavelength.ndim != 1 or temperature.ndim != 1:
        raise ValueError("NO3 wavelength and temperature must be vectors")
    if (
        np.any(wavelength <= 0.0)
        or np.any(temperature <= 0.0)
        or not np.all(np.isfinite(wavelength))
        or not np.all(np.isfinite(temperature))
    ):
        raise ValueError("NO3 photolysis grid must be finite and positive")
    table = np.genfromtxt(table_path, names=True, delimiter=",", dtype=float)
    required = {
        "wavelength_nm",
        "phi_no_298k",
        "phi_no_230k",
        "phi_no_190k",
        "phi_no2_298k",
        "phi_no2_230k",
        "phi_no2_190k",
    }
    if table.dtype.names is None or not required.issubset(table.dtype.names):
        raise ValueError("Sander NO3 quantum-yield table has an invalid schema")
    table_wavelength = np.asarray(table["wavelength_nm"], dtype=float)
    table_temperatures = np.asarray([190.0, 230.0, 298.0])

    def channel(prefix: str, *, shortwave_value: float) -> np.ndarray:
        spectra = np.asarray(
            [
                np.interp(
                    wavelength,
                    table_wavelength,
                    np.asarray(table[f"{prefix}_{suffix}"], dtype=float),
                    left=shortwave_value,
                    right=0.0,
                )
                for suffix in ("190k", "230k", "298k")
            ]
        )
        result = np.empty((len(temperature), len(wavelength)), dtype=float)
        for column in range(len(wavelength)):
            result[:, column] = np.interp(
                temperature,
                table_temperatures,
                spectra[:, column],
            )
        return result

    phi_no = channel("phi_no", shortwave_value=0.0)
    phi_no2 = channel("phi_no2", shortwave_value=1.0)
    if np.any(phi_no < 0.0) or np.any(phi_no2 < 0.0):
        raise ValueError("Sander NO3 quantum yields cannot be negative")
    if np.any(phi_no + phi_no2 > 1.001):
        raise ValueError("Sander NO3 channel quantum-yield sum exceeds unity")
    return phi_no, phi_no2


def load_two_stream_no3_photolysis_sander_2006(
    external_data_directory: Path,
    *,
    solar_zenith_degrees: float = 60.0,
    diurnal_factor: float = 0.5,
    surface_albedo: float = 0.25,
    verify_checksums: bool = True,
) -> dict[str, TracePhotolysisProfile]:
    """Evaluate both NO3 channels using Sander (2006) temperature yields."""

    external = Path(external_data_directory)
    source = (
        external
        / "photochem-v0.6.7-source"
        / "photochem-0.6.7"
        / "examples"
        / "ModernEarth"
    )
    data = external / "photochem-data-0.3.0"
    atmosphere_path = source / "atmosphere.txt"
    solar_path = source / "Sun_now.txt"
    o2_path = data / "xsections" / "O2.h5"
    o3_path = data / "xsections" / "O3.h5"
    no3_path = data / "xsections" / "NO3.h5"
    yield_path = data / "quantum_yields" / "NO3_Sander2006_Table4_16.csv"
    paths = (atmosphere_path, solar_path, o2_path, o3_path, no3_path, yield_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned NO3 photolysis inputs: {missing}")
    if verify_checksums:
        for path in paths:
            _verify(path)
    (
        atmosphere,
        altitude,
        wavelength,
        width,
        photon_flux,
        _channels_o2,
        _channels_o3,
        actinic,
    ) = _cached_two_stream_spectral_state(
        atmosphere_path,
        solar_path,
        o2_path,
        o3_path,
        data / "rayleigh" / "rayleigh.yaml",
        solar_zenith_degrees=solar_zenith_degrees,
        surface_albedo=surface_albedo,
    )
    absorption, _archived_channels = _load_xs(no3_path, wavelength)
    temperature = np.asarray(atmosphere["temp"], dtype=float)
    phi_no, phi_no2 = sander_2006_no3_quantum_yields(
        wavelength, temperature, yield_path
    )
    photons_per_bin = photon_flux * width * float(diurnal_factor)
    common = actinic * photons_per_bin * absorption[None, :]
    rates = {
        "NO3 + hv => NO + O2": np.sum(common * phi_no, axis=1),
        "NO3 + hv => NO2 + O": np.sum(common * phi_no2, axis=1),
    }
    result = {}
    for reaction, rate in rates.items():
        result[reaction] = TracePhotolysisProfile(
            altitude_km=altitude,
            rate_per_s=rate,
            species="NO3",
            reaction=reaction,
            solar_zenith_degrees=float(solar_zenith_degrees),
            diurnal_factor=float(diurnal_factor),
            method=(
                "Photochem 0.6.7 two-stream actinic field and NO3 absorption; "
                "temperature-dependent Sander et al. (2006) Table 4-16 yields"
            ),
            source=(
                "Sander et al. (2006), Table 4-16; "
                f"{yield_path.name} SHA-256 {PINNED_SHA256[yield_path.name]}"
            ),
        )
    return result
