"""Gravity-wave vertical diffusivity from Summers et al. (1997).

This module implements Equations 1-8 and Table 1 of Summers et al.
(1997), *Seasonal variation of middle atmospheric CH4 and H2O with a
new chemical-dynamical model*. The equations are evaluated on the printed
2.66 km vertical spacing. No diffusivity multiplier, cap, or age-of-air fit
is applied.

The historical Summers model wind, temperature, and density fields are not
archived with the paper. ``summers_kzz_from_era5`` is therefore an explicitly
modern, source-derived candidate: it supplies the printed gravity-wave law
with monthly ERA5 zonal wind, temperature, and static stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REFERENCE_PRESSURE_PA = 100_000.0
LOG_PRESSURE_SCALE_HEIGHT_M = 7_000.0
DRY_AIR_GAS_CONSTANT_J_PER_KG_K = 287.05
EARTH_RADIUS_M = 6_371_000.0
SOURCE_ALTITUDE_M = 2_520.0
VERTICAL_SPACING_M = 2_660.0
PRANDTL_NUMBER = 3.0
GAMMA1_S_PER_M2 = (1.0 / (1.5 * 86_400.0)) / 60.0**2


@dataclass(frozen=True)
class GravityWave:
    phase_speed_m_per_s: float
    source_displacement_m: float
    multiplicity: int


SUMMERS_TABLE1_WAVES = (
    GravityWave(+40.0, 15.0, 3),
    GravityWave(+40.0, 30.0, 3),
    GravityWave(+20.0, 40.0, 5),
    GravityWave(-20.0, 40.0, 5),
    GravityWave(-40.0, 30.0, 3),
    GravityWave(-40.0, 15.0, 3),
)


@dataclass(frozen=True)
class SummersKzzField:
    """Summers-law Kzz on a latitude by altitude grid."""

    latitude_degrees: np.ndarray
    altitude_m: np.ndarray
    kzz_m2_per_s: np.ndarray
    zero_phase_kzz_m2_per_s: np.ndarray
    nonzero_phase_kzz_m2_per_s: np.ndarray
    gravity_wave_drag_m_per_s2: np.ndarray
    zero_phase_drag_m_per_s2: np.ndarray
    nonzero_phase_drag_m_per_s2: np.ndarray
    critical_level_absorption_count: np.ndarray
    source: str

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_degrees, dtype=float)
        altitude = np.asarray(self.altitude_m, dtype=float)
        expected = (len(latitude), len(altitude))
        fields = (
            self.kzz_m2_per_s,
            self.zero_phase_kzz_m2_per_s,
            self.nonzero_phase_kzz_m2_per_s,
            self.gravity_wave_drag_m_per_s2,
            self.zero_phase_drag_m_per_s2,
            self.nonzero_phase_drag_m_per_s2,
        )
        if any(np.asarray(field).shape != expected for field in fields):
            raise ValueError("Summers Kzz fields must have shape (latitude, altitude)")
        if not np.all(np.diff(latitude) > 0.0):
            raise ValueError("latitude must increase south to north")
        if not np.all(np.diff(altitude) > 0.0):
            raise ValueError("altitude must increase upward")
        if any(not np.all(np.isfinite(field)) for field in fields):
            raise ValueError("Summers Kzz fields must be finite")
        kzz_fields = (
            self.kzz_m2_per_s,
            self.zero_phase_kzz_m2_per_s,
            self.nonzero_phase_kzz_m2_per_s,
        )
        if any(np.any(np.asarray(field) < 0.0) for field in kzz_fields):
            raise ValueError("Summers Kzz fields must be non-negative")
        if not np.allclose(
            self.kzz_m2_per_s,
            self.zero_phase_kzz_m2_per_s + self.nonzero_phase_kzz_m2_per_s,
        ):
            raise ValueError("total Kzz must equal its wave components")
        if not np.allclose(
            self.gravity_wave_drag_m_per_s2,
            self.zero_phase_drag_m_per_s2 + self.nonzero_phase_drag_m_per_s2,
        ):
            raise ValueError("total gravity-wave drag must equal its components")
        absorption_count = np.asarray(self.critical_level_absorption_count)
        if absorption_count.shape != (len(latitude),):
            raise ValueError("critical-level counts must have one value per latitude")
        if np.any(absorption_count < 0) or np.any(absorption_count % 1 != 0):
            raise ValueError("critical-level counts must be non-negative integers")


def log_pressure_altitude_m(pressure_pa: np.ndarray) -> np.ndarray:
    """Summers log-pressure altitude, z = -7 ln(p/1000 hPa) km."""

    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.ndim != 1 or np.any(pressure <= 0.0):
        raise ValueError("pressure must be a positive one-dimensional array")
    return LOG_PRESSURE_SCALE_HEIGHT_M * np.log(REFERENCE_PRESSURE_PA / pressure)


def summers_vertical_grid_m(maximum_altitude_m: float) -> np.ndarray:
    """Native Summers grid from 2.52 km with 2.66 km spacing."""

    if maximum_altitude_m < SOURCE_ALTITUDE_M:
        raise ValueError("maximum altitude lies below the wave source")
    count = int(np.floor((maximum_altitude_m - SOURCE_ALTITUDE_M) / VERTICAL_SPACING_M))
    return SOURCE_ALTITUDE_M + VERTICAL_SPACING_M * np.arange(count + 1, dtype=float)


def zero_phase_drag_m_per_s2(
    latitude_degrees: np.ndarray,
    altitude_m: np.ndarray,
    zonal_wind_m_per_s: np.ndarray,
) -> np.ndarray:
    """Zero-phase-speed gravity-wave drag from Summers Equation 1."""

    latitude = np.asarray(latitude_degrees, dtype=float)
    altitude = np.asarray(altitude_m, dtype=float)
    wind = np.asarray(zonal_wind_m_per_s, dtype=float)
    if wind.shape != (len(latitude), len(altitude)):
        raise ValueError("wind must have shape (latitude, altitude)")
    onset_m = np.where(latitude < 0.0, 55_000.0, 45_000.0)[:, None]
    ramp = np.clip((altitude[None, :] - onset_m) / 20_000.0, 0.0, 1.0)
    return -(GAMMA1_S_PER_M2 * ramp) * wind**3


def _wave_kzz_column(
    *,
    altitude_m: np.ndarray,
    density_kg_per_m3: np.ndarray,
    brunt_vaisala_per_s: np.ndarray,
    zonal_wind_m_per_s: np.ndarray,
    wave: GravityWave,
    prandtl_number: float,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Evaluate Summers Equations 2-5 and 7 for one nonzero-c wave."""

    altitude = np.asarray(altitude_m, dtype=float)
    density = np.asarray(density_kg_per_m3, dtype=float)
    frequency = np.asarray(brunt_vaisala_per_s, dtype=float)
    wind = np.asarray(zonal_wind_m_per_s, dtype=float)
    if any(array.shape != altitude.shape for array in (density, frequency, wind)):
        raise ValueError("wave-column fields must share the altitude shape")
    if np.any(density <= 0.0) or np.any(frequency <= 0.0):
        raise ValueError("density and Brunt-Vaisala frequency must be positive")
    if wave.phase_speed_m_per_s == 0.0:
        raise ValueError("nonzero-wave recursion requires nonzero phase speed")

    signed_intrinsic = wind - wave.phase_speed_m_per_s
    intrinsic = np.abs(signed_intrinsic)
    displacement = np.zeros_like(altitude)
    displacement[0] = min(
        wave.source_displacement_m,
        intrinsic[0] / frequency[0],
    )
    sign = float(np.sign(wave.phase_speed_m_per_s))
    momentum_flux = np.zeros_like(altitude)
    momentum_flux[0] = (
        sign * density[0] * frequency[0] * intrinsic[0] * displacement[0] ** 2
    )
    drag = np.zeros_like(altitude)
    if signed_intrinsic[0] == 0.0:
        return np.zeros_like(altitude), np.zeros_like(altitude), 0
    critical_level_index = None

    for level in range(1, len(altitude)):
        if signed_intrinsic[level - 1] * signed_intrinsic[level] <= 0.0:
            # Bacmeister's signed-wind recurrence terminates at U=c. Kim et
            # al. (2003), Sec. 4a and Appendix A.d.4, state explicitly that
            # the wave dissipates totally and is absorbed at this level.
            momentum_flux[level] = 0.0
            drag[level] = -momentum_flux[level - 1] / (
                density[level] * (altitude[level] - altitude[level - 1])
            )
            critical_level_index = level
            break
        previous_factor = density[level - 1] * frequency[level - 1] * intrinsic[level - 1]
        current_factor = density[level] * frequency[level] * intrinsic[level]
        if current_factor == 0.0:
            unconstrained = np.inf if previous_factor > 0.0 else 0.0
        else:
            unconstrained = np.sqrt(previous_factor / current_factor) * displacement[level - 1]
        saturation_limit = intrinsic[level] / frequency[level]
        displacement[level] = min(unconstrained, saturation_limit)
        momentum_flux[level] = (
            sign
            * density[level]
            * frequency[level]
            * intrinsic[level]
            * displacement[level] ** 2
        )
        dz = altitude[level] - altitude[level - 1]
        drag[level] = (
            momentum_flux[level] - momentum_flux[level - 1]
        ) / (density[level] * dz)

    return (
        intrinsic / frequency**2 * np.abs(drag) / prandtl_number,
        drag,
        critical_level_index,
    )


def summers_kzz_on_native_grid(
    *,
    latitude_degrees: np.ndarray,
    altitude_m: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    zonal_wind_m_per_s: np.ndarray,
    static_stability_per_s2: np.ndarray,
    prandtl_number: float = PRANDTL_NUMBER,
) -> SummersKzzField:
    """Evaluate the printed Summers gravity-wave law on its native grid."""

    latitude = np.asarray(latitude_degrees, dtype=float)
    altitude = np.asarray(altitude_m, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    wind = np.asarray(zonal_wind_m_per_s, dtype=float)
    stability = np.asarray(static_stability_per_s2, dtype=float)
    expected = (len(latitude), len(altitude))
    if pressure.shape != altitude.shape:
        raise ValueError("pressure must share the altitude dimension")
    if any(field.shape != expected for field in (temperature, wind, stability)):
        raise ValueError("atmospheric fields must have shape (latitude, altitude)")
    if not np.allclose(np.diff(altitude), VERTICAL_SPACING_M, atol=1.0e-6):
        raise ValueError("Summers equations must use the printed 2.66 km spacing")
    if prandtl_number <= 0.0:
        raise ValueError("Prandtl number must be positive")
    if np.any(temperature <= 0.0) or np.any(stability <= 0.0):
        raise ValueError("temperature and static stability must be positive")

    density = pressure[None, :] / (
        DRY_AIR_GAS_CONSTANT_J_PER_KG_K * temperature
    )
    frequency = np.sqrt(stability)
    zero_drag = zero_phase_drag_m_per_s2(latitude, altitude, wind)
    zero = np.abs(wind) / stability * np.abs(zero_drag) / prandtl_number
    packet_integrated_nonzero = np.zeros(expected, dtype=float)
    packet_integrated_nonzero_drag = np.zeros(expected, dtype=float)
    critical_level_absorption_count = np.zeros(len(latitude), dtype=int)
    for latitude_index in range(len(latitude)):
        for wave in SUMMERS_TABLE1_WAVES:
            wave_kzz, wave_drag, critical_level_index = _wave_kzz_column(
                altitude_m=altitude,
                density_kg_per_m3=density[latitude_index],
                brunt_vaisala_per_s=frequency[latitude_index],
                zonal_wind_m_per_s=wind[latitude_index],
                wave=wave,
                prandtl_number=prandtl_number,
            )
            packet_integrated_nonzero[latitude_index] += wave.multiplicity * wave_kzz
            packet_integrated_nonzero_drag[latitude_index] += wave.multiplicity * wave_drag
            if critical_level_index is not None:
                critical_level_absorption_count[latitude_index] += wave.multiplicity
    # Equation 2 integrates each packet from -L to +L, while Table 1 defines
    # n_i as the number of packets encountered along a latitude circle. Their
    # sum is therefore an integrated quantity. Division by the circle length
    # is the dimensionally required conversion to the zonal mean used by the
    # 2-D model. Summers et al. do not print this geometry step in Eqs. 2-8.
    latitude_circle_length_m = (
        2.0 * np.pi * EARTH_RADIUS_M * np.cos(np.deg2rad(latitude))
    )
    if np.any(latitude_circle_length_m <= 0.0):
        raise ValueError("latitude nodes must lie strictly between the poles")
    nonzero = packet_integrated_nonzero / latitude_circle_length_m[:, None]
    nonzero_drag = packet_integrated_nonzero_drag / latitude_circle_length_m[:, None]
    return SummersKzzField(
        latitude_degrees=latitude,
        altitude_m=altitude,
        kzz_m2_per_s=zero + nonzero,
        zero_phase_kzz_m2_per_s=zero,
        nonzero_phase_kzz_m2_per_s=nonzero,
        gravity_wave_drag_m_per_s2=zero_drag + nonzero_drag,
        zero_phase_drag_m_per_s2=zero_drag,
        nonzero_phase_drag_m_per_s2=nonzero_drag,
        critical_level_absorption_count=critical_level_absorption_count,
        source=(
            "Summers et al. (1997), Equations 1-8 and Table 1; "
            f"Pr={prandtl_number:g}; packet sums normalized by latitude-circle "
            "length to form a zonal mean; critical-level absorption follows "
            "Bacmeister (1993) and Kim et al. (2003); atmospheric state supplied "
            "externally"
        ),
    )


def summers_kzz_from_pressure_state(climatology: object) -> SummersKzzField:
    """Drive the printed Summers law with a pressure-coordinate state."""

    latitude = np.asarray(climatology.latitude_degrees, dtype=float)
    source_altitude = log_pressure_altitude_m(climatology.pressure_pa)
    target_altitude = summers_vertical_grid_m(float(source_altitude[-1]))
    target_pressure = REFERENCE_PRESSURE_PA * np.exp(
        -target_altitude / LOG_PRESSURE_SCALE_HEIGHT_M
    )

    def interpolate(field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=float)
        if values.shape != (len(latitude), len(source_altitude)):
            raise ValueError("pressure-state field has an unexpected shape")
        return np.asarray(
            [np.interp(target_altitude, source_altitude, row) for row in values],
            dtype=float,
        )

    return summers_kzz_on_native_grid(
        latitude_degrees=latitude,
        altitude_m=target_altitude,
        pressure_pa=target_pressure,
        temperature_k=interpolate(climatology.temperature_k),
        zonal_wind_m_per_s=interpolate(climatology.zonal_wind_m_per_s),
        static_stability_per_s2=interpolate(
            climatology.static_stability_per_s2
        ),
    )


def summers_kzz_from_era5(climatology: object) -> SummersKzzField:
    """Backward-compatible adapter for an ERA5 pressure-coordinate state."""

    return summers_kzz_from_pressure_state(climatology)
