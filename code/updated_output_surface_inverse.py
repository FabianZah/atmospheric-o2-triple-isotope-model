"""Fast one-coordinate inversion of the updated molecular D17O surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Any, Literal

from scipy.optimize import brentq

from updated_molecular_forward_model import UpdatedForwardInput, run_updated_forward
from updated_output_surface import (
    DEFAULT_OUTPUT_SURFACE_PATH,
    UpdatedOutputSurfaceInput,
    UpdatedOutputSurfacePrediction,
    load_updated_output_surface,
)


SolveCoordinate = Literal["pCO2", "GPP", "pO2"]
_INCREASING_COORDINATES = {"GPP", "pO2"}


@dataclass(frozen=True)
class UpdatedSurfaceInverseInput:
    target_air_cap_delta17_permil: float
    solve_for: SolveCoordinate = "pCO2"
    measurement_uncertainty_permil: float = 0.0
    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.0
    gpp_pgC_per_year: float = 290.0
    solve_bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class UpdatedSurfaceInverseResult:
    inputs: UpdatedSurfaceInverseInput
    status: str
    central_root: float | None
    admissible_interval: tuple[float, float] | None
    measurement_interval_permil: tuple[float, float]
    central_model_at_root_permil: float | None
    accelerated_root_residual_permil: float | None
    live_model_at_root_permil: float | None
    live_root_residual_permil: float | None
    live_root_verified: bool
    surface_data_id: str
    upstream_model_data_id: str
    solve_units: str
    solve_direction: str
    interpolation_guardrail_permil: float
    uncertainty_policy: str
    surface_evaluations: int
    diagnostic: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coordinate_domain(surface, solve_for: SolveCoordinate) -> tuple[float, float]:
    return {
        "pCO2": surface.domain["pco2_ppm"],
        "GPP": surface.domain["gpp_pgC_per_year"],
        "pO2": surface.domain["po2_pal"],
    }[solve_for]


def _units(solve_for: SolveCoordinate) -> str:
    return {"pCO2": "ppm", "GPP": "PgC yr-1", "pO2": "PAL"}[solve_for]


def _request_at(
    request: UpdatedSurfaceInverseInput, value: float
) -> UpdatedOutputSurfaceInput:
    return UpdatedOutputSurfaceInput(
        p_o2_pal=value if request.solve_for == "pO2" else request.p_o2_pal,
        p_co2_ppm=value if request.solve_for == "pCO2" else request.p_co2_ppm,
        gpp_pgC_per_year=(
            value if request.solve_for == "GPP" else request.gpp_pgC_per_year
        ),
    )


def _root_if_crossed(function, lower: float, upper: float) -> float | None:
    lower_value = float(function(lower))
    upper_value = float(function(upper))
    if abs(lower_value) <= 1.0e-12:
        return lower
    if abs(upper_value) <= 1.0e-12:
        return upper
    if lower_value * upper_value > 0.0:
        return None
    return float(
        exp(
            brentq(
                lambda log_value: function(
                    min(max(exp(log_value), lower), upper)
                ),
                log(lower),
                log(upper),
                xtol=1.0e-12,
                rtol=1.0e-12,
                maxiter=100,
            )
        )
    )


def invert_updated_output_surface(
    request: UpdatedSurfaceInverseInput,
    *,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
    verify_live_root: bool = True,
) -> UpdatedSurfaceInverseResult:
    """Solve one physical coordinate with the other two held fixed."""

    if request.solve_for not in {"pCO2", "GPP", "pO2"}:
        raise ValueError("solve_for must be pCO2, GPP, or pO2")
    numeric = (
        request.target_air_cap_delta17_permil,
        request.measurement_uncertainty_permil,
        request.p_o2_pal,
        request.p_co2_ppm,
        request.gpp_pgC_per_year,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("inverse inputs must be finite")
    if request.measurement_uncertainty_permil < 0.0:
        raise ValueError("measurement uncertainty must be non-negative")

    surface = load_updated_output_surface(str(Path(surface_path).resolve()))
    domain = _coordinate_domain(surface, request.solve_for)
    bounds = domain if request.solve_bounds is None else request.solve_bounds
    lower, upper = map(float, bounds)
    if lower <= 0.0 or upper <= lower:
        raise ValueError("solve bounds must be positive with lower < upper")
    if lower < domain[0] or upper > domain[1]:
        raise ValueError(
            f"solve bounds {bounds} exceed {request.solve_for} surface domain {domain}"
        )

    # Validate the two fixed coordinates before starting a root search.
    surface._point(_request_at(request, lower))
    surface._point(_request_at(request, upper))
    evaluations = 0
    cache: dict[float, UpdatedOutputSurfacePrediction] = {}

    def prediction(value: float) -> UpdatedOutputSurfacePrediction:
        nonlocal evaluations
        key = float(value)
        if key not in cache:
            cache[key] = surface.evaluate(_request_at(request, key))
            evaluations += 1
        return cache[key]

    target = request.target_air_cap_delta17_permil
    measurement = (
        target - request.measurement_uncertainty_permil,
        target + request.measurement_uncertainty_permil,
    )

    def central_residual(value: float) -> float:
        return prediction(value).central_cap_delta17_prime_permil - target

    central_root = _root_if_crossed(central_residual, lower, upper)
    increasing = request.solve_for in _INCREASING_COORDINATES

    def model_interval(value: float) -> tuple[float, float]:
        return prediction(value).accelerated_model_guardrail_interval_cap_delta17_permil

    if increasing:
        entry_residual = lambda value: model_interval(value)[1] - measurement[0]
        exit_residual = lambda value: model_interval(value)[0] - measurement[1]
        lower_inside = entry_residual(lower) >= 0.0
        upper_inside = exit_residual(upper) <= 0.0
    else:
        entry_residual = lambda value: model_interval(value)[0] - measurement[1]
        exit_residual = lambda value: model_interval(value)[1] - measurement[0]
        lower_inside = entry_residual(lower) <= 0.0
        upper_inside = exit_residual(upper) >= 0.0

    interval_lower = lower if lower_inside else _root_if_crossed(
        entry_residual, lower, upper
    )
    interval_upper = upper if upper_inside else _root_if_crossed(
        exit_residual, lower, upper
    )
    admissible = None
    if (
        interval_lower is not None
        and interval_upper is not None
        and interval_lower <= interval_upper
    ):
        admissible = (float(interval_lower), float(interval_upper))

    central_value = None
    accelerated_residual = None
    live_value = None
    live_residual = None
    if central_root is not None:
        central_value = prediction(central_root).central_cap_delta17_prime_permil
        accelerated_residual = central_value - target
        if verify_live_root:
            live = run_updated_forward(
                UpdatedForwardInput(**asdict(_request_at(request, central_root)))
            )
            live_value = live.central_cap_delta17_prime_permil
            live_residual = live_value - target

    status = (
        "admissible_interval_found"
        if admissible is not None
        else "no_interval_overlap_in_search_domain"
    )
    diagnostic = (
        "The admissible interval is measurement-model guardrail overlap at fixed "
        "values of the other two coordinates; it is not a posterior interval."
        if admissible is not None
        else "The measurement and accelerated model guardrails do not overlap "
        "inside the requested domain."
    )
    return UpdatedSurfaceInverseResult(
        inputs=request,
        status=status,
        central_root=central_root,
        admissible_interval=admissible,
        measurement_interval_permil=measurement,
        central_model_at_root_permil=central_value,
        accelerated_root_residual_permil=accelerated_residual,
        live_model_at_root_permil=live_value,
        live_root_residual_permil=live_residual,
        live_root_verified=bool(verify_live_root and central_root is not None),
        surface_data_id=surface.surface_data_id,
        upstream_model_data_id=surface.upstream_model_data_id,
        solve_units=_units(request.solve_for),
        solve_direction=(
            "increasing Delta-prime-17O"
            if increasing
            else "decreasing Delta-prime-17O"
        ),
        interpolation_guardrail_permil=(
            surface.maximum_holdout_cap_delta17_residual_permil
        ),
        uncertainty_policy=(
            "measurement interval overlapped with source-isoflux, biological-"
            "process, kernel-interpolation, and output-surface guardrails"
        ),
        surface_evaluations=evaluations,
        diagnostic=diagnostic,
    )
