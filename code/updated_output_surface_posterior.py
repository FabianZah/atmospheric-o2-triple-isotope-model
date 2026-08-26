"""Conditional one-coordinate posterior for the updated molecular surface.

The likelihood contains analytical measurement uncertainty only. Literature
and numerical model guardrails remain non-probabilistic and are returned from
the companion inverse result rather than being treated as Gaussian errors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np

from isotopes import conventional_delta_from_prime
from updated_molecular_forward_model import UpdatedForwardInput, run_updated_forward
from updated_output_surface import (
    DEFAULT_OUTPUT_SURFACE_PATH,
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)
from updated_output_surface_inverse import (
    SolveCoordinate,
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)


PriorKind = Literal["uniform", "log_uniform"]


@dataclass(frozen=True)
class UpdatedConditionalPosteriorInput:
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = None
    solve_for: SolveCoordinate = "pCO2"
    prior: PriorKind = "log_uniform"
    credible_mass: float = 0.95
    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.0
    gpp_pgC_per_year: float = 290.0
    solve_bounds: tuple[float, float] | None = None
    grid_size: int = 4097


@dataclass(frozen=True)
class UpdatedConditionalPosteriorResult:
    inputs: UpdatedConditionalPosteriorInput
    status: str
    coordinate_values: tuple[float, ...]
    model_cap_delta17_permil: tuple[float, ...]
    model_delta18_conventional_permil: tuple[float, ...]
    relative_likelihood: tuple[float, ...]
    prior_density: tuple[float, ...]
    posterior_density: tuple[float, ...]
    posterior_mode: float
    posterior_mean: float
    posterior_median: float
    equal_tailed_credible_interval: tuple[float, float]
    central_likelihood_root: float | None
    nonprobabilistic_admissible_interval: tuple[float, float] | None
    model_at_mode_permil: float
    model_delta18_at_mode_permil: float
    live_model_at_mode_permil: float | None
    live_mode_residual_permil: float | None
    live_mode_verified: bool
    posterior_integral: float
    edge_band_prior_mass: float
    lower_edge_probability: float
    upper_edge_probability: float
    boundary_sensitive: bool
    solve_units: str
    surface_data_id: str
    upstream_model_data_id: str
    probability_scope: str
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
    request: UpdatedConditionalPosteriorInput, value: float
) -> UpdatedOutputSurfaceInput:
    return UpdatedOutputSurfaceInput(
        p_o2_pal=value if request.solve_for == "pO2" else request.p_o2_pal,
        p_co2_ppm=value if request.solve_for == "pCO2" else request.p_co2_ppm,
        gpp_pgC_per_year=(
            value if request.solve_for == "GPP" else request.gpp_pgC_per_year
        ),
    )


def _cumulative_trapezoid(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(values)
    cumulative[1:] = np.cumsum(
        0.5 * (values[:-1] + values[1:]) * np.diff(coordinates)
    )
    return cumulative


def _integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy 1.x compatibility
        trapezoid = np.trapz
    return float(trapezoid(values, coordinates))


def _quantile(cdf: np.ndarray, coordinates: np.ndarray, probability: float) -> float:
    return float(np.interp(probability, cdf, coordinates))


def conditional_updated_posterior(
    request: UpdatedConditionalPosteriorInput,
    *,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
    verify_live_mode: bool = True,
) -> UpdatedConditionalPosteriorResult:
    """Evaluate a bounded conditional posterior for one physical coordinate."""

    numeric = (
        request.target_air_cap_delta17_permil,
        request.measurement_sigma_permil,
        request.credible_mass,
        request.p_o2_pal,
        request.p_co2_ppm,
        request.gpp_pgC_per_year,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("posterior inputs must be finite")
    if request.solve_for not in {"pCO2", "GPP", "pO2"}:
        raise ValueError("solve_for must be pCO2, GPP, or pO2")
    if request.prior not in {"uniform", "log_uniform"}:
        raise ValueError("prior must be uniform or log_uniform")
    if request.measurement_sigma_permil <= 0.0:
        raise ValueError("measurement sigma must be positive")
    d18_values = (
        request.target_air_delta18_conventional_permil,
        request.delta18_measurement_sigma_permil,
    )
    if (d18_values[0] is None) != (d18_values[1] is None):
        raise ValueError(
            "delta18 target and measurement sigma must be provided together"
        )
    if d18_values[0] is not None:
        if not all(isfinite(float(value)) for value in d18_values):
            raise ValueError("delta18 observation inputs must be finite")
        if float(d18_values[1]) <= 0.0:
            raise ValueError("delta18 measurement sigma must be positive")
    if not 0.0 < request.credible_mass < 1.0:
        raise ValueError("credible mass must lie between zero and one")
    if request.grid_size < 257 or request.grid_size % 2 == 0:
        raise ValueError("grid size must be an odd integer of at least 257")

    surface = load_updated_output_surface(str(Path(surface_path).resolve()))
    domain = _coordinate_domain(surface, request.solve_for)
    lower, upper = map(
        float, domain if request.solve_bounds is None else request.solve_bounds
    )
    if lower <= 0.0 or upper <= lower:
        raise ValueError("solve bounds must be positive with lower < upper")
    if lower < domain[0] or upper > domain[1]:
        raise ValueError(
            f"solve bounds {(lower, upper)} exceed {request.solve_for} "
            f"surface domain {domain}"
        )

    # Geometric evaluation resolves the broad pCO2 and GPP domains. Posterior
    # normalization is still performed with respect to the physical coordinate.
    coordinates = (
        np.linspace(lower, upper, request.grid_size)
        if request.solve_for == "pO2"
        else np.geomspace(lower, upper, request.grid_size)
    )
    evaluated = [surface.evaluate(_request_at(request, float(value))) for value in coordinates]
    predictions = np.asarray(
        [item.central_cap_delta17_prime_permil for item in evaluated], dtype=float
    )
    delta18_predictions = np.asarray(
        [
            conventional_delta_from_prime(item.central_delta18_prime_permil)
            for item in evaluated
        ],
        dtype=float,
    )
    residual = predictions - request.target_air_cap_delta17_permil
    log_likelihood = -0.5 * np.square(residual / request.measurement_sigma_permil)
    if request.target_air_delta18_conventional_permil is not None:
        delta18_residual = (
            delta18_predictions - request.target_air_delta18_conventional_permil
        )
        log_likelihood -= 0.5 * np.square(
            delta18_residual / float(request.delta18_measurement_sigma_permil)
        )
    relative_likelihood = np.exp(log_likelihood - float(np.max(log_likelihood)))

    if request.prior == "uniform":
        prior_density = np.full_like(coordinates, 1.0 / (upper - lower))
    else:
        prior_density = 1.0 / (coordinates * np.log(upper / lower))
    unnormalized = relative_likelihood * prior_density
    normalization = _integral(unnormalized, coordinates)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("posterior normalization failed")
    posterior_density = unnormalized / normalization
    cdf = _cumulative_trapezoid(posterior_density, coordinates)
    cdf /= cdf[-1]

    tail = 0.5 * (1.0 - request.credible_mass)
    credible_interval = (
        _quantile(cdf, coordinates, tail),
        _quantile(cdf, coordinates, 1.0 - tail),
    )
    mode_index = int(np.argmax(posterior_density))
    mode = float(coordinates[mode_index])
    mean = _integral(coordinates * posterior_density, coordinates)
    median = _quantile(cdf, coordinates, 0.5)

    edge_fraction = 0.01
    if request.prior == "uniform":
        lower_edge_coordinate = lower + edge_fraction * (upper - lower)
        upper_edge_coordinate = upper - edge_fraction * (upper - lower)
    else:
        prior_ratio = upper / lower
        lower_edge_coordinate = lower * prior_ratio**edge_fraction
        upper_edge_coordinate = lower * prior_ratio ** (1.0 - edge_fraction)
    lower_edge = float(np.interp(lower_edge_coordinate, coordinates, cdf))
    upper_edge = float(
        1.0 - np.interp(upper_edge_coordinate, coordinates, cdf)
    )
    edge_density_ratio = max(
        float(posterior_density[0]), float(posterior_density[-1])
    ) / float(np.max(posterior_density))
    boundary_sensitive = bool(
        max(lower_edge, upper_edge) >= 0.05 or edge_density_ratio >= 0.10
    )

    inverse = invert_updated_output_surface(
        UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=request.target_air_cap_delta17_permil,
            solve_for=request.solve_for,
            measurement_uncertainty_permil=request.measurement_sigma_permil,
            p_o2_pal=request.p_o2_pal,
            p_co2_ppm=request.p_co2_ppm,
            gpp_pgC_per_year=request.gpp_pgC_per_year,
            solve_bounds=(lower, upper),
        ),
        surface_path=surface_path,
        verify_live_root=False,
    )

    live_value = None
    live_residual = None
    if verify_live_mode:
        mode_request = _request_at(request, mode)
        live = run_updated_forward(UpdatedForwardInput(**asdict(mode_request)))
        live_value = live.central_cap_delta17_prime_permil
        live_residual = live_value - request.target_air_cap_delta17_permil

    diagnostic = (
        "Posterior mass is sensitive to at least one declared prior bound."
        if boundary_sensitive
        else "Posterior mass is resolved away from the declared prior bounds."
    )
    return UpdatedConditionalPosteriorResult(
        inputs=request,
        status=("boundary_sensitive" if boundary_sensitive else "posterior_computed"),
        coordinate_values=tuple(map(float, coordinates)),
        model_cap_delta17_permil=tuple(map(float, predictions)),
        model_delta18_conventional_permil=tuple(map(float, delta18_predictions)),
        relative_likelihood=tuple(map(float, relative_likelihood)),
        prior_density=tuple(map(float, prior_density)),
        posterior_density=tuple(map(float, posterior_density)),
        posterior_mode=mode,
        posterior_mean=mean,
        posterior_median=median,
        equal_tailed_credible_interval=credible_interval,
        central_likelihood_root=inverse.central_root,
        nonprobabilistic_admissible_interval=inverse.admissible_interval,
        model_at_mode_permil=float(predictions[mode_index]),
        model_delta18_at_mode_permil=float(delta18_predictions[mode_index]),
        live_model_at_mode_permil=live_value,
        live_mode_residual_permil=live_residual,
        live_mode_verified=bool(verify_live_mode),
        posterior_integral=_integral(posterior_density, coordinates),
        edge_band_prior_mass=edge_fraction,
        lower_edge_probability=lower_edge,
        upper_edge_probability=upper_edge,
        boundary_sensitive=boundary_sensitive,
        solve_units=_units(request.solve_for),
        surface_data_id=surface.surface_data_id,
        upstream_model_data_id=surface.upstream_model_data_id,
        probability_scope=(
            "conditional on the two fixed physical coordinates, the central "
            "updated model, the declared bounded prior, and Gaussian analytical "
            "measurement uncertainty"
            + (
                " for Delta-prime-17O and conventional delta-18O"
                if request.target_air_delta18_conventional_permil is not None
                else " only (Delta-prime-17O)"
            )
        ),
        diagnostic=diagnostic,
    )
