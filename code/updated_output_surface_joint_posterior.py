"""Joint isotope posterior on the updated molecular output surface.

One atmospheric O2 isotope coordinate generally identifies a ridge rather
than unique pCO2, GPP, and pO2 values. This module preserves that geometry.
Analytical and explicitly declared Gaussian model-discrepancy terms enter the
likelihood. Literature-corner and interpolation guardrails are not silently
converted into probability distributions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np

from updated_output_surface import (
    DEFAULT_OUTPUT_SURFACE_PATH,
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
)


Coordinate = Literal["pCO2", "GPP", "pO2"]
PriorKind = Literal["uniform", "log_uniform", "normal"]
COORDINATES: tuple[Coordinate, ...] = ("pCO2", "GPP", "pO2")


@dataclass(frozen=True)
class UpdatedJointPosteriorInput:
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = None
    free_coordinates: tuple[Coordinate, ...] = ("pCO2", "GPP")
    model_discrepancy_sigma_permil: float = 0.0
    model_discrepancy_source: str | None = None
    credible_mass: float = 0.95
    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.0
    gpp_pgC_per_year: float = 290.0
    pco2_bounds_ppm: tuple[float, float] | None = None
    gpp_bounds_pgC_per_year: tuple[float, float] | None = None
    po2_bounds_pal: tuple[float, float] | None = None
    pco2_prior: PriorKind = "log_uniform"
    gpp_prior: PriorKind = "log_uniform"
    po2_prior: PriorKind = "uniform"
    pco2_prior_mean: float | None = None
    pco2_prior_sigma: float | None = None
    gpp_prior_mean: float | None = None
    gpp_prior_sigma: float | None = None
    po2_prior_mean: float | None = None
    po2_prior_sigma: float | None = None
    pco2_grid_size: int = 161
    gpp_grid_size: int = 121
    po2_grid_size: int = 81


@dataclass(frozen=True)
class UpdatedJointPosteriorResult:
    inputs: UpdatedJointPosteriorInput
    status: str
    free_coordinates: tuple[Coordinate, ...]
    axes: dict[str, tuple[float, ...]]
    posterior_shape: tuple[int, ...]
    posterior_density: tuple[float, ...]
    posterior_probability_mass: tuple[float, ...]
    model_cap_delta17_permil: tuple[float, ...]
    model_delta18_conventional_permil: tuple[float, ...]
    marginal_density: dict[str, tuple[float, ...]]
    marginal_probability_mass: dict[str, tuple[float, ...]]
    map_coordinates: dict[str, float]
    posterior_mean: dict[str, float]
    posterior_median: dict[str, float]
    equal_tailed_credible_intervals: dict[str, tuple[float, float]]
    edge_probabilities: dict[str, tuple[float, float]]
    boundary_sensitive: bool
    hpd_density_threshold: float
    hpd_mask: tuple[bool, ...]
    hpd_probability_mass: float
    posterior_integral: float
    effective_likelihood_sigma_permil: float
    probabilistic_model_discrepancy_included: bool
    surface_data_id: str
    upstream_model_data_id: str
    probability_scope: str
    diagnostic: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _domain_key(coordinate: Coordinate) -> str:
    return {"pCO2": "pco2_ppm", "GPP": "gpp_pgC_per_year", "pO2": "po2_pal"}[
        coordinate
    ]


def _bounds(request: UpdatedJointPosteriorInput, coordinate: Coordinate):
    return {
        "pCO2": request.pco2_bounds_ppm,
        "GPP": request.gpp_bounds_pgC_per_year,
        "pO2": request.po2_bounds_pal,
    }[coordinate]


def _prior(request: UpdatedJointPosteriorInput, coordinate: Coordinate) -> PriorKind:
    return {
        "pCO2": request.pco2_prior,
        "GPP": request.gpp_prior,
        "pO2": request.po2_prior,
    }[coordinate]


def _grid_size(request: UpdatedJointPosteriorInput, coordinate: Coordinate) -> int:
    return {
        "pCO2": request.pco2_grid_size,
        "GPP": request.gpp_grid_size,
        "pO2": request.po2_grid_size,
    }[coordinate]


def _prior_mean(
    request: UpdatedJointPosteriorInput, coordinate: Coordinate
) -> float | None:
    return {
        "pCO2": request.pco2_prior_mean,
        "GPP": request.gpp_prior_mean,
        "pO2": request.po2_prior_mean,
    }[coordinate]


def _prior_sigma(
    request: UpdatedJointPosteriorInput, coordinate: Coordinate
) -> float | None:
    return {
        "pCO2": request.pco2_prior_sigma,
        "GPP": request.gpp_prior_sigma,
        "pO2": request.po2_prior_sigma,
    }[coordinate]


def _fixed_value(request: UpdatedJointPosteriorInput, coordinate: Coordinate) -> float:
    return {
        "pCO2": request.p_co2_ppm,
        "GPP": request.gpp_pgC_per_year,
        "pO2": request.p_o2_pal,
    }[coordinate]


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    weights = np.empty_like(axis)
    weights[0] = 0.5 * (axis[1] - axis[0])
    weights[-1] = 0.5 * (axis[-1] - axis[-2])
    weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return weights


def _quantile(axis: np.ndarray, mass: np.ndarray, probability: float) -> float:
    cdf = np.cumsum(mass)
    cdf /= cdf[-1]
    return float(np.interp(probability, cdf, axis))


def _edge_probability(
    axis: np.ndarray, mass: np.ndarray, prior: PriorKind
) -> tuple[float, float]:
    fraction = 0.01
    if prior != "log_uniform":
        lower_limit = axis[0] + fraction * (axis[-1] - axis[0])
        upper_limit = axis[-1] - fraction * (axis[-1] - axis[0])
    else:
        ratio = axis[-1] / axis[0]
        lower_limit = axis[0] * ratio**fraction
        upper_limit = axis[0] * ratio ** (1.0 - fraction)
    return (
        float(np.sum(mass[axis <= lower_limit])),
        float(np.sum(mass[axis >= upper_limit])),
    )


def joint_updated_posterior(
    request: UpdatedJointPosteriorInput,
    *,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
) -> UpdatedJointPosteriorResult:
    """Evaluate a two- or three-coordinate posterior without collapsing its ridge."""

    numeric = (
        request.target_air_cap_delta17_permil,
        request.measurement_sigma_permil,
        request.model_discrepancy_sigma_permil,
        request.credible_mass,
        request.p_o2_pal,
        request.p_co2_ppm,
        request.gpp_pgC_per_year,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("joint posterior inputs must be finite")
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
    if request.model_discrepancy_sigma_permil < 0.0:
        raise ValueError("model discrepancy sigma must be non-negative")
    if request.model_discrepancy_sigma_permil > 0.0 and not (
        request.model_discrepancy_source or ""
    ).strip():
        raise ValueError(
            "a positive model discrepancy sigma requires a traceable source"
        )
    if not 0.0 < request.credible_mass < 1.0:
        raise ValueError("credible mass must lie between zero and one")
    free = tuple(request.free_coordinates)
    if len(free) not in (1, 2, 3) or len(set(free)) != len(free):
        raise ValueError("free_coordinates must contain one to three unique coordinates")
    if any(item not in COORDINATES for item in free):
        raise ValueError("unknown free coordinate")

    surface = load_updated_output_surface(str(Path(surface_path).resolve()))
    axes: dict[Coordinate, np.ndarray] = {}
    priors: dict[Coordinate, np.ndarray] = {}
    weights: dict[Coordinate, np.ndarray] = {}
    for coordinate in free:
        prior = _prior(request, coordinate)
        if prior not in {"uniform", "log_uniform", "normal"}:
            raise ValueError(f"unknown {coordinate} prior: {prior}")
        domain = tuple(map(float, surface.domain[_domain_key(coordinate)]))
        bounds = domain if _bounds(request, coordinate) is None else _bounds(
            request, coordinate
        )
        lower, upper = map(float, bounds)
        if lower <= 0.0 or upper <= lower or lower < domain[0] or upper > domain[1]:
            raise ValueError(
                f"{coordinate} bounds {(lower, upper)} exceed surface domain {domain}"
            )
        size = _grid_size(request, coordinate)
        if size < 17 or size % 2 == 0:
            raise ValueError(f"{coordinate} grid size must be odd and at least 17")
        axis = (
            np.geomspace(lower, upper, size)
            if prior == "log_uniform" or coordinate == "pCO2"
            else np.linspace(lower, upper, size)
        )
        axes[coordinate] = axis
        weights[coordinate] = _trapezoid_weights(axis)
        if prior == "uniform":
            prior_density = np.full_like(axis, 1.0 / (upper - lower))
        elif prior == "log_uniform":
            prior_density = 1.0 / (axis * np.log(upper / lower))
        else:
            mean = _prior_mean(request, coordinate)
            sigma = _prior_sigma(request, coordinate)
            if (
                mean is None
                or sigma is None
                or not isfinite(mean)
                or not isfinite(sigma)
            ):
                raise ValueError(f"{coordinate} normal prior requires finite mean and sigma")
            if sigma <= 0.0 or not lower <= mean <= upper:
                raise ValueError(
                    f"{coordinate} normal prior requires positive sigma and mean within bounds"
                )
            prior_density = np.exp(-0.5 * np.square((axis - mean) / sigma))
            prior_normalization = float(np.sum(prior_density * weights[coordinate]))
            if not np.isfinite(prior_normalization) or prior_normalization <= 0.0:
                raise RuntimeError(f"{coordinate} normal prior normalization failed")
            prior_density /= prior_normalization
        priors[coordinate] = prior_density

    for coordinate in COORDINATES:
        if coordinate in free:
            continue
        value = _fixed_value(request, coordinate)
        lower, upper = surface.domain[_domain_key(coordinate)]
        if not lower <= value <= upper:
            raise ValueError(
                f"fixed {coordinate}={value:g} is outside surface domain "
                f"[{lower:g}, {upper:g}]"
            )

    mesh = np.meshgrid(*(axes[item] for item in free), indexing="ij")
    fields: dict[Coordinate, np.ndarray | float] = {
        item: _fixed_value(request, item) for item in COORDINATES
    }
    fields.update({item: values for item, values in zip(free, mesh, strict=True)})
    predictions = surface.evaluate_central_cap_delta17_grid(
        p_o2_pal=fields["pO2"],
        p_co2_ppm=fields["pCO2"],
        gpp_pgC_per_year=fields["GPP"],
    )
    delta18_predictions = 1000.0 * np.expm1(
        surface.evaluate_central_delta18_prime_grid(
            p_o2_pal=fields["pO2"],
            p_co2_ppm=fields["pCO2"],
            gpp_pgC_per_year=fields["GPP"],
        )
        / 1000.0
    )

    effective_sigma = sqrt(
        request.measurement_sigma_permil**2
        + request.model_discrepancy_sigma_permil**2
    )
    log_likelihood = -0.5 * np.square(
        (predictions - request.target_air_cap_delta17_permil) / effective_sigma
    )
    if request.target_air_delta18_conventional_permil is not None:
        log_likelihood -= 0.5 * np.square(
            (
                delta18_predictions
                - request.target_air_delta18_conventional_permil
            )
            / float(request.delta18_measurement_sigma_permil)
        )
    log_likelihood -= float(np.max(log_likelihood))
    density = np.exp(log_likelihood)
    volume_weights = np.ones_like(density)
    for dimension, coordinate in enumerate(free):
        shape = [1] * len(free)
        shape[dimension] = len(axes[coordinate])
        density *= priors[coordinate].reshape(shape)
        volume_weights *= weights[coordinate].reshape(shape)
    normalization = float(np.sum(density * volume_weights))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("joint posterior normalization failed")
    density /= normalization
    probability_mass = density * volume_weights
    probability_mass /= float(np.sum(probability_mass))

    marginal_density: dict[str, tuple[float, ...]] = {}
    marginal_mass: dict[str, tuple[float, ...]] = {}
    means: dict[str, float] = {}
    medians: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    edge_probabilities: dict[str, tuple[float, float]] = {}
    tail = 0.5 * (1.0 - request.credible_mass)
    for dimension, coordinate in enumerate(free):
        sum_axes = tuple(index for index in range(len(free)) if index != dimension)
        mass = np.sum(probability_mass, axis=sum_axes) if sum_axes else probability_mass
        mass = np.asarray(mass, dtype=float)
        mass /= float(np.sum(mass))
        axis = axes[coordinate]
        one_density = mass / weights[coordinate]
        one_density /= float(np.sum(one_density * weights[coordinate]))
        marginal_density[coordinate] = tuple(map(float, one_density))
        marginal_mass[coordinate] = tuple(map(float, mass))
        means[coordinate] = float(np.sum(axis * mass))
        medians[coordinate] = _quantile(axis, mass, 0.5)
        intervals[coordinate] = (
            _quantile(axis, mass, tail),
            _quantile(axis, mass, 1.0 - tail),
        )
        edge_probabilities[coordinate] = _edge_probability(
            axis, mass, _prior(request, coordinate)
        )

    mode_index = np.unravel_index(int(np.argmax(density)), density.shape)
    map_coordinates = {
        coordinate: float(axes[coordinate][mode_index[dimension]])
        for dimension, coordinate in enumerate(free)
    }
    flat_density = density.reshape(-1)
    flat_mass = probability_mass.reshape(-1)
    order = np.argsort(flat_density)[::-1]
    cumulative = np.cumsum(flat_mass[order])
    cutoff_position = min(
        int(np.searchsorted(cumulative, request.credible_mass, side="left")),
        len(order) - 1,
    )
    threshold = float(flat_density[order[cutoff_position]])
    hpd_mask = density >= threshold
    hpd_mass = float(np.sum(probability_mass[hpd_mask]))
    boundary_sensitive = any(
        max(probabilities) >= 0.05
        for probabilities in edge_probabilities.values()
    )
    discrepancy_included = request.model_discrepancy_sigma_permil > 0.0
    scope = (
        "joint posterior conditional on the declared independent coordinate priors, "
        "the central updated model, Gaussian analytical measurement uncertainty"
    )
    if request.target_air_delta18_conventional_permil is not None:
        scope += " and conventional delta-18O"
    if discrepancy_included:
        scope += (
            ", and explicit Gaussian model discrepancy from "
            f"{request.model_discrepancy_source}"
        )
    else:
        scope += "; no probabilistic structural model discrepancy is included"
    scope += (
        ". Literature-corner and interpolation guardrails remain "
        "non-probabilistic diagnostics."
    )
    diagnostic = (
        "Posterior mass reaches at least one declared prior boundary; report "
        "the affected marginal as prior-bound sensitive."
        if boundary_sensitive
        else "Joint posterior mass is resolved away from all declared prior bounds."
    )
    return UpdatedJointPosteriorResult(
        inputs=request,
        status="boundary_sensitive" if boundary_sensitive else "posterior_computed",
        free_coordinates=free,
        axes={item: tuple(map(float, axes[item])) for item in free},
        posterior_shape=tuple(density.shape),
        posterior_density=tuple(map(float, flat_density)),
        posterior_probability_mass=tuple(map(float, flat_mass)),
        model_cap_delta17_permil=tuple(map(float, predictions.reshape(-1))),
        model_delta18_conventional_permil=tuple(
            map(float, delta18_predictions.reshape(-1))
        ),
        marginal_density=marginal_density,
        marginal_probability_mass=marginal_mass,
        map_coordinates=map_coordinates,
        posterior_mean=means,
        posterior_median=medians,
        equal_tailed_credible_intervals=intervals,
        edge_probabilities=edge_probabilities,
        boundary_sensitive=boundary_sensitive,
        hpd_density_threshold=threshold,
        hpd_mask=tuple(map(bool, hpd_mask.reshape(-1))),
        hpd_probability_mass=hpd_mass,
        posterior_integral=float(np.sum(probability_mass)),
        effective_likelihood_sigma_permil=effective_sigma,
        probabilistic_model_discrepancy_included=discrepancy_included,
        surface_data_id=surface.surface_data_id,
        upstream_model_data_id=surface.upstream_model_data_id,
        probability_scope=scope,
        diagnostic=diagnostic,
    )
