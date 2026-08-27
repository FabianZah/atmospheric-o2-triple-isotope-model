"""pCO2 inference with explicit geological GPP and pO2 constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np

from updated_output_surface import DEFAULT_OUTPUT_SURFACE_PATH, load_updated_output_surface
from updated_output_surface_joint_posterior import (
    UpdatedJointPosteriorInput,
    joint_updated_posterior,
)


ConstraintKind = Literal["fixed", "normal", "range"]
Coordinate = Literal["pCO2", "GPP", "pO2"]
COORDINATES: tuple[Coordinate, ...] = ("pCO2", "GPP", "pO2")
MIN_RESOLVED_AXIS_INTERVALS = 8


@dataclass(frozen=True)
class CoordinateConstraint:
    kind: ConstraintKind
    center: float | None = None
    sigma: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class ConstrainedPCO2Input:
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float
    gpp_constraint: CoordinateConstraint
    po2_constraint: CoordinateConstraint
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = None
    pco2_bounds_ppm: tuple[float, float] = (50.0, 60000.0)
    credible_mass: float = 0.95
    pco2_grid_size: int = 181
    gpp_grid_size: int = 81
    po2_grid_size: int = 17


@dataclass(frozen=True)
class ConstrainedCoordinateInput:
    solve_for: Coordinate
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float
    constraints: dict[Coordinate, CoordinateConstraint]
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = None
    credible_mass: float = 0.95
    pco2_grid_size: int = 181
    gpp_grid_size: int = 81
    po2_grid_size: int = 41


@dataclass(frozen=True)
class ConstrainedCoordinateResult:
    inputs: ConstrainedCoordinateInput
    status: str
    solve_for: Coordinate
    solve_axis: tuple[float, ...]
    solve_marginal_probability_mass: tuple[float, ...]
    solve_marginal_density: tuple[float, ...]
    posterior_median: float
    equal_tailed_credible_interval: tuple[float, float]
    field_x_coordinate: Coordinate | None
    field_y_coordinate: Coordinate | None
    field_x_axis: tuple[float, ...] | None
    field_y_axis: tuple[float, ...] | None
    field_probability_mass: tuple[float, ...] | None
    field_density: tuple[float, ...] | None
    field_shape: tuple[int, int] | None
    field_hpd_mask: tuple[bool, ...] | None
    field_hpd_probability_mass: float | None
    constraint_posterior_medians: dict[str, float]
    constraint_equal_tailed_credible_intervals: dict[str, tuple[float, float]]
    effective_constraint_bounds: dict[str, tuple[float, float]]
    solve_boundary_sensitive: bool
    solve_boundary_direction: Literal["lower", "upper"] | None
    solve_boundary_probability_mass: float
    solve_mode_at_boundary: bool
    numerical_refinement_applied: bool
    initial_solve_axis_size: int
    final_solve_axis_size: int
    initial_solve_bounds: tuple[float, float]
    final_solve_bounds: tuple[float, float]
    probability_scope: str
    surface_data_id: str
    upstream_model_data_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstrainedPCO2Result:
    inputs: ConstrainedPCO2Input
    status: str
    pco2_axis_ppm: tuple[float, ...]
    pco2_marginal_probability_mass: tuple[float, ...]
    pco2_marginal_density: tuple[float, ...]
    pco2_posterior_median_ppm: float
    pco2_equal_tailed_credible_interval_ppm: tuple[float, float]
    gpp_axis_pgC_per_year: tuple[float, ...] | None
    pco2_gpp_probability_mass: tuple[float, ...] | None
    pco2_gpp_density: tuple[float, ...] | None
    pco2_gpp_shape: tuple[int, int] | None
    pco2_gpp_hpd_mask: tuple[bool, ...] | None
    pco2_gpp_hpd_probability_mass: float | None
    gpp_posterior_median_pgC_per_year: float | None
    gpp_equal_tailed_credible_interval_pgC_per_year: tuple[float, float] | None
    po2_posterior_median_pal: float | None
    po2_equal_tailed_credible_interval_pal: tuple[float, float] | None
    gpp_effective_bounds_pgC_per_year: tuple[float, float]
    po2_effective_bounds_pal: tuple[float, float]
    pco2_boundary_sensitive: bool
    probability_scope: str
    surface_data_id: str
    upstream_model_data_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    weights = np.empty_like(axis)
    weights[0] = 0.5 * (axis[1] - axis[0])
    weights[-1] = 0.5 * (axis[-1] - axis[-2])
    weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return weights


def _resolve_constraint(
    constraint: CoordinateConstraint,
    domain: tuple[float, float],
    label: str,
) -> tuple[float, tuple[float, float], str, float | None, float | None]:
    lower_domain, upper_domain = map(float, domain)
    if constraint.kind == "fixed":
        if constraint.center is None or not isfinite(constraint.center):
            raise ValueError(f"fixed {label} constraint requires a finite value")
        center = float(constraint.center)
        if not lower_domain <= center <= upper_domain:
            raise ValueError(f"fixed {label}={center:g} is outside model domain {domain}")
        return center, (center, center), "fixed", None, None

    if constraint.kind == "normal":
        if constraint.center is None or constraint.sigma is None:
            raise ValueError(f"{label} 1-sigma constraint requires center and sigma")
        center = float(constraint.center)
        sigma = float(constraint.sigma)
        if not isfinite(center) or not isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"{label} 1-sigma constraint requires finite center and positive sigma")
        if not lower_domain <= center <= upper_domain:
            raise ValueError(f"{label} center={center:g} is outside model domain {domain}")
        bounds = (
            max(lower_domain, center - 4.0 * sigma),
            min(upper_domain, center + 4.0 * sigma),
        )
        if bounds[1] <= bounds[0]:
            raise ValueError(f"{label} 1-sigma constraint has no support inside model domain")
        return center, bounds, "normal", center, sigma

    if constraint.kind == "range":
        if constraint.lower is None or constraint.upper is None:
            raise ValueError(f"{label} range constraint requires lower and upper bounds")
        lower = float(constraint.lower)
        upper = float(constraint.upper)
        if not all(isfinite(value) for value in (lower, upper)) or upper <= lower:
            raise ValueError(f"{label} range must have finite lower < upper")
        if lower < lower_domain or upper > upper_domain:
            raise ValueError(f"{label} range {(lower, upper)} exceeds model domain {domain}")
        return 0.5 * (lower + upper), (lower, upper), "uniform", None, None

    raise ValueError(f"unknown {label} constraint kind: {constraint.kind}")


def _pair_hpd(
    mass: np.ndarray,
    pco2_axis: np.ndarray,
    gpp_axis: np.ndarray,
    credible_mass: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.multiply.outer(
        _trapezoid_weights(pco2_axis), _trapezoid_weights(gpp_axis)
    )
    density = mass / weights
    density /= float(np.sum(density * weights))
    order = np.argsort(density.reshape(-1))[::-1]
    flat_mass = mass.reshape(-1)
    cumulative = np.cumsum(flat_mass[order])
    position = min(
        int(np.searchsorted(cumulative, credible_mass, side="left")),
        len(order) - 1,
    )
    threshold = float(density.reshape(-1)[order[position]])
    mask = density >= threshold
    return density, mask, float(np.sum(mass[mask]))


def _domain_key(coordinate: Coordinate) -> str:
    return {
        "pCO2": "pco2_ppm",
        "GPP": "gpp_pgC_per_year",
        "pO2": "po2_pal",
    }[coordinate]


def _bounds_field(coordinate: Coordinate) -> str:
    return {
        "pCO2": "pco2_bounds_ppm",
        "GPP": "gpp_bounds_pgC_per_year",
        "pO2": "po2_bounds_pal",
    }[coordinate]


def _adaptive_solve_refinement(
    joint,
    solve_for: Coordinate,
    *,
    surface_path: Path,
) -> tuple[Any, bool, tuple[float, float], int]:
    """Refine a narrow solved-coordinate posterior without enlarging the grid."""

    initial_axis = np.asarray(joint.axes[solve_for], dtype=float)
    initial_bounds = (float(initial_axis[0]), float(initial_axis[-1]))
    applied = False
    current = joint
    for _ in range(3):
        axis = np.asarray(current.axes[solve_for], dtype=float)
        mass = np.asarray(current.marginal_probability_mass[solve_for], dtype=float)
        if max(current.edge_probabilities[solve_for]) >= 0.05:
            break
        cumulative = np.cumsum(mass)
        cumulative /= cumulative[-1]
        lower_index = int(np.searchsorted(cumulative, 1.0e-6, side="left"))
        upper_index = int(
            np.searchsorted(cumulative, 1.0 - 1.0e-6, side="left")
        )
        if upper_index - lower_index >= MIN_RESOLVED_AXIS_INTERVALS:
            break
        lower_index = max(0, lower_index - 2)
        upper_index = min(len(axis) - 1, upper_index + 2)
        if lower_index == 0 or upper_index == len(axis) - 1:
            break
        updates = {
            _bounds_field(solve_for): (
                float(axis[lower_index]),
                float(axis[upper_index]),
            )
        }
        current = joint_updated_posterior(
            replace(current.inputs, **updates), surface_path=surface_path
        )
        applied = True

    return current, applied, initial_bounds, len(initial_axis)


def _preferred_companion(
    solve_for: Coordinate, free_constraints: list[Coordinate]
) -> Coordinate | None:
    preference: dict[Coordinate, tuple[Coordinate, Coordinate]] = {
        "pCO2": ("GPP", "pO2"),
        "GPP": ("pCO2", "pO2"),
        "pO2": ("pCO2", "GPP"),
    }
    return next(
        (item for item in preference[solve_for] if item in free_constraints), None
    )


def constrained_coordinate_posterior(
    request: ConstrainedCoordinateInput,
    *,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
) -> ConstrainedCoordinateResult:
    """Infer one coordinate while propagating constraints on the other two."""

    if request.solve_for not in COORDINATES:
        raise ValueError(f"unknown solve coordinate: {request.solve_for}")
    expected = set(COORDINATES) - {request.solve_for}
    if set(request.constraints) != expected:
        raise ValueError(
            f"constraints must contain exactly {sorted(expected)} when solving "
            f"for {request.solve_for}"
        )

    surface = load_updated_output_surface(str(Path(surface_path).resolve()))
    resolved: dict[
        Coordinate,
        tuple[float, tuple[float, float], str, float | None, float | None],
    ] = {}
    for coordinate, constraint in request.constraints.items():
        resolved[coordinate] = _resolve_constraint(
            constraint,
            tuple(surface.domain[_domain_key(coordinate)]),
            coordinate,
        )

    free_constraints = [
        coordinate
        for coordinate in COORDINATES
        if coordinate != request.solve_for
        and request.constraints[coordinate].kind != "fixed"
    ]
    free = tuple(
        coordinate
        for coordinate in COORDINATES
        if coordinate == request.solve_for or coordinate in free_constraints
    )
    fixed_values: dict[Coordinate, float] = {
        "pCO2": 294.0,
        "GPP": 290.0,
        "pO2": 1.0,
    }
    for coordinate, values in resolved.items():
        fixed_values[coordinate] = values[0]

    bounds: dict[Coordinate, tuple[float, float] | None] = {
        "pCO2": None,
        "GPP": None,
        "pO2": None,
    }
    priors: dict[Coordinate, str] = {
        "pCO2": "uniform",
        "GPP": "uniform",
        "pO2": "uniform",
    }
    prior_means: dict[Coordinate, float | None] = dict.fromkeys(COORDINATES)
    prior_sigmas: dict[Coordinate, float | None] = dict.fromkeys(COORDINATES)
    for coordinate in free_constraints:
        _, coordinate_bounds, prior, mean, sigma = resolved[coordinate]
        bounds[coordinate] = coordinate_bounds
        priors[coordinate] = prior
        prior_means[coordinate] = mean
        prior_sigmas[coordinate] = sigma

    joint = joint_updated_posterior(
        UpdatedJointPosteriorInput(
            target_air_cap_delta17_permil=request.target_air_cap_delta17_permil,
            measurement_sigma_permil=request.measurement_sigma_permil,
            target_air_delta18_conventional_permil=(
                request.target_air_delta18_conventional_permil
            ),
            delta18_measurement_sigma_permil=request.delta18_measurement_sigma_permil,
            free_coordinates=free,
            credible_mass=request.credible_mass,
            p_o2_pal=fixed_values["pO2"],
            p_co2_ppm=fixed_values["pCO2"],
            gpp_pgC_per_year=fixed_values["GPP"],
            pco2_bounds_ppm=bounds["pCO2"],
            gpp_bounds_pgC_per_year=bounds["GPP"],
            po2_bounds_pal=bounds["pO2"],
            pco2_prior=priors["pCO2"],
            gpp_prior=priors["GPP"],
            po2_prior=priors["pO2"],
            pco2_prior_mean=prior_means["pCO2"],
            pco2_prior_sigma=prior_sigmas["pCO2"],
            gpp_prior_mean=prior_means["GPP"],
            gpp_prior_sigma=prior_sigmas["GPP"],
            po2_prior_mean=prior_means["pO2"],
            po2_prior_sigma=prior_sigmas["pO2"],
            pco2_grid_size=request.pco2_grid_size,
            gpp_grid_size=request.gpp_grid_size,
            po2_grid_size=request.po2_grid_size,
        ),
        surface_path=surface_path,
    )

    initial_solve_mass = np.asarray(
        joint.marginal_probability_mass[request.solve_for], dtype=float
    )
    boundary_sensitive = max(
        float(np.sum(initial_solve_mass[:2])),
        float(np.sum(initial_solve_mass[-2:])),
    ) >= 0.05
    joint, refinement_applied, initial_solve_bounds, initial_solve_axis_size = (
        _adaptive_solve_refinement(
            joint,
            request.solve_for,
            surface_path=surface_path,
        )
    )
    companion = _preferred_companion(request.solve_for, free_constraints)
    field_coordinates: tuple[Coordinate, Coordinate] | None = None
    field_axes: tuple[np.ndarray, np.ndarray] | None = None
    pair_mass: np.ndarray | None = None
    pair_density: np.ndarray | None = None
    pair_mask: np.ndarray | None = None
    pair_hpd_mass: float | None = None
    if companion is not None:
        field_coordinates = tuple(
            item for item in COORDINATES if item in {request.solve_for, companion}
        )
        field_axes = tuple(
            np.asarray(joint.axes[item], dtype=float) for item in field_coordinates
        )
        full_mass = np.asarray(joint.posterior_probability_mass, dtype=float).reshape(
            joint.posterior_shape
        )
        sum_axes = tuple(
            index for index, coordinate in enumerate(free) if coordinate not in field_coordinates
        )
        pair_mass = np.sum(full_mass, axis=sum_axes) if sum_axes else full_mass
        pair_mass = np.asarray(pair_mass, dtype=float)
        pair_mass /= float(np.sum(pair_mass))
        pair_density, pair_mask, pair_hpd_mass = _pair_hpd(
            pair_mass,
            field_axes[0],
            field_axes[1],
            request.credible_mass,
        )
    scope = (
        "Posterior from the central updated model, Gaussian isotope measurement "
        f"likelihoods, a bounded uniform {request.solve_for} prior in its reported "
        "units, and the explicitly supplied constraints on the other coordinates. "
        "Gaussian coordinate constraints are normalized over center +/- 4 sigma "
        "clipped to the operational domain. No log-uniform coordinate prior or "
        "probabilistic structural model discrepancy is included."
    )
    if refinement_applied:
        scope += (
            " Solved-coordinate quadrature was adaptively refined over resolved "
            "support without changing the declared prior or likelihood."
        )
    final_solve_axis = np.asarray(joint.axes[request.solve_for], dtype=float)
    final_solve_mass = np.asarray(
        joint.marginal_probability_mass[request.solve_for], dtype=float
    )
    final_solve_density = np.asarray(
        joint.marginal_density[request.solve_for], dtype=float
    )
    lower_edge_mass = float(np.sum(final_solve_mass[:2]))
    upper_edge_mass = float(np.sum(final_solve_mass[-2:]))
    boundary_direction: Literal["lower", "upper"] | None = None
    boundary_probability_mass = max(lower_edge_mass, upper_edge_mass)
    if boundary_sensitive:
        boundary_direction = (
            "lower" if lower_edge_mass >= upper_edge_mass else "upper"
        )
    mode_index = int(np.argmax(final_solve_density))
    mode_at_boundary = mode_index in {0, len(final_solve_density) - 1}
    return ConstrainedCoordinateResult(
        inputs=request,
        status=(
            "solve_boundary_sensitive" if boundary_sensitive else "posterior_computed"
        ),
        solve_for=request.solve_for,
        solve_axis=joint.axes[request.solve_for],
        solve_marginal_probability_mass=(
            joint.marginal_probability_mass[request.solve_for]
        ),
        solve_marginal_density=joint.marginal_density[request.solve_for],
        posterior_median=joint.posterior_median[request.solve_for],
        equal_tailed_credible_interval=(
            joint.equal_tailed_credible_intervals[request.solve_for]
        ),
        field_x_coordinate=(None if field_coordinates is None else field_coordinates[0]),
        field_y_coordinate=(None if field_coordinates is None else field_coordinates[1]),
        field_x_axis=(None if field_axes is None else tuple(map(float, field_axes[0]))),
        field_y_axis=(None if field_axes is None else tuple(map(float, field_axes[1]))),
        field_probability_mass=(
            None if pair_mass is None else tuple(map(float, pair_mass.reshape(-1)))
        ),
        field_density=(
            None if pair_density is None else tuple(map(float, pair_density.reshape(-1)))
        ),
        field_shape=(None if pair_mass is None else tuple(pair_mass.shape)),
        field_hpd_mask=(
            None if pair_mask is None else tuple(map(bool, pair_mask.reshape(-1)))
        ),
        field_hpd_probability_mass=pair_hpd_mass,
        constraint_posterior_medians={
            item: joint.posterior_median[item] for item in free_constraints
        },
        constraint_equal_tailed_credible_intervals={
            item: joint.equal_tailed_credible_intervals[item]
            for item in free_constraints
        },
        effective_constraint_bounds={
            item: values[1] for item, values in resolved.items()
        },
        solve_boundary_sensitive=boundary_sensitive,
        solve_boundary_direction=boundary_direction,
        solve_boundary_probability_mass=boundary_probability_mass,
        solve_mode_at_boundary=mode_at_boundary,
        numerical_refinement_applied=refinement_applied,
        initial_solve_axis_size=initial_solve_axis_size,
        final_solve_axis_size=len(final_solve_axis),
        initial_solve_bounds=initial_solve_bounds,
        final_solve_bounds=(
            float(final_solve_axis[0]), float(final_solve_axis[-1])
        ),
        probability_scope=scope,
        surface_data_id=joint.surface_data_id,
        upstream_model_data_id=joint.upstream_model_data_id,
    )


def constrained_pco2_posterior(
    request: ConstrainedPCO2Input,
    *,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
) -> ConstrainedPCO2Result:
    """Infer pCO2 while propagating explicit GPP and pO2 constraints."""

    surface = load_updated_output_surface(str(Path(surface_path).resolve()))
    gpp_center, gpp_bounds, gpp_prior, gpp_mean, gpp_sigma = _resolve_constraint(
        request.gpp_constraint,
        tuple(surface.domain["gpp_pgC_per_year"]),
        "GPP",
    )
    po2_center, po2_bounds, po2_prior, po2_mean, po2_sigma = _resolve_constraint(
        request.po2_constraint,
        tuple(surface.domain["po2_pal"]),
        "pO2",
    )

    free: list[str] = ["pCO2"]
    if request.gpp_constraint.kind != "fixed":
        free.append("GPP")
    if request.po2_constraint.kind != "fixed":
        free.append("pO2")

    joint = joint_updated_posterior(
        UpdatedJointPosteriorInput(
            target_air_cap_delta17_permil=request.target_air_cap_delta17_permil,
            measurement_sigma_permil=request.measurement_sigma_permil,
            target_air_delta18_conventional_permil=(
                request.target_air_delta18_conventional_permil
            ),
            delta18_measurement_sigma_permil=request.delta18_measurement_sigma_permil,
            free_coordinates=tuple(free),
            credible_mass=request.credible_mass,
            p_o2_pal=po2_center,
            gpp_pgC_per_year=gpp_center,
            pco2_bounds_ppm=request.pco2_bounds_ppm,
            gpp_bounds_pgC_per_year=(
                gpp_bounds if request.gpp_constraint.kind != "fixed" else None
            ),
            po2_bounds_pal=(
                po2_bounds if request.po2_constraint.kind != "fixed" else None
            ),
            pco2_prior="uniform",
            gpp_prior=gpp_prior,
            po2_prior=po2_prior,
            gpp_prior_mean=gpp_mean,
            gpp_prior_sigma=gpp_sigma,
            po2_prior_mean=po2_mean,
            po2_prior_sigma=po2_sigma,
            pco2_grid_size=request.pco2_grid_size,
            gpp_grid_size=request.gpp_grid_size,
            po2_grid_size=request.po2_grid_size,
        ),
        surface_path=surface_path,
    )

    pco2_axis = np.asarray(joint.axes["pCO2"], dtype=float)
    gpp_axis: np.ndarray | None = None
    pair_mass: np.ndarray | None = None
    pair_density: np.ndarray | None = None
    pair_mask: np.ndarray | None = None
    pair_hpd_mass: float | None = None
    if "GPP" in free:
        gpp_axis = np.asarray(joint.axes["GPP"], dtype=float)
        full_mass = np.asarray(joint.posterior_probability_mass, dtype=float).reshape(
            joint.posterior_shape
        )
        if "pO2" in free:
            pair_mass = np.sum(full_mass, axis=free.index("pO2"))
        else:
            pair_mass = full_mass
        pair_mass /= float(np.sum(pair_mass))
        pair_density, pair_mask, pair_hpd_mass = _pair_hpd(
            pair_mass, pco2_axis, gpp_axis, request.credible_mass
        )

    pco2_edges = joint.edge_probabilities["pCO2"]
    scope = (
        "Posterior from the central updated model, Gaussian isotope measurement "
        "likelihoods, a bounded uniform pCO2 prior, and the explicitly supplied "
        "GPP and pO2 constraints. Gaussian coordinate constraints are normalized "
        "over center +/- 4 sigma clipped to the operational domain. No log-uniform "
        "coordinate prior or probabilistic structural model discrepancy is included."
    )
    return ConstrainedPCO2Result(
        inputs=request,
        status="pco2_boundary_sensitive" if max(pco2_edges) >= 0.05 else "posterior_computed",
        pco2_axis_ppm=tuple(map(float, pco2_axis)),
        pco2_marginal_probability_mass=joint.marginal_probability_mass["pCO2"],
        pco2_marginal_density=joint.marginal_density["pCO2"],
        pco2_posterior_median_ppm=joint.posterior_median["pCO2"],
        pco2_equal_tailed_credible_interval_ppm=(
            joint.equal_tailed_credible_intervals["pCO2"]
        ),
        gpp_axis_pgC_per_year=(
            None if gpp_axis is None else tuple(map(float, gpp_axis))
        ),
        pco2_gpp_probability_mass=(
            None if pair_mass is None else tuple(map(float, pair_mass.reshape(-1)))
        ),
        pco2_gpp_density=(
            None if pair_density is None else tuple(map(float, pair_density.reshape(-1)))
        ),
        pco2_gpp_shape=(None if pair_mass is None else tuple(pair_mass.shape)),
        pco2_gpp_hpd_mask=(
            None if pair_mask is None else tuple(map(bool, pair_mask.reshape(-1)))
        ),
        pco2_gpp_hpd_probability_mass=pair_hpd_mass,
        gpp_posterior_median_pgC_per_year=joint.posterior_median.get("GPP"),
        gpp_equal_tailed_credible_interval_pgC_per_year=(
            joint.equal_tailed_credible_intervals.get("GPP")
        ),
        po2_posterior_median_pal=joint.posterior_median.get("pO2"),
        po2_equal_tailed_credible_interval_pal=(
            joint.equal_tailed_credible_intervals.get("pO2")
        ),
        gpp_effective_bounds_pgC_per_year=gpp_bounds,
        po2_effective_bounds_pal=po2_bounds,
        pco2_boundary_sensitive=max(pco2_edges) >= 0.05,
        probability_scope=scope,
        surface_data_id=joint.surface_data_id,
        upstream_model_data_id=joint.upstream_model_data_id,
    )
