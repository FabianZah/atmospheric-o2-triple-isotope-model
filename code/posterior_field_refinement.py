"""On-demand, memory-bounded recomputation of narrow marginal probability fields."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from threading import Lock
from time import monotonic

import numpy as np
from scipy.ndimage import label
from scipy.special import logsumexp

from posterior_coordinate_quadrature import PosteriorResolutionError, integrated_coordinate_log_likelihood
from updated_output_surface_joint_posterior import _quantile, _trapezoid_weights


PAIR_AXIS_FLOORS = {"pCO2": 721, "GPP": 961, "pO2": 721}
MAX_FIELD_POINTS = 1_000_000
MAX_FORWARD_POINTS = 150_000_000
MAX_SECONDS = 300.0
CHUNK_VOLUME_POINTS = 120_000
_CACHE_LOCK = Lock()
_LAST_REFINEMENT = None


def field_resolution_trigger(mask, mass):
    """Detect disconnected support or appreciable one-cell-wide support.

    This requests a numerical check; disconnected regions may also be real.
    Refinement never joins components or changes the density threshold by hand.
    """
    _, components = label(mask, np.ones((3, 3), dtype=int))
    padded = np.pad(mask, 1)
    thin = mask & ((~padded[:-2, 1:-1] & ~padded[2:, 1:-1])
                   | (~padded[1:-1, :-2] & ~padded[1:-1, 2:]))
    thin_mass = float(np.sum(mass[thin]))
    return {"needed": bool(components > 1 or thin_mass > .005),
            "components": int(components), "one_cell_wide_probability_mass": thin_mass}


def _refined_axis(axis, marginal_mass, occupied, coordinate):
    cumulative = np.cumsum(marginal_mass) / np.sum(marginal_mass)
    lower = int(np.searchsorted(cumulative, 1e-6))
    upper = min(len(axis)-1, int(np.searchsorted(cumulative, 1-1e-6)))
    support = np.flatnonzero(occupied)
    if len(support):
        lower, upper = min(lower, support[0]), max(upper, support[-1])
    lower, upper = max(0, lower-2), min(len(axis)-1, upper+2)
    count = max(PAIR_AXIS_FLOORS[coordinate], 2*(upper-lower)+1)
    spacing = np.geomspace if coordinate in ("pCO2", "GPP") else np.linspace
    # Retain outer quadrature nodes: concentration must not truncate the prior.
    return np.r_[axis[:lower], spacing(axis[lower], axis[upper], count), axis[upper+1:]]


def refine_integrated_pair_field(result, joint, surface):
    """Recompute a triggered Gaussian-air field and every reported marginal.

    Stream the nuisance-coordinate integrals by columns; never allocate or
    serialize the entire refined three-dimensional posterior. The original
    nuisance axis and its adaptive integration tolerances are preserved.
    """
    global _LAST_REFINEMENT
    if (joint.inputs.integrate_coordinate is None or result.field_shape is None
            or result.inputs.sulfate is not None):
        return result
    old_mass = np.asarray(result.field_probability_mass).reshape(result.field_shape)
    old_mask = np.asarray(result.field_hpd_mask).reshape(result.field_shape)
    trigger = field_resolution_trigger(old_mask, old_mass)
    if not trigger["needed"]:
        return result

    cache_key = (surface, json.dumps(asdict(result.inputs), sort_keys=True),
                 tuple(result.field_x_axis), tuple(result.field_y_axis),
                 tuple(joint.axes[joint.inputs.integrate_coordinate]))
    with _CACHE_LOCK:
        if _LAST_REFINEMENT is not None and _LAST_REFINEMENT[0] == cache_key:
            return _LAST_REFINEMENT[1]

    started = monotonic()
    coordinates = (result.field_x_coordinate, result.field_y_coordinate)
    old_axes = (np.asarray(result.field_x_axis), np.asarray(result.field_y_axis))
    axes = tuple(_refined_axis(axis, old_mass.sum(axis=1-dim),
                               old_mask.any(axis=1-dim), coordinate)
                 for dim, (coordinate, axis) in enumerate(zip(coordinates, old_axes, strict=True)))
    x, y = axes
    if len(x)*len(y) > MAX_FIELD_POINTS:
        raise PosteriorResolutionError("Probability-field refinement exceeds its grid budget; no refined result was returned")
    nuisance = joint.inputs.integrate_coordinate
    z = np.asarray(joint.axes[nuisance])
    wx, wy, wz = map(_trapezoid_weights, (x, y, z))
    logs = np.empty((len(x), len(y)))
    nuisance_logs = np.full(len(z), -np.inf)
    chunk_columns = max(1, CHUNK_VOLUME_POINTS // (len(y)*len(z)))
    diagnostics = {"status": "refined", "trigger": trigger,
                   "method": "streamed marginal field with adaptive nuisance-coordinate integration",
                   "original_shape": list(old_mass.shape), "refined_shape": [len(x), len(y)],
                   "forward_points": 0, "maximum_integration_level": 0,
                   "maximum_final_relative_integral_change": 0.0,
                   "maximum_final_response_error_sigma": 0.0,
                   "maximum_field_points": MAX_FIELD_POINTS,
                   "maximum_forward_points": MAX_FORWARD_POINTS, "maximum_seconds": MAX_SECONDS,
                   "outer_quadrature_nodes_retained": True}
    log_priors = []
    for coordinate, axis in zip(coordinates, axes, strict=True):
        prefix = {"pCO2": "pco2", "GPP": "gpp", "pO2": "po2"}[coordinate]
        prior = getattr(joint.inputs, prefix+"_prior")
        if prior == "normal":
            mean = getattr(joint.inputs, prefix+"_prior_mean")
            sigma = getattr(joint.inputs, prefix+"_prior_sigma")
            log_priors.append(-.5*((axis-mean)/sigma)**2)
        elif prior == "log_uniform":
            log_priors.append(-np.log(axis))
        else:
            log_priors.append(np.zeros_like(axis))

    for start in range(0, len(x), chunk_columns):
        if monotonic()-started > MAX_SECONDS:
            raise PosteriorResolutionError("Probability-field refinement reached its time budget; no refined result was returned")
        stop = min(start+chunk_columns, len(x))
        xx, yy, zz = np.meshgrid(x[start:stop], y, z, indexing="ij")
        fields = {coordinates[0]: xx, coordinates[1]: yy, nuisance: zz}
        values, local = integrated_coordinate_log_likelihood(surface, joint.inputs, fields, z, nuisance)
        diagnostics["forward_points"] += local["forward_points"]
        if diagnostics["forward_points"] > MAX_FORWARD_POINTS:
            raise PosteriorResolutionError("Probability-field refinement reached its evaluation budget; no refined result was returned")
        diagnostics["maximum_integration_level"] = max(diagnostics["maximum_integration_level"], local["maximum_level"])
        for key in ("maximum_final_relative_integral_change", "maximum_final_response_error_sigma"):
            diagnostics[key] = max(diagnostics[key], local[key])
        values += log_priors[0][start:stop, None, None] + log_priors[1][None, :, None]
        values += np.log(wz)[None, None, :]
        logs[start:stop] = logsumexp(values, axis=2)
        weighted = values + np.log(wx[start:stop, None, None]) + np.log(wy[None, :, None])
        nuisance_logs = np.logaddexp(nuisance_logs, logsumexp(weighted, axis=(0, 1)))

    from updated_constrained_pco2_posterior import _pair_hpd

    log_mass = logs + np.log(wx[:, None]) + np.log(wy[None, :])
    normalization = logsumexp(log_mass)
    mass = np.exp(log_mass-normalization)
    density, mask, hpd_mass = _pair_hpd(mass, x, y, result.inputs.credible_mass)
    nuisance_mass = np.exp(nuisance_logs-normalization)
    marginal_axes = {coordinates[0]: x, coordinates[1]: y, nuisance: z}
    marginal_mass = {coordinates[0]: mass.sum(axis=1), coordinates[1]: mass.sum(axis=0), nuisance: nuisance_mass}
    medians, intervals = {}, {}
    tail = (1-result.inputs.credible_mass)/2
    for coordinate, axis in marginal_axes.items():
        local_mass = marginal_mass[coordinate]
        local_mass /= np.sum(local_mass)
        medians[coordinate] = _quantile(axis, local_mass, .5)
        intervals[coordinate] = tuple(_quantile(axis, local_mass, q) for q in (tail, 1-tail))
    solved = result.solve_for
    solve_axis, solve_mass = marginal_axes[solved], marginal_mass[solved]
    solve_density = solve_mass/_trapezoid_weights(solve_axis)
    mode = int(np.argmax(solve_density))
    edge_mass = (float(solve_mass[:2].sum()), float(solve_mass[-2:].sum()))
    diagnostics.update({"elapsed_seconds": monotonic()-started,
                        "refined_components": int(label(mask, np.ones((3, 3), dtype=int))[1]),
                        "posterior_median_change": medians[solved]-result.posterior_median,
                        "credible_interval_changes": (np.asarray(intervals[solved])-result.equal_tailed_credible_interval).tolist()})
    sizes = result.effective_grid_sizes | {c: len(a) for c, a in marginal_axes.items()}
    updated = replace(result,
        solve_axis=tuple(map(float, solve_axis)), solve_marginal_probability_mass=tuple(map(float, solve_mass)),
        solve_marginal_density=tuple(map(float, solve_density)), posterior_median=medians[solved],
        equal_tailed_credible_interval=intervals[solved],
        field_x_axis=tuple(map(float, x)), field_y_axis=tuple(map(float, y)),
        field_probability_mass=tuple(map(float, mass.ravel())), field_density=tuple(map(float, density.ravel())),
        field_shape=mass.shape, field_hpd_mask=tuple(map(bool, mask.ravel())),
        field_hpd_probability_mass=hpd_mass, field_hpd_density_threshold=float(density[mask].min()),
        constraint_posterior_medians={c: medians[c] for c in result.constraint_posterior_medians},
        constraint_equal_tailed_credible_intervals={c: intervals[c] for c in result.constraint_equal_tailed_credible_intervals},
        solve_boundary_probability_mass=max(edge_mass), solve_mode_at_boundary=mode in (0, len(solve_axis)-1),
        numerical_refinement_applied=True, hpd_resolution_refinement_applied=True,
        multidimensional_resolution_applied=True, effective_grid_sizes=sizes,
        final_solve_axis_size=len(solve_axis), final_solve_bounds=(float(solve_axis[0]), float(solve_axis[-1])),
        field_refinement_diagnostics=diagnostics,
        probability_scope=result.probability_scope+" The marginal field was recomputed on concentrated finer axes, retaining outer quadrature nodes. All reported marginals use that recomputation; separate credible regions are retained when present.")
    # Keep one completed expensive result for an identical rerun/XLSX request.
    # The model object and all inputs participate in the key; no disk cache.
    with _CACHE_LOCK:
        _LAST_REFINEMENT = (cache_key, updated)
    return updated
