"""Finite-volume Gaussian likelihood integration for a nuisance coordinate.

Integrate the locally affine forward predictions analytically, refining the
forward approximation rather than sampling an arbitrarily narrow Gaussian.
All intervals are retained, including separate branches of a nonmonotone model.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

from posterior_surface_slice import prepare_coordinate_slice


class PosteriorResolutionError(RuntimeError):
    """A bounded numerical calculation could not meet its resolution target."""


def _log_normal_interval(lower, upper):
    # Reflect positive-tail intervals to avoid subtracting two rounded ones.
    a = np.where(lower > 0, -upper, lower)
    b = np.where(lower > 0, -lower, upper)
    la, lb = log_ndtr(a), log_ndtr(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lb + np.log(-np.expm1(np.minimum(la-lb, 0.0)))


def affine_gaussian_log_integral(left, right, residual_left, residual_right,
                                 *, prior_mean=None, prior_sigma=None):
    """Log integral of Gaussian observations and an optional Gaussian prior.

    Residuals are standardized, shaped (intervals, observations). Normalizing
    constants independent of the coordinate are omitted and cancel later.
    """
    width = right-left
    residual = (residual_left+residual_right)/2
    slope = (residual_right-residual_left)/width[:, None]
    if prior_mean is not None:
        residual = np.column_stack((residual, ((left+right)/2-prior_mean)/prior_sigma))
        slope = np.column_stack((slope, np.full(width.shape, 1/prior_sigma)))
    precision = np.sum(slope*slope, axis=1)
    moving = precision > 1e-28
    safe_precision = np.where(moving, precision, 1.0)
    center = -np.sum(residual*slope, axis=1)/safe_precision
    # Evaluate at the conditional mean to avoid cancellation in C - B**2/A.
    floor = np.sum((residual+slope*center[:, None])**2, axis=1)
    root = np.sqrt(safe_precision)
    logs = (-floor/2 + .5*np.log(2*np.pi/safe_precision)
            + _log_normal_interval((-width/2-center)*root, (width/2-center)*root))
    flat = -np.sum(residual*residual, axis=1)/2 + np.log(width)
    return np.where(moving, logs, flat)


def integrated_coordinate_log_likelihood(surface, request, fields, axis, coordinate):
    """Return cell-averaged likelihood times the nuisance prior, with diagnostics."""
    edges = np.r_[axis[0], (axis[:-1]+axis[1:])/2, axis[-1]]
    arrays = np.broadcast_arrays(*(np.asarray(fields[c]) for c in ("pO2", "pCO2", "GPP")))
    shape = arrays[0].shape
    base = {c: a.ravel() for c, a in zip(("pO2", "pCO2", "GPP"), arrays, strict=True)}
    indices = np.arange(base[coordinate].size)
    bins = np.searchsorted(axis, base[coordinate])
    left, right = edges[bins], edges[bins+1]
    widths = right-left
    prefix = {"pO2": "po2", "pCO2": "pco2", "GPP": "gpp"}[coordinate]
    prior = getattr(request, prefix+"_prior")
    if prior not in ("normal", "uniform"):
        raise ValueError("cell integration requires a uniform or normal nuisance prior")
    mean = getattr(request, prefix+"_prior_mean") if prior == "normal" else None
    sigma = getattr(request, prefix+"_prior_sigma") if prior == "normal" else None
    settings = {"relative_tolerance": 1e-3, "response_tolerance_sigma": .01,
                "maximum_levels": 10, "maximum_forward_points": 12_000_000,
                "relative_tail_cutoff": 30.0}
    point_count = 0
    prepared = prepare_coordinate_slice(
        surface, base, coordinate, request.target_air_delta18_conventional_permil is not None)

    def evaluate(ids, values):
        nonlocal point_count
        point_count += len(ids)
        if point_count > settings["maximum_forward_points"]:
            raise PosteriorResolutionError("Nuisance-coordinate integration exceeded its calculation budget")
        answer = []
        for start in range(0, len(ids), 100_000):
            part = ids[start:start+100_000]
            kwargs = {"p_o2_pal": base["pO2"][part], "p_co2_ppm": base["pCO2"][part],
                      "gpp_pgC_per_year": base["GPP"][part]}
            kwargs[{"pO2": "p_o2_pal", "pCO2": "p_co2_ppm", "GPP": "gpp_pgC_per_year"}[coordinate]] = values[start:start+100_000]
            if prepared is None:
                d17 = surface.evaluate_central_cap_delta17_grid(**kwargs)
                d18_prime = None
            else:
                d17, d18_prime = prepared.evaluate(part, values[start:start+100_000])
            effective_sigma = np.hypot(request.measurement_sigma_permil, request.model_discrepancy_sigma_permil)
            residuals = [(d17-request.target_air_cap_delta17_permil)/effective_sigma]
            if request.target_air_delta18_conventional_permil is not None:
                if d18_prime is None:
                    d18_prime = surface.evaluate_central_delta18_prime_grid(**kwargs)
                d18 = 1000*np.expm1(d18_prime/1000)
                residuals.append((d18-request.target_air_delta18_conventional_permil)/request.delta18_measurement_sigma_permil)
            answer.append(np.column_stack(residuals))
        return np.concatenate(answer)

    def integral(a, b, fa, fb):
        return affine_gaussian_log_integral(a, b, fa, fb, prior_mean=mean, prior_sigma=sigma)

    fleft, fright = evaluate(indices, left), evaluate(indices, right)
    result = np.full(len(indices), -np.inf)
    peak_log_density = -np.inf
    max_change, max_response = 0., 0.
    for level in range(1, settings["maximum_levels"]+1):
        middle = (left+right)/2
        fmiddle = evaluate(indices, middle)
        coarse = integral(left, right, fleft, fright)
        first = integral(left, middle, fleft, fmiddle)
        second = integral(middle, right, fmiddle, fright)
        fine = np.logaddexp(first, second)
        peak_log_density = max(peak_log_density, float(np.max(fine-np.log(right-left))))
        relevant = fine-np.log(right-left) >= peak_log_density-settings["relative_tail_cutoff"]
        with np.errstate(over="ignore", invalid="ignore"):
            change = np.abs(np.expm1(coarse-fine))
        change = np.where(np.isneginf(coarse) & np.isneginf(fine), 0., change)
        response = np.max(np.abs(fmiddle-(fleft+fright)/2), axis=1)
        done = ~relevant | ((change <= settings["relative_tolerance"])
                            & (response <= settings["response_tolerance_sigma"]))
        np.logaddexp.at(result, indices[done], fine[done])
        if np.any(done & relevant):
            max_change = max(max_change, float(np.max(change[done & relevant])))
            max_response = max(max_response, float(np.max(response[done & relevant])))
        remaining = ~done
        if not np.any(remaining):
            break
        indices = np.tile(indices[remaining], 2)
        left, right = (np.r_[left[remaining], middle[remaining]],
                       np.r_[middle[remaining], right[remaining]])
        fleft, fright = (np.concatenate((fleft[remaining], fmiddle[remaining])),
                         np.concatenate((fmiddle[remaining], fright[remaining])))
    else:
        raise PosteriorResolutionError("Nuisance-coordinate integration did not converge; no posterior was returned")
    return (result-np.log(widths)).reshape(shape), {
        "status": "converged", "coordinate": coordinate,
        "method": "adaptive piecewise-affine forward response with analytic Gaussian cell integrals",
        "settings": settings, "maximum_level": level, "forward_points": point_count,
        "fixed_tensor_factors_reused": prepared is not None,
        "maximum_final_relative_integral_change": max_change,
        "maximum_final_response_error_sigma": max_response,
    }
