"""Checked isotope-space reuse of the exact sulfate measurement likelihood.

Atmospheric coordinates only enter the sulfate likelihood through predicted
air anomaly and delta18O. Integrate process uncertainties at adaptive nodes in
those two coordinates, then validate interpolation against new exact nodes.
This is numerical reuse of the measurement likelihood, not an atmospheric fit.
"""

import logging
import numpy as np
from scipy.interpolate import RectBivariateSpline

from sulfate_uncertainty import (
    SulfateIntegrationError,
    SulfateLikelihoodResult,
    SulfateIntegrationSettings,
    exact_sulfate_likelihood,
    _COMPUTE_BUDGET,
    _bounds,
    _center,
)


def _unique_nodes(values):
    """Merge roundoff-equivalent seeds without moving either domain endpoint."""
    values = np.sort(np.asarray(values, float))
    tolerance = max(64 * np.finfo(float).eps * np.max(np.abs(values)), 1e-10)
    selected = [values[0]]
    for value in values[1:]:
        if value - selected[-1] > tolerance:
            selected.append(value)
    selected[-1] = values[-1]
    return np.asarray(selected)


def scalable_sulfate_likelihood(air17, air18, request):
    a, d = np.broadcast_arrays(np.asarray(air17, float), np.asarray(air18, float))
    settings = SulfateIntegrationSettings(
        relative_tolerance=5e-5,
        absolute_tolerance=2.5e-11,
        max_level=7,
        adaptive_delta18=True,
        focused_fraction_quadrature=True,
    )
    if a.size < 2048:
        return exact_sulfate_likelihood(a, d, request, settings=settings)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(d)) or np.any(d <= -1000):
        raise ValueError("atmospheric isotope predictions must be finite and physical")
    if np.ptp(a) < 1e-9 or np.ptp(d) < 1e-9:
        pairs, inverse = np.unique(
            np.column_stack((a.ravel(), d.ravel())), axis=0, return_inverse=True
        )
        result = exact_sulfate_likelihood(
            pairs[:, 0], pairs[:, 1], request, settings=settings
        )
        return SulfateLikelihoodResult(
            result.log_likelihood[inverse].reshape(a.shape), result.diagnostics
        )
    budget = _COMPUTE_BUDGET.get()
    cached = budget.likelihood_tables.get(request) if budget is not None else None
    if cached is not None:
        bounds, spline, floor, scale, diagnostics = cached
        if (
            bounds[0] <= a.min()
            and a.max() <= bounds[1]
            and bounds[2] <= d.min()
            and d.max() <= bounds[3]
        ):
            budget.claim(0)
            compatibility = np.maximum(
                0.0, floor * np.expm1(spline.ev(a.ravel(), d.ravel()))
            ).reshape(a.shape)
            if np.max(compatibility) >= 1e-7:
                with np.errstate(divide="ignore"):
                    logs = np.log(compatibility) - np.log(scale)
                detail = dict(
                    diagnostics["likelihood_interpolation"],
                    requested_atmospheric_states=a.size,
                    reused_within_request=True,
                )
                return SulfateLikelihoodResult(
                    logs, dict(diagnostics, likelihood_interpolation=detail)
                )
    scale = request.cap_delta17_sigma_permil * np.sqrt(
        2 * np.pi * (1 - request.isotope_error_correlation**2)
    )
    floor, rtol, atol = 1e-10, 1e-4, 7.5e-11
    # A small, explicitly evaluated halo covers nearby isotope extrema found
    # by subsequent atmospheric-grid refinement. It adds no posterior states.
    ax_pad, d_pad = max(1e-6, 0.01 * np.ptp(a)), max(1e-6, 0.02 * np.ptp(d))
    ax_low, ax_high = a.min() - ax_pad, a.max() + ax_pad
    d_low, d_high = max(-999.999999, d.min() - d_pad), d.max() + d_pad
    x = _unique_nodes(
        np.r_[np.linspace(ax_low, ax_high, 17), np.quantile(a, np.linspace(0, 1, 17))]
    )
    y = np.linspace(d_low, d_high, 4)
    # First-order coordinates only seed sampling; every likelihood value and
    # acceptance check below comes from the exact atom-balance integration.
    fl, fh = _bounds(request.incorporation, fraction=True)
    bl, bh = _bounds(request.background)
    shift = request.fractionation_metadata()["anomaly_shift_528_permil"]
    seeds = []
    for f in (fl, _center(request.incorporation), fh):
        if f > 0:
            for b in (bl, _center(request.background), bh):
                seeds.extend(
                    (
                        request.measured_cap_delta17_permil
                        + (
                            np.array([-8, -4, 0, 4, 8])
                            * request.cap_delta17_sigma_permil
                        )
                        - (1 - f) * b
                    )
                    / f
                    - shift
                )
    x = _unique_nodes(np.r_[x, np.clip(seeds, ax_low, ax_high)])
    cache = {}
    diagnostics = None
    total_roots = 0
    max_exact_level = 0
    alternate_states = 0
    max_change, max_inner_error = 0.0, 0.0
    delta18_checks, delta18_retries = 0, 0

    def evaluate(xs, ys):
        nonlocal diagnostics, total_roots, max_exact_level, alternate_states
        nonlocal max_change, max_inner_error, delta18_checks, delta18_retries
        xx, yy = np.broadcast_arrays(xs, ys)
        pairs = list(zip(xx.ravel(), yy.ravel()))
        missing = list(dict.fromkeys(p for p in pairs if p not in cache))
        if len(cache) + len(missing) > 40000:
            raise SulfateIntegrationError(
                "Sulfate likelihood interpolation did not converge; no posterior was returned."
            )
        if missing:
            mx, my = np.asarray(missing).T
            result = exact_sulfate_likelihood(mx, my, request, settings=settings)
            diagnostics = result.diagnostics
            total_roots += diagnostics["exact_forward_root_evaluations"]
            max_exact_level = max(max_exact_level, diagnostics["maximum_level"])
            alternate_states += diagnostics["alternate_integration_order_states"]
            max_change = max(
                max_change, diagnostics["maximum_final_compatibility_change"]
            )
            max_inner_error = max(
                max_inner_error,
                diagnostics["maximum_inner_compatibility_error_estimate"],
            )
            delta18_checks += diagnostics["delta18_quadrature_check_states"]
            delta18_retries += diagnostics["full_delta18_quadrature_retry_states"]
            cache.update(zip(missing, np.exp(result.log_likelihood) * scale))
        if budget is not None:
            budget.claim(0)
        return np.asarray([cache[p] for p in pairs]).reshape(xx.shape)

    worst_error = 0.0
    for refinement in range(9):
        values = evaluate(x[:, None], y[None, :])
        spline = RectBivariateSpline(x, y, np.log1p(values / floor), kx=3, ky=3, s=0)
        xm, ym = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
        checks = ((xm, y), (x, ym), (xm, ym))
        bad_x, bad_y = np.zeros(len(x) - 1, bool), np.zeros(len(y) - 1, bool)
        worst_error = 0.0
        for kind, (cx, cy) in enumerate(checks):
            exact = evaluate(cx[:, None], cy[None, :])
            interpolated = np.maximum(0.0, floor * np.expm1(spline(cx, cy)))
            error = np.abs(exact - interpolated) / (atol + rtol * exact)
            worst_error = max(worst_error, float(error.max()))
            bad = error > 1.0
            if kind == 0:
                bad_x |= np.any(bad, axis=1)
            elif kind == 1:
                bad_y |= np.any(bad, axis=0)
            else:
                # A center residual caused by an already unresolved x interval
                # must not force refinement of the entire, smooth delta18 axis.
                unresolved = bad & ~bad_x[:, None] & ~bad_y[None, :]
                bad_x |= np.any(unresolved, axis=1)
        if not (np.any(bad_x) or np.any(bad_y)):
            # Off-midpoint checks at actual requested states detect defects
            # not exposed by the structured refinement lattice.
            indices = np.unique(
                np.linspace(0, a.size - 1, min(257, a.size)).astype(int)
            )
            qa, qd = a.ravel()[indices], d.ravel()[indices]
            exact = evaluate(qa, qd)
            interpolated = np.maximum(0.0, floor * np.expm1(spline.ev(qa, qd)))
            error = np.abs(exact - interpolated) / (atol + rtol * exact)
            worst_error = max(worst_error, float(error.max()))
            if np.any(error > 1):
                bad_x[
                    np.clip(np.searchsorted(x, qa[error > 1]) - 1, 0, len(bad_x) - 1)
                ] = True
                bad_y[
                    np.clip(np.searchsorted(y, qd[error > 1]) - 1, 0, len(bad_y) - 1)
                ] = True
            else:
                break
        logging.getLogger(__name__).debug(
            "Sulfate table refinement %s: shape %s, error ratio %.6g, failed x %s, y %s",
            refinement,
            (len(x), len(y)),
            worst_error,
            xm[bad_x],
            ym[bad_y],
        )
        x = _unique_nodes(np.r_[x, xm[bad_x]])
        y = _unique_nodes(np.r_[y, ym[bad_y]])
    else:
        raise SulfateIntegrationError(
            "Sulfate likelihood interpolation did not converge; no posterior was returned."
        )
    compatibility = np.maximum(
        0.0, floor * np.expm1(spline.ev(a.ravel(), d.ravel()))
    ).reshape(a.shape)
    # A tail-only request needs direct evaluation: the absolute interpolation
    # tolerance must never create an artificial posterior from zero evidence.
    if np.max(compatibility) < 1e-7:
        return exact_sulfate_likelihood(a, d, request, settings=settings)
    with np.errstate(divide="ignore"):
        log_likelihood = np.log(compatibility) - np.log(scale)
    diagnostics = dict(
        diagnostics,
        exact_forward_root_evaluations=total_roots,
        maximum_level=max_exact_level,
        alternate_integration_order_states=alternate_states,
        maximum_final_compatibility_change=max_change,
        maximum_inner_compatibility_error_estimate=max_inner_error,
        delta18_quadrature_check_states=delta18_checks,
        full_delta18_quadrature_retry_states=delta18_retries,
        likelihood_interpolation={
            "coordinates": [
                "air_cap_delta17_permil",
                "air_delta18_conventional_permil",
            ],
            "relative_tolerance": rtol,
            "absolute_compatibility_tolerance": atol,
            "maximum_check_error_over_tolerance": worst_error,
            "exact_isotope_states": len(cache),
            "requested_atmospheric_states": a.size,
            "grid_shape": [len(x), len(y)],
            "refinements": refinement,
            "reused_within_request": False,
            "scope": "Checked numerical interpolation of the exact sulfate likelihood; no change to priors or atom balance",
        },
    )
    if budget is not None:
        if len(budget.likelihood_tables) >= 2:
            budget.likelihood_tables.pop(next(iter(budget.likelihood_tables)))
        budget.likelihood_tables[request] = (
            (x[0], x[-1], y[0], y[-1]),
            spline,
            floor,
            scale,
            diagnostics,
        )
    return SulfateLikelihoodResult(log_likelihood, diagnostics)
