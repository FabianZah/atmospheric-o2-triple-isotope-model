"""Convergence-checked measurement likelihood for conditional sulfate transfer.

Narrow analytical likelihoods are integrated over a locally linear *forward*
response, whose values are exact isotope-atom roots. Successive refinement
checks both response interpolation and the nuisance integrals. There is no
Gaussian approximation to inferred atmospheric composition or air prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from math import isfinite, log, pi, sqrt
from typing import Literal
from time import monotonic

import numpy as np
from scipy.special import roots_hermitenorm, roots_legendre

from sulfate_to_air import (
    IncorporationConstraint,
    _log_normal_interval,
    exact_sulfate_grid,
    air_fractionation_factors,
    air_fractionation_metadata,
)
from isotopes import R17_VSMOW, R18_VSMOW


class SulfateIntegrationError(RuntimeError):
    """No uncertainty result may be reported from this unfinished integration."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


class SulfateComputeLimitError(SulfateIntegrationError):
    """The browser calculation has exhausted its declared resource budget."""


@dataclass
class _ComputeBudget:
    deadline: float
    remaining_roots: int
    likelihood_tables: dict = field(default_factory=dict)

    def claim(self, roots: int) -> None:
        self.remaining_roots -= roots
        if monotonic() >= self.deadline or self.remaining_roots < 0:
            raise SulfateComputeLimitError(
                "The sulfate uncertainty calculation reached its computational limit "
                "before completion. No result was returned.",
                diagnostics={
                    "limit": "root_evaluations" if self.remaining_roots < 0 else "time",
                    "remaining_roots": self.remaining_roots,
                },
            )


_COMPUTE_BUDGET: ContextVar[_ComputeBudget | None] = ContextVar(
    "sulfate_compute_budget", default=None
)


@contextmanager
def sulfate_computation_budget(
    *, max_seconds: float = 600.0, max_roots: int = 1_000_000_000
):
    """Share one budget across all likelihood/refinement calls in a browser request."""
    if not isfinite(max_seconds) or max_seconds <= 0 or max_roots <= 0:
        raise ValueError("sulfate calculation budgets must be positive")
    budget = _ComputeBudget(monotonic() + max_seconds, max_roots)
    token = _COMPUTE_BUDGET.set(budget)
    try:
        yield
        budget.claim(0)
    finally:
        _COMPUTE_BUDGET.reset(token)


@dataclass(frozen=True)
class BackgroundConstraint:
    kind: Literal["fixed", "normal", "range"]
    center: float | None = None
    sigma: float | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        values = (self.center, self.sigma, self.lower, self.upper)
        if any(v is not None and not isfinite(v) for v in values):
            raise ValueError("background constraints must be finite")
        if self.kind == "fixed":
            good = self.center is not None and all(v is None for v in values[1:])
        elif self.kind == "normal":
            good = (
                self.center is not None
                and self.sigma is not None
                and self.sigma > 0.0
                and self.lower is None
                and self.upper is None
            )
        elif self.kind == "range":
            good = (
                self.center is None
                and self.sigma is None
                and self.lower is not None
                and self.upper is not None
                and self.lower < self.upper
            )
        else:
            good = False
        if not good:
            raise ValueError(
                "background requires fixed center, normal center/sigma, or ordered range bounds"
            )


@dataclass(frozen=True)
class SulfateLikelihoodInput:
    measured_cap_delta17_permil: float
    measured_delta18_permil: float
    cap_delta17_sigma_permil: float
    delta18_sigma_permil: float
    isotope_error_correlation: float
    incorporation: IncorporationConstraint
    background: BackgroundConstraint
    alpha18_air_to_sulfate: float
    theta_air_to_sulfate: float | None
    assumption_note: str
    alpha17_air_to_sulfate: float | None = None
    fractionation_treatment: str = "specified"

    def __post_init__(self):
        numeric = (
            self.measured_cap_delta17_permil,
            self.measured_delta18_permil,
            self.cap_delta17_sigma_permil,
            self.delta18_sigma_permil,
            self.isotope_error_correlation,
            self.alpha18_air_to_sulfate,
        )
        if not all(isfinite(v) for v in numeric):
            raise ValueError("sulfate likelihood inputs must be finite")
        if self.cap_delta17_sigma_permil <= 0.0 or self.delta18_sigma_permil < 0.0:
            raise ValueError(
                "sulfate Delta-prime-17O sigma must be positive; delta18O sigma must be nonnegative"
            )
        if self.measured_delta18_permil - 8.0 * self.delta18_sigma_permil <= -1000.0:
            raise ValueError(
                "delta18O measurement support must remain above -1000 per mil through eight sigma"
            )
        if abs(self.isotope_error_correlation) >= 1.0 or (
            self.delta18_sigma_permil == 0.0 and self.isotope_error_correlation != 0.0
        ):
            raise ValueError("invalid isotope-error correlation")
        transfer = self.fractionation_metadata()
        if not isinstance(
            self.incorporation, IncorporationConstraint
        ) or not isinstance(self.background, BackgroundConstraint):
            raise ValueError(
                "explicit incorporation and background constraints are required"
            )
        fraction = transfer["required_incorporation_fraction"]
        if fraction is not None and (
            self.incorporation.kind != "fixed" or self.incorporation.center != fraction
        ):
            raise ValueError(
                "Peng sulfite-oxidation treatments require a fixed 25% O2 contribution"
            )
        if self.incorporation.kind == "fixed" and self.incorporation.center == 1.0:
            raise ValueError(
                "100% air requires a joint direct-air measurement model, not the conditional sulfate closure"
            )
        if (
            not isinstance(self.assumption_note, str)
            or not self.assumption_note.strip()
        ):
            raise ValueError(
                "document the conditional formation and primary-preservation assumptions"
            )

    def fractionation_metadata(self) -> dict:
        return air_fractionation_metadata(
            self.alpha18_air_to_sulfate,
            self.theta_air_to_sulfate,
            self.alpha17_air_to_sulfate,
            self.fractionation_treatment,
        )


@dataclass(frozen=True)
class SulfateIntegrationSettings:
    relative_tolerance: float = 2e-4
    absolute_tolerance: float = 1e-10
    response_tolerance_sigma: float = 0.002
    max_level: int = 5
    chunk_size: int = 1024
    adaptive_delta18: bool = False
    focused_fraction_quadrature: bool = False

    def __post_init__(self):
        for value in (
            self.relative_tolerance,
            self.absolute_tolerance,
            self.response_tolerance_sigma,
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError("integration tolerances must be finite and positive")
        if not isinstance(self.max_level, int) or not 2 <= self.max_level <= 8:
            raise ValueError("max_level must be an integer from 2 to 8")
        if not isinstance(self.chunk_size, int) or not 1 <= self.chunk_size <= 2048:
            raise ValueError("chunk_size must be an integer from 1 to 2048")
        if not isinstance(self.adaptive_delta18, bool):
            raise ValueError("adaptive_delta18 must be boolean")
        if not isinstance(self.focused_fraction_quadrature, bool):
            raise ValueError("focused_fraction_quadrature must be boolean")


@dataclass(frozen=True)
class SulfateLikelihoodResult:
    log_likelihood: np.ndarray
    diagnostics: dict


def _spread(c):
    if c.kind == "fixed":
        return 0.0
    return c.sigma if c.kind == "normal" else (c.upper - c.lower) / sqrt(12.0)


def _center(c):
    return (c.lower + c.upper) / 2.0 if c.kind == "range" else c.center


def _bounds(c, fraction=False):
    if c.kind == "fixed":
        return c.center, c.center
    if c.kind == "range":
        return c.lower, c.upper
    low, high = c.center - 8.0 * c.sigma, c.center + 8.0 * c.sigma
    return (max(0.0, low), min(1.0, high)) if fraction else (low, high)


@lru_cache(maxsize=24)
def _rule(order, normal=False):
    if normal:
        x, w = roots_hermitenorm(order)
        return x, w / sqrt(2.0 * pi)
    x, w = roots_legendre(order)
    return (x + 1.0) / 2.0, w / 2.0


def _nodes(c, order, fraction=False):
    if c.kind == "fixed":
        return np.array([c.center]), np.ones(1)
    if c.kind == "range":
        x, w = _rule(order)
        return c.lower + (c.upper - c.lower) * x, w
    if fraction:
        x, w = _rule(order)
        edges = np.unique(
            np.clip(
                c.center + c.sigma * np.array([-8.0, -4.0, 0.0, 4.0, 8.0]), 0.0, 1.0
            )
        )
        width = np.diff(edges)
        nodes = edges[:-1, None] + width[:, None] * x[None, :]
        normalization = _log_normal_interval(
            np.asarray(-c.center / c.sigma), np.asarray((1.0 - c.center) / c.sigma)
        )
        prior = np.exp(-0.5 * ((nodes - c.center) / c.sigma) ** 2 - normalization) / (
            c.sigma * sqrt(2 * pi)
        )
        return nodes.ravel(), (width[:, None] * w[None, :] * prior).ravel()
    x, w = _rule(order, True)
    return c.center + c.sigma * x, w


def _cell_integral(x0, x1, y0, y1, measured, sigma, constraint, fraction=False):
    """Exact Gaussian observation integral for an affine forward response on a cell."""
    dx = x1 - x0
    slope = (y1 - y0) / np.where(dx > 0, dx, 1.0)
    midpoint = (y1 + y0) / 2.0
    intercept = midpoint - slope * (x0 + x1) / 2.0
    gaussian_mid = np.exp(-0.5 * ((measured - midpoint) / sigma) ** 2)
    if constraint.kind == "range":
        a, b = (measured - y0) / sigma, (measured - y1) / sigma
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (
                np.exp(
                    _log_normal_interval(np.minimum(a, b), np.maximum(a, b))
                    - np.log(np.abs(slope))
                )
                * sigma
                * sqrt(2.0 * pi)
            )
        narrow = np.abs(y1 - y0) / sigma < 1e-6
        value = np.where(narrow, gaussian_mid * dx, value)
        return value / (constraint.upper - constraint.lower)
    mean, sd = constraint.center, constraint.sigma
    width = np.hypot(sigma, slope * sd)
    conditional_sd = sd * sigma / width
    conditional_mean = mean * (sigma / width) ** 2 + slope * (sd / width) ** 2 * (
        measured - intercept
    )
    mass = _log_normal_interval(
        (x0 - conditional_mean) / conditional_sd,
        (x1 - conditional_mean) / conditional_sd,
    )
    normalization = (
        _log_normal_interval(np.asarray(-mean / sd), np.asarray((1.0 - mean) / sd))
        if fraction
        else 0.0
    )
    return np.where(
        dx > 0,
        np.exp(
            -0.5 * ((measured - intercept - slope * mean) / width) ** 2
            + mass
            - normalization
        )
        * sigma
        / width,
        0.0,
    )


def _physical_fraction_limit(air17, air18, sulfate18, request):
    r18 = R18_VSMOW * (1.0 + air18 / 1000.0) * request.alpha18_air_to_sulfate
    r17 = R17_VSMOW * np.exp(air17 / 1000.0) * (1.0 + air18 / 1000.0) ** 0.528
    r17 *= air_fractionation_factors(
        request.alpha18_air_to_sulfate,
        request.theta_air_to_sulfate,
        request.alpha17_air_to_sulfate,
    )[0]
    total = 1.0 + r17 + r18
    q = R18_VSMOW * (1.0 + sulfate18 / 1000.0)
    return np.minimum(
        1.0, np.minimum(total / (r17 + 1.0 + q), total / (r17 + (1.0 + q) * r18 / q))
    )


def _outer_fraction_rule(air17, request, order, physical_upper=None):
    constraint = request.incorporation
    low, prior_high = _bounds(constraint, fraction=True)
    high = np.full_like(air17, prior_high)
    if physical_upper is not None:
        high = np.maximum(low, np.minimum(high, physical_upper))
    center = _center(request.background)
    gain = air17 + request.fractionation_metadata()["anomaly_shift_528_permil"] - center
    sigma = np.hypot(
        request.cap_delta17_sigma_permil,
        (1 - _center(constraint)) * _spread(request.background),
    )
    observed = request.measured_cap_delta17_permil - center
    if constraint.kind == "normal":
        width = np.hypot(sigma, gain * constraint.sigma)
        sd = constraint.sigma * sigma / width
        mean = (
            constraint.center * (sigma / width) ** 2
            + gain * (constraint.sigma / width) ** 2 * observed
        )
    else:
        safe_gain = np.where(np.abs(gain) > 1e-12, gain, 1e-12)
        mean, sd = observed / safe_gain, sigma / np.abs(safe_gain)
    focus = np.clip(
        mean[:, None] + sd[:, None] * np.array([-8.0, -4.0, 0.0, 4.0, 8.0]),
        low,
        high[:, None],
    )
    if request.background.kind == "range":
        # A bounded non-air distribution has two rounded support edges, not
        # one Gaussian peak. Resolve each edge on the analytical-error scale.
        for background_edge in (request.background.lower, request.background.upper):
            edge_gain = (
                air17
                + request.fractionation_metadata()["anomaly_shift_528_permil"]
                - background_edge
            )
            edge_gain = np.where(np.abs(edge_gain) > 1e-12, edge_gain, 1e-12)
            edge_center = (
                request.measured_cap_delta17_permil - background_edge
            ) / edge_gain
            edge_sd = request.cap_delta17_sigma_permil / np.abs(edge_gain)
            focus = np.column_stack(
                (
                    focus,
                    np.clip(
                        edge_center[:, None]
                        + edge_sd[:, None] * np.array([-8.0, 0.0, 8.0]),
                        low,
                        high[:, None],
                    ),
                )
            )
    # The inferred non-air isotope ratios become strongly nonlinear as its
    # inventory vanishes. Resolve this physical endpoint without renormalizing
    # away any incompatible part of the entered fraction distribution.
    endpoint = high[:, None] - (high - low)[:, None] * np.array(
        [0.1, 0.01, 0.001, 0.0001]
    )
    endpoint = np.where(high[:, None] > 0.9, endpoint, high[:, None])
    focus = np.column_stack((focus, endpoint))
    edges = np.sort(
        np.column_stack((np.full_like(air17, low), focus, high)),
        axis=1,
    )
    x, w = _rule(order)
    widths = np.diff(edges, axis=1)
    nodes = edges[:, :-1, None] + widths[..., None] * x
    weights = widths[..., None] * w
    if constraint.kind == "range":
        weights = weights / (prior_high - low)
    else:
        normalization = _log_normal_interval(
            np.asarray(-constraint.center / constraint.sigma),
            np.asarray((1 - constraint.center) / constraint.sigma),
        )
        weights = weights * np.exp(
            -0.5 * ((nodes - constraint.center) / constraint.sigma) ** 2 - normalization
        )
        weights /= constraint.sigma * sqrt(2 * pi)
    nodes = np.where(weights > 0, nodes, 0.5)
    return nodes.reshape(len(air17), -1), weights.reshape(len(air17), -1)


def _evaluate_level(
    air17,
    air18,
    request,
    level,
    inner_fraction,
    *,
    isotope_order=None,
    focused_fraction=False,
    inner_segments=None,
    split_delta18_boundary=False,
):
    order, segments = 2**level + 1, 2 ** (level + 1)
    if inner_segments is not None:
        segments = inner_segments
    inner = request.incorporation if inner_fraction else request.background
    outer = request.background if inner_fraction else request.incorporation
    if (
        request.delta18_sigma_permil
        and split_delta18_boundary
        and _bounds(request.incorporation, fraction=True)[1] > 0.9
    ):
        # At matching sulfate and transferred-air delta18, the limiting
        # non-air inventory switches between its 16O and 18O constraints.
        # Split the Gaussian rule at that boundary instead of asking a global
        # Hermite rule to resolve a moving endpoint feature.
        boundary = (
            1000 * ((1 + air18 / 1000) * request.alpha18_air_to_sulfate - 1)
            - request.measured_delta18_permil
        ) / request.delta18_sigma_permil
        edges = np.sort(
            np.column_stack(
                (
                    np.broadcast_to([-8.0, -4.0, 0.0, 4.0, 8.0], (air18.size, 5)),
                    np.clip(boundary, -8.0, 8.0),
                )
            ),
            axis=1,
        )
        q, w = _rule(order if isotope_order is None else isotope_order)
        widths = np.diff(edges, axis=1)
        iso_nodes = (edges[:, :-1, None] + widths[..., None] * q).reshape(
            air18.size, -1
        )
        iso_weights = (widths[..., None] * w).reshape(air18.size, -1)
        iso_weights *= np.exp(-0.5 * iso_nodes**2) / sqrt(2 * pi)
    elif request.delta18_sigma_permil:
        iso_nodes, iso_weights = _rule(
            order if isotope_order is None else isotope_order, True
        )
    else:
        iso_nodes, iso_weights = np.zeros(1), np.ones(1)
    sigma = request.cap_delta17_sigma_permil * sqrt(
        1.0 - request.isotope_error_correlation**2
    )
    compatibility = np.zeros_like(air17)
    response_error = np.zeros_like(air17)
    root_count = 0
    allowed = (
        request.measured_delta18_permil + request.delta18_sigma_permil * iso_nodes
        > -1000.0
    )
    if iso_nodes.ndim == 1:
        iso_nodes, iso_weights = iso_nodes[allowed], iso_weights[allowed]
        iso_nodes = np.broadcast_to(iso_nodes, (air18.size, iso_nodes.size))
        iso_weights = np.broadcast_to(iso_weights, iso_nodes.shape)
    else:
        iso_weights = np.where(allowed, iso_weights, 0.0)
    if focused_fraction and not inner_fraction and outer.kind != "fixed":
        nodes, weights = [], []
        for z, weight in zip(iso_nodes.T, iso_weights.T):
            upper = np.nextafter(
                _physical_fraction_limit(
                    air17,
                    air18,
                    request.measured_delta18_permil + request.delta18_sigma_permil * z,
                    request,
                ),
                0.0,
            )
            x, w = _outer_fraction_rule(air17, request, order, upper)
            nodes.append(x)
            weights.append(w * weight[:, None])
        outer_count = nodes[0].shape[1]
        outer_nodes, outer_weights = np.concatenate(nodes, axis=1), np.concatenate(
            weights, axis=1
        )
    else:
        x, w = _nodes(outer, order, fraction=not inner_fraction)
        outer_count = x.size
        outer_nodes = np.broadcast_to(
            np.tile(x, iso_nodes.shape[1]),
            (air17.size, iso_nodes.shape[1] * outer_count),
        )
        outer_weights = (iso_weights[..., None] * w).reshape(air17.size, -1)
    process_count = outer_nodes.shape[1]
    knot_count = 1 if inner.kind == "fixed" else 2 * segments + 1
    batch_size = max(1, min(128, 65536 // max(1, air17.size * knot_count)))
    for start in range(0, process_count, batch_size):
        selected = np.arange(start, min(start + batch_size, process_count))
        all_weights = outer_weights[:, selected]
        rows, columns = np.nonzero(all_weights > 0)
        if not rows.size:
            continue
        z = iso_nodes[rows, selected[columns] // outer_count]
        outer_values = outer_nodes[rows, selected[columns]]
        w = all_weights[rows, columns]
        local17, local18 = air17[rows], air18[rows]
        shape = (len(rows),)
        sulfate18 = request.measured_delta18_permil + request.delta18_sigma_permil * z
        measured = (
            request.measured_cap_delta17_permil
            + request.isotope_error_correlation * request.cap_delta17_sigma_permil * z
        )[:, None]
        if inner.kind == "fixed":
            knots = np.full((*shape, 1), inner.center)
            supported = np.ones(shape, dtype=bool)
        else:
            low, high = _bounds(inner, fraction=inner_fraction)
            low = np.full(shape, low)
            high = np.full(shape, high)
            if inner_fraction:
                high = np.minimum(
                    high,
                    np.nextafter(
                        _physical_fraction_limit(local17, local18, sulfate18, request),
                        0.0,
                    ),
                )
            supported = high > low
            # Unsupported cells carry no prior mass; dummy coordinates
            # keep array arithmetic defined and their contributions zero.
            high = np.where(supported, high, low + 1e-8)
            coarse = low[..., None] + (high - low)[..., None] * np.linspace(
                0.0, 1.0, segments + 1
            )
            if inner_fraction:
                gain = (
                    local17
                    + request.fractionation_metadata()["anomaly_shift_528_permil"]
                    - outer_values
                )
                observed = measured[..., 0] - outer_values
                if inner.kind == "normal":
                    width = np.hypot(sigma, gain * inner.sigma)
                    focus_sd = inner.sigma * sigma / width
                    focus_mean = (
                        inner.center * (sigma / width) ** 2
                        + gain * (inner.sigma / width) ** 2 * observed
                    )
                else:
                    safe_gain = np.where(np.abs(gain) > 1e-12, gain, 1e-12)
                    focus_mean, focus_sd = observed / safe_gain, sigma / np.abs(
                        safe_gain
                    )
                focused = focus_sd < (high - low) / 10
                if np.any(focused):
                    # These extra knots locate a potentially narrow response;
                    # they do not supply probabilities. Retain the full prior
                    # interval and evaluate every knot by exact atom balance.
                    focus = np.clip(
                        focus_mean[..., None]
                        + focus_sd[..., None] * np.linspace(-8.0, 8.0, segments + 1),
                        low[..., None],
                        high[..., None],
                    )
                    focus = np.where(focused[..., None], focus, coarse)
                    coarse = np.sort(np.concatenate((coarse, focus), axis=-1), axis=-1)
            knots = np.empty((*shape, 2 * coarse.shape[-1] - 1))
            knots[..., ::2] = coarse
            knots[..., 1::2] = (coarse[..., :-1] + coarse[..., 1:]) / 2
        outer_grid = np.broadcast_to(outer_values[..., None], knots.shape)
        fractions = knots if inner_fraction else outer_grid
        backgrounds = outer_grid if inner_fraction else knots
        fractions = np.clip(fractions, 0.0, 1.0)
        budget = _COMPUTE_BUDGET.get()
        if budget is not None:
            budget.claim(fractions.size)
        values, physical = exact_sulfate_grid(
            air_cap_delta17_permil=local17[:, None],
            air_delta18_permil=local18[:, None],
            sulfate_delta18_permil=sulfate18[:, None],
            fraction=fractions,
            background_cap_delta17_permil=backgrounds,
            alpha18_air_to_sulfate=request.alpha18_air_to_sulfate,
            theta_air_to_sulfate=request.theta_air_to_sulfate,
            alpha17_air_to_sulfate=request.alpha17_air_to_sulfate,
        )
        root_count += values.size
        if inner.kind == "fixed":
            integral = np.where(
                physical[..., 0],
                np.exp(-0.5 * ((measured[..., 0] - values[..., 0]) / sigma) ** 2),
                0.0,
            )
        else:
            valid = (
                physical[..., ::2][..., :-1]
                & physical[..., ::2][..., 1:]
                & physical[..., 1::2]
            )
            y0, y1 = values[..., ::2][..., :-1], values[..., ::2][..., 1:]
            mid = values[..., 1::2]
            x0, x1 = knots[..., ::2][..., :-1], knots[..., ::2][..., 1:]
            coarse = _cell_integral(
                x0, x1, y0, y1, measured, sigma, inner, inner_fraction
            )
            xm = knots[..., 1::2]
            fine = _cell_integral(
                x0, xm, y0, mid, measured, sigma, inner, inner_fraction
            ) + _cell_integral(xm, x1, mid, y1, measured, sigma, inner, inner_fraction)
            # Compare nested integrals before summing, so errors cannot
            # cancel across cells or process nodes. The prior weights are
            # part of this error estimate, just as they are of the integral.
            # A large response error in a negligible prior tail must not
            # veto an accurately resolved measurement-space probability.
            response_error += np.bincount(
                rows,
                weights=w
                * np.where(
                    supported,
                    np.sum(np.where(valid, np.abs(fine - coarse), 0.0), axis=-1),
                    0.0,
                ),
                minlength=air17.size,
            )
            integral = np.sum(np.where(valid, fine, 0.0), axis=-1)
        compatibility += np.bincount(
            rows, weights=w * np.where(supported, integral, 0.0), minlength=air17.size
        )
    return compatibility, response_error, root_count


def exact_sulfate_likelihood(
    air_cap_delta17_permil,
    air_delta18_permil,
    request: SulfateLikelihoodInput,
    *,
    settings: SulfateIntegrationSettings = SulfateIntegrationSettings(),
) -> SulfateLikelihoodResult:
    """Evaluate a conditional likelihood per per-mil sulfate Delta-prime-17O.

    The delta18 measurement is integrated with a locally flat latent sulfate
    delta18 prior. Isotope-error covariance is included explicitly. Background,
    incorporation and measurement constraints are otherwise independent.
    Formation fractionation is conditional fixed, not assigned an uncertainty
    or silently calibrated. Unconverged calculations raise an error.
    """
    a17, a18 = np.broadcast_arrays(
        np.asarray(air_cap_delta17_permil, dtype=float),
        np.asarray(air_delta18_permil, dtype=float),
    )
    if (
        not np.all(np.isfinite(a17))
        or not np.all(np.isfinite(a18))
        or np.any(a18 <= -1000.0)
    ):
        raise ValueError("atmospheric isotope predictions must be finite and physical")
    flat17, flat18 = a17.ravel(), a18.ravel()
    answer = np.empty_like(flat17)
    max_level, total_roots, max_change, max_response = 0, 0, 0.0, 0.0
    alternate_order_states = 0
    delta18_check_states, full_delta18_retry_states = 0, 0
    can_reverse = (
        request.incorporation.kind != "fixed" and request.background.kind != "fixed"
    )
    # Integrating the broader response analytically prevents a narrow peak
    # from hiding between nodes of the other constraint's quadrature.
    f_strength = np.abs(flat17 - _center(request.background)) * _spread(
        request.incorporation
    )
    b_strength = (1.0 - _center(request.incorporation)) * _spread(request.background)
    choose_f = f_strength >= b_strength
    # Integrating a substantial Gaussian non-air contribution analytically
    # leaves the bounded incorporation integral to the outer quadrature.
    # Its convolution is smoother than sampling that Gaussian with Hermite
    # nodes after integrating a bounded fraction interval.
    if request.background.kind == "normal" and request.incorporation.kind == "range":
        choose_f &= (b_strength < 0.5 * request.cap_delta17_sigma_permil) | (
            f_strength > 2.5 * b_strength
        )
    if request.incorporation.kind == "fixed":
        choose_f[:] = False
    if settings.focused_fraction_quadrature:
        choose_f[:] = False
    if request.background.kind == "fixed":
        choose_f[:] = True
    for inner_fraction in (True, False):
        indices = np.flatnonzero(choose_f == inner_fraction)
        for start in range(0, len(indices), settings.chunk_size):
            index = indices[start : start + settings.chunk_size]
            orders = (
                (inner_fraction, not inner_fraction)
                if can_reverse
                else (inner_fraction,)
            )
            split_options = (
                (False, True)
                if (
                    settings.focused_fraction_quadrature
                    and request.delta18_sigma_permil
                    and _bounds(request.incorporation, fraction=True)[1] > 0.9
                )
                else (False,)
            )
            attempts = [
                (item, settings.adaptive_delta18, split)
                for item in orders
                for split in split_options
            ]
            if settings.adaptive_delta18:
                attempts.extend(
                    (item, False, split) for item in orders for split in split_options
                )
            # The spread heuristic can leave a poorly resolved Gaussian in the
            # outer quadrature. Reverse the same independent integrals only for
            # unfinished states; require two fresh convergence checks as before.
            for attempt, (integrate_fraction, fast_delta18, split_delta18) in enumerate(
                attempts
            ):
                if integrate_fraction != inner_fraction:
                    alternate_order_states += index.size
                if settings.adaptive_delta18 and not fast_delta18:
                    full_delta18_retry_states += index.size
                previous, stable = None, np.zeros(index.size, dtype=int)
                isotope_cap = 5
                inner_cap = 1
                for level in range(1, settings.max_level + 1):
                    options = (
                        {"isotope_order": min(2**level + 1, isotope_cap)}
                        if fast_delta18
                        else {}
                    )
                    options["split_delta18_boundary"] = split_delta18
                    if settings.focused_fraction_quadrature:
                        options["focused_fraction"] = True
                        if not integrate_fraction:
                            options["inner_segments"] = min(2 ** (level + 1), inner_cap)
                    value, response, roots = _evaluate_level(
                        flat17[index],
                        flat18[index],
                        request,
                        level,
                        integrate_fraction,
                        **options,
                    )
                    total_roots += roots
                    if previous is not None:
                        change = np.abs(value - previous)
                        tolerance = (
                            settings.absolute_tolerance
                            + settings.relative_tolerance * np.abs(value)
                        )
                        if np.any(response > tolerance):
                            inner_cap *= 2
                        good = (
                            (change <= tolerance)
                            & (response <= tolerance)
                            & (
                                response
                                <= np.exp(-0.5) * settings.response_tolerance_sigma
                            )
                        )
                        stable = np.where(good, stable + 1, 0)
                        done = stable >= 2
                        if (
                            fast_delta18
                            and request.delta18_sigma_permil
                            and np.any(done)
                        ):
                            delta18_check_states += int(np.sum(done))
                            checked, checked_response, checked_roots = _evaluate_level(
                                flat17[index[done]],
                                flat18[index[done]],
                                request,
                                level,
                                integrate_fraction,
                                isotope_order=2 * options["isotope_order"] - 1,
                                focused_fraction=settings.focused_fraction_quadrature,
                                inner_segments=options.get("inner_segments"),
                                split_delta18_boundary=split_delta18,
                            )
                            total_roots += checked_roots
                            iso_change = np.abs(checked - value[done])
                            iso_good = (
                                (
                                    iso_change
                                    <= settings.absolute_tolerance
                                    + settings.relative_tolerance * np.abs(checked)
                                )
                                & (
                                    checked_response
                                    <= settings.absolute_tolerance
                                    + settings.relative_tolerance * np.abs(checked)
                                )
                                & (
                                    checked_response
                                    <= np.exp(-0.5) * settings.response_tolerance_sigma
                                )
                            )
                            done_indices = np.flatnonzero(done)
                            stable[done_indices[~iso_good]] = 0
                            check_order = 2 * options["isotope_order"] - 1
                            for _ in range(3):
                                pending = np.flatnonzero(~iso_good)
                                if not pending.size or check_order >= 129:
                                    break
                                check_order = 2 * check_order - 1
                                refined, refined_response, roots = _evaluate_level(
                                    flat17[index[done_indices[pending]]],
                                    flat18[index[done_indices[pending]]],
                                    request,
                                    level,
                                    integrate_fraction,
                                    isotope_order=check_order,
                                    focused_fraction=settings.focused_fraction_quadrature,
                                    inner_segments=options.get("inner_segments"),
                                    split_delta18_boundary=split_delta18,
                                )
                                total_roots += roots
                                delta18_check_states += pending.size
                                delta = np.abs(refined - checked[pending])
                                tol = (
                                    settings.absolute_tolerance
                                    + settings.relative_tolerance * np.abs(refined)
                                )
                                resolved = (
                                    (delta <= tol)
                                    & (refined_response <= tol)
                                    & (
                                        refined_response
                                        <= np.exp(-0.5)
                                        * settings.response_tolerance_sigma
                                    )
                                )
                                checked[pending], checked_response[pending] = (
                                    refined,
                                    refined_response,
                                )
                                iso_change[pending] = delta
                                # At a promoted isotope order, independently
                                # recheck both process-grid comparisons. More
                                # isotope nodes must not bypass those checks.
                                if np.any(resolved):
                                    selected = pending[resolved]
                                    lower = []
                                    for previous_level in (level - 1, level - 2):
                                        v, e, roots = _evaluate_level(
                                            flat17[index[done_indices[selected]]],
                                            flat18[index[done_indices[selected]]],
                                            request,
                                            previous_level,
                                            integrate_fraction,
                                            isotope_order=check_order,
                                            focused_fraction=settings.focused_fraction_quadrature,
                                            inner_segments=options.get(
                                                "inner_segments"
                                            ),
                                            split_delta18_boundary=split_delta18,
                                        )
                                        total_roots += roots
                                        lower.append((v, e))
                                    local_tol = tol[resolved]
                                    first = (
                                        np.abs(checked[selected] - lower[0][0])
                                        <= local_tol
                                    ) & (lower[0][1] <= local_tol)
                                    second = (
                                        np.abs(lower[0][0] - lower[1][0]) <= local_tol
                                    ) & (lower[1][1] <= local_tol)
                                    stable[done_indices[selected]] = np.where(
                                        first, 1 + second.astype(int), 0
                                    )
                                    iso_good[selected] = first & second
                                    # Further isotope refinement cannot repair
                                    # a process grid. Resume that grid instead.
                                    if np.any(~(first & second)):
                                        isotope_cap = max(isotope_cap, check_order)
                                        break
                                isotope_cap = max(isotope_cap, check_order)
                            value[done_indices] = checked
                            response[done_indices] = checked_response
                            value[done_indices[iso_good]] = checked[iso_good]
                            change[done_indices[iso_good]] = np.maximum(
                                change[done_indices[iso_good]], iso_change[iso_good]
                            )
                            response[done_indices[iso_good]] = checked_response[
                                iso_good
                            ]
                            done[done_indices[~iso_good]] = False
                            if np.any(~iso_good):
                                # A failed isotope-order check calls for more
                                # isotope nodes, not repeated refinement of
                                # only the independent process integrals.
                                isotope_cap = min(max(isotope_cap, check_order), 129)
                        if np.any(done):
                            answer[index[done]] = value[done]
                            max_level = max(max_level, level)
                            max_change = max(max_change, float(np.max(change[done])))
                            max_response = max(
                                max_response, float(np.max(response[done]))
                            )
                            index, stable = index[~done], stable[~done]
                            value = value[~done]
                            if not index.size:
                                break
                    previous = value
                if not index.size:
                    break
            if index.size:
                raise SulfateIntegrationError(
                    "Sulfate uncertainty integration did not converge; no posterior was returned.",
                    diagnostics={
                        "unfinished_states": index.size,
                        "air_cap_delta17_permil": flat17[index[:8]].tolist(),
                        "air_delta18_permil": flat18[index[:8]].tolist(),
                        "last_compatibility": value[:8].tolist(),
                        "stable_checks": stable[:8].tolist(),
                        "last_response_error": response[:8].tolist(),
                    },
                )
    sigma = request.cap_delta17_sigma_permil * sqrt(
        1.0 - request.isotope_error_correlation**2
    )
    with np.errstate(divide="ignore"):
        logs = np.log(answer) - log(sigma * sqrt(2.0 * pi))
    return SulfateLikelihoodResult(
        logs.reshape(a17.shape),
        {
            "status": "converged",
            "observation": asdict(request),
            "settings": asdict(settings),
            "fractionation": request.fractionation_metadata(),
            "maximum_level": max_level,
            "exact_forward_root_evaluations": total_roots,
            "alternate_integration_order_states": alternate_order_states,
            "delta18_quadrature_check_states": delta18_check_states,
            "full_delta18_quadrature_retry_states": full_delta18_retry_states,
            "maximum_final_compatibility_change": max_change,
            "maximum_inner_compatibility_error_estimate": max_response,
            "inner_error_check": "Sum of prior-weighted absolute fine-minus-coarse cell integrals; checked against the likelihood tolerance",
            "likelihood_coordinate": "measured sulfate log-0.528 anomaly, per mil; conditional on delta18 measurement",
            "delta18_treatment": "locally flat latent sulfate delta18 prior; Gaussian analytical covariance integrated",
            "normal_inner_tail_omission": "at most 1.3e-15 prior mass outside +/-8 sigma before physical fraction bounds",
            "physical_support": "incompatible process combinations contribute zero without reweighting remaining prior mass",
            "air_prior": None,
        },
    )
