"""Reuse fixed tensor factors during repeated one-coordinate integration.

The coefficients belong to the existing output surface. Contracting its two
fixed dimensions is an algebraic rearrangement, not a new interpolation fit.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


COORDINATES = ("pO2", "pCO2", "GPP")


def _spline_weights(knots, degree, queries):
    basis = BSpline.design_matrix(queries, knots, degree, extrapolate=False)
    return (basis.indices.reshape(-1, degree+1),
            basis.data.reshape(-1, degree+1))


def _contract(values, dimension, selections):
    fixed = [i for i in range(3) if i != dimension]
    tensor = values.transpose(*fixed, dimension)
    (first, wa), (second, wb) = selections
    answer = np.zeros((len(first), tensor.shape[-1]))
    for a in range(first.shape[1]):
        for b in range(second.shape[1]):
            answer += (wa[:, a]*wb[:, b])[:, None] * tensor[first[:, a], second[:, b], :]
    return answer


class CoordinateSurfaceSlice:
    def __init__(self, surface, base, coordinate, include_delta18):
        self.dimension = COORDINATES.index(coordinate)
        fixed = [i for i in range(3) if i != self.dimension]
        unique = [np.unique(base[COORDINATES[i]], return_inverse=True) for i in fixed]
        codes = unique[0][1]*len(unique[1][0])+unique[1][1]
        pairs, self.pair_ids = np.unique(codes, return_inverse=True)
        physical = [unique[0][0][pairs//len(unique[1][0])],
                    unique[1][0][pairs % len(unique[1][0])]]
        domain = surface.domain
        names = ("po2_pal", "pco2_ppm", "gpp_pgC_per_year")
        for i, values in zip(fixed, physical, strict=True):
            lo, hi = domain[names[i]]
            if not np.all(np.isfinite(values)) or np.any(values < lo) or np.any(values > hi):
                raise ValueError(f"{COORDINATES[i]} grid is outside output-surface domain [{lo:g}, {hi:g}]")
        self.bounds = domain[names[self.dimension]]
        # RGI constructs the NdBSpline once; reuse that exact coefficient array.
        spline = surface._central_d17_interpolator._spline
        selections = [_spline_weights(spline.t[i], spline.k[i], np.log(v) if i == 2 else v)
                      for i, v in zip(fixed, physical, strict=True)]
        self.d17_coefficients = _contract(spline.c, self.dimension, selections)
        self.knots, self.degree = spline.t[self.dimension], spline.k[self.dimension]
        self.d18_coefficients = None
        if include_delta18:
            quadratic = surface._delta18_interpolator
            self.quadratic_weights = quadratic._vectorized_indices_and_weights
            selections = [self.quadratic_weights(quadratic.axes[i], np.log(v) if i in (1, 2) else v)
                          for i, v in zip(fixed, physical, strict=True)]
            self.d18_coefficients = _contract(quadratic.values, self.dimension, selections)
            self.d18_axis = quadratic.axes[self.dimension]

    def evaluate(self, ids, values):
        lo, hi = self.bounds
        if not np.all(np.isfinite(values)) or np.any(values < lo) or np.any(values > hi):
            raise ValueError("integration coordinate is outside output-surface domain")
        rows = self.pair_ids[ids]
        query = np.log(values) if self.dimension == 2 else values
        indices, weights = _spline_weights(self.knots, self.degree, query)
        d17 = np.sum(self.d17_coefficients[rows[:, None], indices]*weights, axis=1)
        d18 = None
        if self.d18_coefficients is not None:
            query = np.log(values) if self.dimension in (1, 2) else values
            indices, weights = self.quadratic_weights(self.d18_axis, query)
            d18 = np.sum(self.d18_coefficients[rows[:, None], indices]*weights, axis=1)
        return d17, d18


def prepare_coordinate_slice(surface, base, coordinate, include_delta18):
    """Use the original evaluator for diagnostic or unsupported surfaces."""
    spline = getattr(getattr(surface, "_central_d17_interpolator", None), "_spline", None)
    if spline is None or not all(hasattr(spline, key) for key in ("t", "k", "c")):
        return None
    if include_delta18 and not surface._delta18_uses_log_pco2:
        return None
    return CoordinateSurfaceSlice(surface, base, coordinate, include_delta18)
