"""Readable, area-balanced contour selection for public isotope fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IsotopeContourSelection:
    levels_permil: tuple[float, ...]
    label_decimals: int
    reference_step_permil: float
    strategy: str = "plot_area_balanced_readable"


def _contour_level_count(minimum: float, maximum: float, step: float) -> int:
    epsilon = max(1.0, abs(maximum - minimum)) * 1.0e-10
    first = math.ceil((minimum + epsilon) / step)
    last = math.floor((maximum - epsilon) / step)
    return max(0, last - first + 1)


def _contour_decimals(step: float) -> int:
    for decimals in range(7):
        scaled = step * 10**decimals
        if abs(scaled - round(scaled)) < 1.0e-8:
            return decimals
    return 6


def _rounded_float(value: float, decimals: int) -> float:
    return float(f"{value:.{decimals}f}")


def _adaptive_reference(minimum: float, maximum: float) -> IsotopeContourSelection:
    span = maximum - minimum
    if not math.isfinite(span) or span <= 0.0:
        return IsotopeContourSelection((), 1, 0.0, "empty")
    target_count = 9
    minimum_count = 8
    maximum_count = 12
    minimum_step = 0.001
    reference_exponent = math.floor(math.log10(span / target_count))
    candidates: list[tuple[float, int, float]] = []
    for exponent in range(reference_exponent - 2, reference_exponent + 4):
        for multiplier in (1.0, 2.0, 2.5, 5.0):
            step = multiplier * 10**exponent
            if step < minimum_step:
                continue
            count = _contour_level_count(minimum, maximum, step)
            if not count:
                continue
            outside_penalty = (
                20 * (minimum_count - count)
                if count < minimum_count
                else 20 * (count - maximum_count)
                if count > maximum_count
                else 0
            )
            candidates.append((outside_penalty + abs(count - target_count), -step, step))
    step = min(candidates)[2] if candidates else minimum_step
    epsilon = max(1.0, abs(span)) * 1.0e-10
    first = math.ceil((minimum + epsilon) / step)
    last = math.floor((maximum - epsilon) / step)
    decimals = _contour_decimals(step)
    levels = tuple(
        _rounded_float(index * step, decimals) for index in range(first, last + 1)
    )
    return IsotopeContourSelection(levels, decimals, step, "regular")


def _node_widths(axis: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    coordinates = np.log10(axis) if logarithmic else axis
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("contour axes must be one-dimensional with at least two points")
    widths = np.empty_like(coordinates, dtype=float)
    widths[0] = 0.5 * (coordinates[1] - coordinates[0])
    widths[-1] = 0.5 * (coordinates[-1] - coordinates[-2])
    widths[1:-1] = 0.5 * (coordinates[2:] - coordinates[:-2])
    if np.any(widths <= 0.0):
        raise ValueError("contour axes must be strictly increasing")
    return widths


def isotope_field_area_weights(
    pco2_ppm: np.ndarray, gpp_pgC_per_year: np.ndarray
) -> np.ndarray:
    """Return node weights in the displayed log-pCO2/linear-GPP geometry."""

    x_width = _node_widths(np.asarray(pco2_ppm, dtype=float), logarithmic=True)
    y_width = _node_widths(np.asarray(gpp_pgC_per_year, dtype=float), logarithmic=False)
    weights = np.outer(x_width, y_width)
    return weights / float(np.sum(weights))


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    flat_weights = np.asarray(weights, dtype=float).reshape(-1)
    finite = np.isfinite(flat_values) & np.isfinite(flat_weights) & (flat_weights > 0.0)
    if not np.any(finite):
        raise ValueError("isotope field contains no finite weighted values")
    order = np.argsort(flat_values[finite])
    selected_values = flat_values[finite][order]
    selected_weights = flat_weights[finite][order]
    cumulative = np.cumsum(selected_weights)
    cumulative = (cumulative - 0.5 * selected_weights) / cumulative[-1]
    return np.interp(probabilities, cumulative, selected_values)


def _readable_value(value: float, decimals: int, reference_step: float) -> float:
    magnitude = abs(value)
    magnitude_increment = (
        1.0
        if magnitude >= 10.0
        else 0.5
        if magnitude >= 5.0
        else 0.5 * 10.0 ** (-decimals)
    )
    increment = min(magnitude_increment, reference_step)
    output_decimals = max(decimals, _contour_decimals(increment))
    return _rounded_float(round(value / increment) * increment, output_decimals)


def select_isotope_field_contours(
    values_permil: np.ndarray,
    pco2_ppm: np.ndarray,
    gpp_pgC_per_year: np.ndarray,
) -> IsotopeContourSelection:
    """Select labeled contours that remain readable across public axis windows."""

    values = np.asarray(values_permil, dtype=float)
    if values.shape != (len(pco2_ppm), len(gpp_pgC_per_year)):
        raise ValueError("isotope field shape does not match its axes")
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    reference = _adaptive_reference(minimum, maximum)
    if not reference.levels_permil:
        return reference

    target_count = min(10, max(8, len(reference.levels_permil)))
    decimals = max(1, reference.label_decimals)
    probabilities = np.arange(1, target_count + 1, dtype=float) / (target_count + 1)
    weights = isotope_field_area_weights(
        np.asarray(pco2_ppm, dtype=float), np.asarray(gpp_pgC_per_year, dtype=float)
    )
    raw_levels = _weighted_quantile(values, weights, probabilities)
    rounded_levels = {
        _readable_value(
            float(level), decimals, reference.reference_step_permil
        )
        for level in raw_levels
        if minimum < level < maximum
    }
    levels = sorted(level for level in rounded_levels if minimum < level < maximum)

    desired_minimum = min(8, target_count)
    supplements = [
        level
        for level in reference.levels_permil
        if minimum < level < maximum and level not in levels
    ]
    while len(levels) < desired_minimum and supplements:
        supplements.sort(
            key=lambda candidate: min(abs(candidate - level) for level in levels),
            reverse=True,
        )
        levels.append(_rounded_float(supplements.pop(0), decimals))
        levels.sort()
    if len(levels) < 4:
        return reference
    label_decimals = decimals + int(any(abs(level) < 5.0 for level in levels))
    return IsotopeContourSelection(
        tuple(levels), label_decimals, reference.reference_step_permil
    )
