"""Steady-state isotope predictions with independent input constraints.

Normal constraints use the same domain-truncated, +/-4 sigma support as the
inverse solver. Ranges are uniform in physical coordinates. Numerical checks
compare independent scrambled Sobol replicates and successive refinements.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from updated_constrained_pco2_posterior import CoordinateConstraint, _resolve_constraint
from updated_output_surface import load_updated_output_surface


class ForwardResolutionError(RuntimeError):
    """Input propagation has not met its numerical accuracy checks."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ForwardIsotopeConstraints:
    pco2_constraint: CoordinateConstraint = CoordinateConstraint("fixed", center=294.0)
    gpp_constraint: CoordinateConstraint = CoordinateConstraint("fixed", center=290.0)
    po2_constraint: CoordinateConstraint = CoordinateConstraint("fixed", center=1.0)


_SEEDS = (52817, 18002)
_MIN_POWER = 12
_MAX_POWER = 22
_ABS_TOLERANCE = 0.0001  # per mil; one tenth of the displayed isotope precision
_REL_TOLERANCE = 0.0002  # fraction of the propagated central 95% interval width


def _transform(unit, resolved):
    center, (low, high), kind, mean, sigma = resolved
    if kind == "fixed":
        return np.full_like(unit, center)
    if kind == "uniform":
        return low + (high - low) * unit
    lower_cdf, upper_cdf = ndtr((np.array([low, high]) - mean) / sigma)
    return np.clip(mean + sigma * ndtri(lower_cdf + unit * (upper_cdf - lower_cdf)), low, high)


def _evaluate(surface, coordinates):
    args = dict(zip(("p_co2_ppm", "gpp_pgC_per_year", "p_o2_pal"), coordinates))
    d17 = surface.evaluate_central_cap_delta17_grid(**args)
    d18_prime = surface.evaluate_central_delta18_prime_grid(**args)
    result = np.column_stack((d17, 1000.0 * np.expm1(d18_prime / 1000.0)))
    if not np.all(np.isfinite(result)):
        raise ForwardResolutionError("Non-finite forward isotope prediction")
    return result


def _ordered_quantiles(values):
    positions = np.array([0.025, 0.5, 0.975]) * (len(values) - 1)
    lower = positions.astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    return values[lower] + (positions - lower) * (values[upper] - values[lower])


def _merged_order_statistic(left, right, index):
    """Select from two sorted equal-weight samples without concatenating them."""
    count = index + 1
    lo, hi = max(0, count - len(right)), min(count, len(left))
    while lo <= hi:
        i = (lo + hi) // 2
        j = count - i
        a = left[i-1] if i else -np.inf
        b = right[j-1] if j else -np.inf
        anext = left[i] if i < len(left) else np.inf
        bnext = right[j] if j < len(right) else np.inf
        if a > bnext:
            hi = i - 1
        elif b > anext:
            lo = i + 1
        else:
            return max(a, b)
    raise RuntimeError("Invalid sorted-sample partition")


def _replicate_statistics(replicas):
    stats = []
    for replica in replicas:
        stats.append(np.array([
            [*_ordered_quantiles(column), np.mean(column), np.std(column)]
            for column in replica
        ]).T)
    combined = np.empty((5, 2))
    for isotope in range(2):
        left, right = replicas[0][isotope], replicas[1][isotope]
        for row, q in enumerate((.025, .5, .975)):
            position = q * (len(left) + len(right) - 1)
            lower = int(position)
            low = _merged_order_statistic(left, right, lower)
            high = _merged_order_statistic(left, right, lower + 1)
            combined[row, isotope] = low + (position - lower) * (high - low)
    combined[3] = .5 * (stats[0][3] + stats[1][3])
    combined[4] = np.sqrt(.5 * (stats[0][4]**2 + stats[1][4]**2)
                          + .25 * (stats[0][3] - stats[1][3])**2)
    return stats, combined


def predict_isotopes(request: ForwardIsotopeConstraints, *, surface=None):
    surface = surface if surface is not None else load_updated_output_surface()
    specs = (
        ("pCO2", request.pco2_constraint, "pco2_ppm"),
        ("GPP", request.gpp_constraint, "gpp_pgC_per_year"),
        ("pO2", request.po2_constraint, "po2_pal"),
    )
    resolved = [_resolve_constraint(c, surface.domain[key], name) for name, c, key in specs]
    uncertain = [i for i, r in enumerate(resolved) if r[2] != "fixed"]
    central = _evaluate(surface, [np.array([r[0]]) for r in resolved])[0]
    checks = {"converged": True, "samples": 1, "method": "fixed inputs"}
    if not uncertain:
        combined = np.vstack((central, central, central, central, np.zeros(2)))
    else:
        samplers = [qmc.Sobol(len(uncertain), scramble=True, seed=seed) for seed in _SEEDS]
        # Preallocate to avoid doubling memory at each refinement. Only populated
        # pages are touched; sorted marginals permit exact merged quantiles.
        storage = np.empty((2, 2, 2**_MAX_POWER))
        populated = 0
        previous = None
        passes = 0
        for power in range(_MIN_POWER, _MAX_POWER + 1):
            count = 2**power - populated
            for i, sampler in enumerate(samplers):
                # Bound interpolation working arrays independently of refinement size.
                for offset in range(0, count, 8192):
                    unit = sampler.random(min(8192, count - offset))
                    columns = [np.full(len(unit), r[0]) for r in resolved]
                    for j, coordinate in enumerate(uncertain):
                        columns[coordinate] = _transform(unit[:, j], resolved[coordinate])
                    storage[i, :, populated + offset:populated + offset + len(unit)] = _evaluate(surface, columns).T
                storage[i, :, :2**power].sort(axis=1)
            populated = 2**power
            stats, combined = _replicate_statistics(storage[:, :, :populated])
            tolerance = _ABS_TOLERANCE + _REL_TOLERANCE * (combined[2] - combined[0])
            replicate_error = np.max(np.abs(stats[0] - stats[1]), axis=0)
            refinement_error = (np.max(np.abs(combined - previous), axis=0)
                                if previous is not None else np.full(2, np.inf))
            passes = passes + 1 if np.all(np.maximum(replicate_error, refinement_error) <= tolerance) else 0
            previous = combined
            if passes >= 2:
                checks = {
                    "converged": True, "samples": 2 * populated,
                    "method": "two reproducible scrambled Sobol replicates; two successive refinement checks",
                    "seeds": list(_SEEDS), "samples_per_replicate": 2**power,
                    "tolerance_permil": tolerance.tolist(),
                    "replicate_error_permil": replicate_error.tolist(),
                    "refinement_error_permil": refinement_error.tolist(),
                }
                break
        else:
            raise ForwardResolutionError(
                "Isotope uncertainty propagation did not reach its numerical accuracy target. "
                "No interval was returned.",
                {"tolerance_permil": tolerance.tolist(), "replicate_error_permil": replicate_error.tolist(),
                 "refinement_error_permil": refinement_error.tolist(), "consecutive_passes": passes},
            )
    outputs = {}
    for i, name in enumerate(("cap_delta17_prime_permil", "delta18_conventional_permil")):
        outputs[name] = {
            "at_input_centers": float(central[i]),
            "median": float(combined[1, i]),
            "mean": float(combined[3, i]),
            "standard_deviation": float(combined[4, i]),
            "interval95": combined[[0, 2], i].tolist() if uncertain else None,
        }
    return {
        "inputs": asdict(request), "isotopes": outputs,
        "uncertain_inputs": [specs[i][0] for i in uncertain],
        "effective_bounds": {name: list(r[1]) for (name, _, _), r in zip(specs, resolved)},
        "constraint_convention": "Independent constraints; uniform physical-coordinate ranges; normal constraints truncated at +/-4 sigma and the model domain, then renormalized.",
        "interval_convention": "Equal-tailed 95% input-propagated interval of the steady-state central model.",
        "isotope_convention": "Delta-prime-17O: lambda=0.528; conventional delta18O: VSMOW; per mil.",
        "numerical_checks": checks,
    }
