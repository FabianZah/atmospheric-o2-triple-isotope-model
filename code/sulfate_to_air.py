"""Conditional isotope-atom sulfate transfer and a first-order reference.

The exact forward/inverse functions conserve all three oxygen isotopes while
conditioning on sulfate delta18O and specified formation assumptions. They
infer the effective non-air delta18O instead of assigning a mineral endmember.

The first-order relation is S = f*(A + T) + (1-f)*B + bias. B is the
effective non-air contribution after formation; T is the air-path anomaly
shift. Neither has a universal sulfate calibration. The original coordinate
of this relation is retained when measurements are converted to/from OXYTIB.

Its analytical reference likelihood integrates a specified incorporation
constraint in sulfate-measurement space. It supplies no atmospheric prior and
is not a replacement for uncertainty integration through the exact transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, expm1, isfinite, log, log1p, pi
from typing import Literal

import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import brentq

from isotopes import R17_VSMOW, R18_VSMOW


OXYTIB_SLOPE = 0.528


def _finite(*values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError("isotope values, fractions, and uncertainties must be finite")


def _fraction(value: float, *, inverse: bool = False) -> None:
    _finite(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("atmospheric incorporation must lie between zero and one")
    if inverse and value == 0.0:
        raise ValueError("zero atmospheric incorporation provides no finite air inversion")


def _log_delta18(value: float) -> float:
    _finite(value)
    if value <= -1000.0:
        raise ValueError("delta18O must exceed -1000 per mil")
    return 1000.0 * log1p(value / 1000.0)


@dataclass(frozen=True)
class AnomalyCoordinate:
    form: Literal["logarithmic", "linear"]
    slope: float

    def __post_init__(self) -> None:
        _finite(self.slope)
        if self.form not in ("logarithmic", "linear") or not 0.0 < self.slope < 1.0:
            raise ValueError("specify linear/logarithmic notation and a slope between zero and one")

    def from_oxytib(self, anomaly: float, delta18: float) -> float:
        """OXYTIB log-0.528 to this coordinate, on the same VSMOW calibration."""
        _finite(anomaly)
        prime18 = _log_delta18(delta18)
        if self.form == "logarithmic":
            return anomaly + (OXYTIB_SLOPE - self.slope) * prime18
        return 1000.0 * expm1((anomaly + OXYTIB_SLOPE * prime18) / 1000.0) - self.slope * delta18

    def to_oxytib(self, anomaly: float, delta18: float) -> float:
        _finite(anomaly)
        prime18 = _log_delta18(delta18)
        if self.form == "logarithmic":
            return anomaly + (self.slope - OXYTIB_SLOPE) * prime18
        delta17 = anomaly + self.slope * delta18
        if delta17 <= -1000.0:
            raise ValueError("inferred delta17O must exceed -1000 per mil")
        return 1000.0 * log1p(delta17 / 1000.0) - OXYTIB_SLOPE * prime18


@dataclass(frozen=True)
class ConditionalSulfateTransfer:
    """A stated scenario, with no automatic formation/preservation correction.

    All anomaly-valued coefficients are in ``coordinate``, in per mil.
    ``air_scope`` protects historical photochemical-component calculations from
    being passed off as measurements of full atmospheric O2.
    """

    coordinate: AnomalyCoordinate
    background_permil: float
    air_path_shift_permil: float
    reported_bias_permil: float
    air_scope: Literal["full_air", "historical_photochemical_component"]
    assumption_note: str

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, AnomalyCoordinate):
            raise ValueError("an isotope coordinate is required")
        _finite(self.background_permil, self.air_path_shift_permil, self.reported_bias_permil)
        if self.air_scope not in ("full_air", "historical_photochemical_component"):
            raise ValueError("specify the full-air or historical-component scope")
        if not isinstance(self.assumption_note, str) or not self.assumption_note.strip():
            raise ValueError("document the incorporation, background, formation, and preservation assumptions")

    def sulfate_native(self, air_anomaly: float, fraction: float) -> float:
        _finite(air_anomaly)
        _fraction(fraction)
        return (fraction * (air_anomaly + self.air_path_shift_permil)
                + (1.0 - fraction) * self.background_permil + self.reported_bias_permil)

    def air_native(self, sulfate_anomaly: float, fraction: float) -> float:
        _finite(sulfate_anomaly)
        _fraction(fraction, inverse=True)
        return ((sulfate_anomaly - self.reported_bias_permil
                 - (1.0 - fraction) * self.background_permil) / fraction
                - self.air_path_shift_permil)

    def infer_air(
        self, *, sulfate_cap_delta17_permil: float, sulfate_delta18_permil: float,
        fraction: float, air_delta18_permil: float,
    ) -> dict:
        """Return full-air log-0.528 conditional on an explicit air delta18O.

        Paired sulfate/air delta18O values are needed for historical coordinate
        conversions. They do not calibrate the transfer coefficients.
        """
        if self.air_scope != "full_air":
            raise ValueError("a historical photochemical component is not a full-air OXYTIB target")
        sulfate_native = self.coordinate.from_oxytib(sulfate_cap_delta17_permil, sulfate_delta18_permil)
        air_native = self.air_native(sulfate_native, fraction)
        return {
            "air_cap_delta17_permil": self.coordinate.to_oxytib(air_native, air_delta18_permil),
            "air_delta18_assumption_permil": air_delta18_permil,
            "air_native_anomaly_permil": air_native,
            "sulfate_native_anomaly_permil": sulfate_native,
            "sulfate_cap_delta17_permil": sulfate_cap_delta17_permil,
            "sulfate_delta18_permil": sulfate_delta18_permil,
            "atmospheric_incorporation_fraction": fraction,
            "output_coordinate": {"form": "logarithmic", "slope": OXYTIB_SLOPE, "standard": "VSMOW"},
            "transfer": asdict(self),
            "scope": "Conditional first-order transfer; primary signal preservation assumed",
        }


@dataclass(frozen=True)
class IncorporationConstraint:
    """Independent constraint on final oxygen inheritance, not an air prior.

    ``normal`` is the entered Gaussian restricted and renormalized to [0, 1].
    ``range`` assigns uniform density per unit fraction inside the bounds.
    A range including zero is permitted in the forward likelihood, although
    it cannot give a finite endpoint envelope by division.
    """

    kind: Literal["fixed", "normal", "range"]
    center: float | None = None
    sigma: float | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "fixed":
            if self.center is None or any(x is not None for x in (self.sigma, self.lower, self.upper)):
                raise ValueError("fixed incorporation requires only center")
            _fraction(self.center)
        elif self.kind == "normal":
            if self.center is None or self.sigma is None or self.lower is not None or self.upper is not None:
                raise ValueError("normal incorporation requires center and sigma")
            _fraction(self.center)
            _finite(self.sigma)
            if self.sigma <= 0.0:
                raise ValueError("incorporation sigma must be positive")
        elif self.kind == "range":
            if self.lower is None or self.upper is None or self.center is not None or self.sigma is not None:
                raise ValueError("incorporation range requires only lower and upper")
            _fraction(self.lower)
            _fraction(self.upper)
            if self.lower >= self.upper:
                raise ValueError("incorporation range requires lower < upper")
        else:
            raise ValueError("incorporation constraint must be fixed, normal, or range")


def _log_normal_interval(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Log standard-normal probability between ordered endpoints, including tails."""
    lower, upper = np.broadcast_arrays(lower, upper)
    # Reflect positive-tail intervals to avoid subtracting two CDFs near one.
    reflect = lower > 0.0
    lo = np.where(reflect, -upper, lower)
    hi = np.where(reflect, -lower, upper)
    log_hi, log_lo = log_ndtr(hi), log_ndtr(lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        return log_hi + np.log(-np.expm1(log_lo - log_hi))


def sulfate_log_likelihood(
    air_cap_delta17_permil: np.ndarray | float, *, measured_sulfate_permil: float,
    analytical_sigma_permil: float, incorporation: IncorporationConstraint,
    transfer: ConditionalSulfateTransfer,
) -> np.ndarray:
    """First-order log p(sulfate | air), integrated over f analytically.

    This function requires a log-0.528 transfer and log-0.528 observations.
    B, T and analytical bias are conditional fixed values. Their uncertain
    ranges can be assessed separately without inventing a preservation prior.
    Returned density is per per-mil *sulfate measurement*, not per air anomaly.
    It must be combined with atmospheric/geological priors before normalizing
    a posterior. No inverse-space Jacobian or Gaussian air approximation is used.
    """
    if transfer.coordinate != AnomalyCoordinate("logarithmic", OXYTIB_SLOPE) or transfer.air_scope != "full_air":
        raise ValueError("likelihood requires a full-air log-0.528 transfer with coefficients in that coordinate")
    _finite(measured_sulfate_permil, analytical_sigma_permil)
    if analytical_sigma_permil <= 0.0:
        raise ValueError("likelihood requires a positive analytical sigma; use air_native for exact scalar inversion")
    air = np.asarray(air_cap_delta17_permil, dtype=float)
    if not np.all(np.isfinite(air)):
        raise ValueError("predicted air anomalies must be finite")
    sigma = analytical_sigma_permil
    observed = measured_sulfate_permil - transfer.reported_bias_permil - transfer.background_permil
    gain = air + transfer.air_path_shift_permil - transfer.background_permil

    def gaussian(center: np.ndarray, width: np.ndarray | float) -> np.ndarray:
        return -0.5 * ((observed - center) / width)**2 - np.log(width) - 0.5 * log(2.0 * pi)

    if incorporation.kind == "fixed":
        return gaussian(gain * incorporation.center, sigma)
    if incorporation.kind == "range":
        low, high = incorporation.lower, incorporation.upper
        first = (observed - gain * low) / sigma
        last = (observed - gain * high) / sigma
        # At negligible interval width, the midpoint Gaussian is the stable
        # limiting expression (relative correction O((gain*width/sigma)^2)).
        narrow = np.abs(gain) * (high - low) / sigma < 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            integrated = (_log_normal_interval(np.minimum(first, last), np.maximum(first, last))
                          - np.log(np.abs(gain)) - log(high - low))
        return np.where(narrow, gaussian(gain * (low + high) / 2.0, sigma), integrated)

    # Product of a Gaussian likelihood and Gaussian f constraint is Gaussian
    # in f. Integrating that product on [0,1] gives this closed-form truncation.
    mean, spread = incorporation.center, incorporation.sigma
    width = np.hypot(sigma, gain * spread)
    conditional_spread = spread * sigma / width
    conditional_mean = (mean * (sigma / width)**2
                        + gain * (spread / width)**2 * observed)
    prior_mass = _log_normal_interval(np.asarray(-mean / spread), np.asarray((1.0 - mean) / spread))
    conditional_mass = _log_normal_interval(-conditional_mean / conditional_spread,
                                           (1.0 - conditional_mean) / conditional_spread)
    return gaussian(gain * mean, width) + conditional_mass - prior_mass


PENG_AIR_TREATMENTS = {
    "peng_2026_irreversible": (-21.6, .5145, "Irreversible O2 incorporation"),
    "peng_2026_equilibrium": (-3.5, .5199, "Equilibrated O2 incorporation"),
}


def air_fractionation_factors(alpha18: float, theta: float | None,
                             alpha17: float | None = None) -> tuple[float, float]:
    """Return (alpha17, alpha18); accept either legacy theta or independent ratios."""
    _finite(alpha18)
    if alpha18 <= 0:
        raise ValueError("air-path alpha18 must be positive")
    if alpha17 is None:
        if theta is None or not isfinite(theta) or not 0 < theta < 1:
            raise ValueError("air-path theta must lie between zero and one")
        alpha17 = alpha18**theta
    else:
        _finite(alpha17)
        if theta is not None:
            raise ValueError("specify alpha17 or theta, not both")
        if alpha17 <= 0:
            raise ValueError("air-path alpha17 must be positive")
    return alpha17, alpha18


def named_air_fractionation(treatment: str) -> tuple[float, float]:
    if treatment == "none":
        return 1., 1.
    if treatment not in PENG_AIR_TREATMENTS:
        raise ValueError("unknown sulfate fractionation treatment")
    log18, theta, _ = PENG_AIR_TREATMENTS[treatment]
    return exp(theta*log18/1000.), exp(log18/1000.)


def air_fractionation_metadata(alpha18: float, theta: float | None,
                               alpha17: float | None = None, treatment: str = "specified") -> dict:
    a17, a18 = air_fractionation_factors(alpha18, theta, alpha17)
    if treatment != "specified":
        expected = named_air_fractionation(treatment)
        if not np.allclose((a17, a18), expected, rtol=1e-13, atol=0):
            raise ValueError("fractionation factors conflict with the named treatment")
    log18, log17 = 1000.*log(a18), 1000.*log(a17)
    peng = treatment in PENG_AIR_TREATMENTS
    return {
        "treatment": treatment,
        "label": PENG_AIR_TREATMENTS[treatment][2] if peng else
                 ("No fractionation assumed" if treatment == "none" else "User-specified isotope transfer"),
        "alpha18": a18, "alpha17": a17,
        "log18_shift_permil": log18, "log17_shift_permil": log17,
        "anomaly_shift_528_permil": log17-OXYTIB_SLOPE*log18,
        "temperature_c": 25. if peng else None,
        "source_doi": "10.1016/j.gca.2025.11.036" if peng else None,
        "required_incorporation_fraction": .25 if peng else None,
        "scope": "O2-derived oxygen before sulfate mixing; non-air oxygen remains separately constrained",
        "uncertainty_treatment": ("Conditional fixed transfer; theoretical 2SE is not a geological process prior"
                                  if peng else "Conditional fixed transfer"),
    }


@dataclass(frozen=True)
class SulfateProcessAssumptions:
    """Conditional primary-sulfate process, with explicitly supplied coefficients.

    Background is the effective non-air anomaly after its formation processes,
    in log-0.528 VSMOW notation. Its delta18O is inferred by atom conservation,
    not assigned a universal water/mineral value. alpha18 is transferred-air
    oxygen / atmospheric oxygen. Supply alpha17 independently with theta=None,
    or retain the legacy alpha17 = alpha18**theta convention. Preservation is an
    applicability assumption; formation-stage exchange belongs in f already.
    """

    background_cap_delta17_permil: float
    alpha18_air_to_sulfate: float
    theta_air_to_sulfate: float | None
    assumption_note: str
    alpha17_air_to_sulfate: float | None = None

    def __post_init__(self) -> None:
        _finite(self.background_cap_delta17_permil)
        air_fractionation_factors(self.alpha18_air_to_sulfate, self.theta_air_to_sulfate,
                                 self.alpha17_air_to_sulfate)
        if not isinstance(self.assumption_note, str) or not self.assumption_note.strip():
            raise ValueError("document the background, air-path fractionation, and primary-preservation assumptions")


def _composition(anomaly: float, delta18: float) -> np.ndarray:
    _finite(anomaly)
    prime18 = _log_delta18(delta18)
    r18 = R18_VSMOW * exp(prime18 / 1000.0)
    r17 = R17_VSMOW * exp((anomaly + OXYTIB_SLOPE * prime18) / 1000.0)
    return _atoms(r17, r18)


def _atoms(r17: float, r18: float) -> np.ndarray:
    _finite(r17, r18)
    if r17 <= 0.0 or r18 <= 0.0:
        raise ValueError("isotope ratios must be positive")
    values = np.array([1.0, r17, r18])
    return values / values.sum()


def _isotopes(atoms: np.ndarray) -> dict:
    if not np.all(np.isfinite(atoms)) or np.any(atoms <= 0.0):
        raise ValueError("specified inputs require a nonphysical oxygen endmember")
    prime18 = 1000.0 * log(atoms[2] / atoms[0] / R18_VSMOW)
    prime17 = 1000.0 * log(atoms[1] / atoms[0] / R17_VSMOW)
    return {"delta18_permil": 1000.0 * expm1(prime18 / 1000.0),
            "cap_delta17_permil": prime17 - OXYTIB_SLOPE * prime18}


def _background_residual(atoms: np.ndarray, anomaly: float) -> float:
    # Unnormalized oxygen amounts suffice: both sides are homogeneous of
    # degree one. Tiny negative endpoint roundoff is bounded before powers.
    if np.min(atoms) < -1e-14:
        raise ValueError("specified inputs require negative oxygen abundances")
    o16, o17, o18 = np.maximum(atoms, 0.0)
    coefficient = R17_VSMOW * exp(anomaly / 1000.0) / R18_VSMOW**OXYTIB_SLOPE
    return o17 - coefficient * o16**(1.0-OXYTIB_SLOPE) * o18**OXYTIB_SLOPE


def _positive_root(residual, lower: float, upper: float) -> float:
    if lower >= upper:
        raise ValueError("no physical sulfate transfer satisfies the specified isotopes and incorporation")
    left, right = residual(lower), residual(upper)
    if (left > 0.0 and right > 0.0) or (left < 0.0 and right < 0.0):
        raise ValueError("no physical sulfate transfer satisfies the specified isotopes and incorporation")
    return float(brentq(residual, lower, upper, xtol=1e-18, rtol=1e-14))


def exact_air_from_sulfate(
    *, sulfate_cap_delta17_permil: float, sulfate_delta18_permil: float,
    fraction: float, air_delta18_permil: float, process: SulfateProcessAssumptions,
) -> dict:
    """Infer full-air log-0.528 by conserving 16O, 17O and 18O separately.

    f counts all oxygen atoms inherited from air. Given sulfate's measured
    isotope pair, f, atmospheric delta18O and the explicit process assumptions,
    only one scalar remains unknown. Its residual is monotone. No non-air
    delta18O or extra sulfate-reaction ODE is needed. This conditional closure
    does not identify f, background, fractionation or preservation from sulfate.
    """
    _fraction(fraction, inverse=True)
    sulfate = _composition(sulfate_cap_delta17_permil, sulfate_delta18_permil)
    _log_delta18(air_delta18_permil)
    q18 = R18_VSMOW * (1.0+air_delta18_permil/1000.0) * process.alpha18_air_to_sulfate
    alpha17, _ = air_fractionation_factors(process.alpha18_air_to_sulfate, process.theta_air_to_sulfate,
                                         process.alpha17_air_to_sulfate)

    def transferred_atoms(o17):
        o16 = (1.0-o17) / (1.0+q18)
        return np.array([o16, o17, q18*o16])

    if fraction == 1.0:
        if not np.isclose(sulfate[2]/sulfate[0], q18, rtol=1e-12, atol=0.0):
            raise ValueError("100% incorporation requires matching transferred-air and sulfate delta18O")
        transferred, background = sulfate, None
    else:
        lower = max(0.0, 1.0-(1.0+q18)*sulfate[0]/fraction,
                    1.0-(1.0+q18)*sulfate[2]/(fraction*q18))
        upper = min(1.0, sulfate[1]/fraction)
        root = _positive_root(lambda z: _background_residual(sulfate-fraction*transferred_atoms(z),
                                                           process.background_cap_delta17_permil), lower, upper)
        transferred = transferred_atoms(root)
        background = (sulfate-fraction*transferred)/(1.0-fraction)
        _isotopes(background)
    air = _atoms(transferred[1]/transferred[0]/alpha17,
                 transferred[2]/transferred[0]/process.alpha18_air_to_sulfate)
    recombined = fraction*transferred if background is None else fraction*transferred+(1.0-fraction)*background
    return {
        "air": _isotopes(air), "transferred_air": _isotopes(transferred),
        "implied_nonair_oxygen": None if background is None else _isotopes(background),
        "sulfate": _isotopes(sulfate), "atmospheric_incorporation_fraction": fraction,
        "air_delta18_assumption_permil": air_delta18_permil, "process": asdict(process),
        "maximum_atom_closure_residual": float(np.max(np.abs(sulfate-recombined))),
        "coordinate": {"form": "logarithmic", "slope": OXYTIB_SLOPE, "standard": "VSMOW"},
        "scope": "Exact isotope-atom balance conditional on the specified primary-sulfate process",
    }


def exact_sulfate_from_air(
    *, air_cap_delta17_permil: float, air_delta18_permil: float,
    sulfate_delta18_permil: float, fraction: float, process: SulfateProcessAssumptions,
) -> float:
    """Predict sulfate anomaly conditional on its delta18O, for a forward likelihood.

    Paired atmospheric delta18O can come from the atmospheric model instead of
    assuming modern air for every scenario. Sulfate delta18O is a conditioning
    measurement, not an independently predicted observable of this closure.
    """
    _fraction(fraction)
    _log_delta18(sulfate_delta18_permil)
    air = _composition(air_cap_delta17_permil, air_delta18_permil)
    alpha17, _ = air_fractionation_factors(process.alpha18_air_to_sulfate, process.theta_air_to_sulfate,
                                         process.alpha17_air_to_sulfate)
    transferred = _atoms(air[1]/air[0] * alpha17,
                         air[2]/air[0] * process.alpha18_air_to_sulfate)
    q18 = R18_VSMOW * (1.0+sulfate_delta18_permil/1000.0)
    if fraction == 0.0:
        return process.background_cap_delta17_permil
    if fraction == 1.0:
        if not np.isclose(transferred[2]/transferred[0], q18, rtol=1e-12, atol=0.0):
            raise ValueError("100% incorporation requires matching transferred-air and sulfate delta18O")
        return _isotopes(transferred)["cap_delta17_permil"]

    def sulfate_atoms(o17):
        o16 = (1.0-o17)/(1.0+q18)
        return np.array([o16, o17, q18*o16])

    lower = fraction*transferred[1]
    upper = min(1.0, 1.0-(1.0+q18)*fraction*transferred[0],
                1.0-(1.0+q18)*fraction*transferred[2]/q18)
    root = _positive_root(lambda z: _background_residual(sulfate_atoms(z)-fraction*transferred,
                                                       process.background_cap_delta17_permil), lower, upper)
    sulfate = sulfate_atoms(root)
    _isotopes((sulfate-fraction*transferred)/(1.0-fraction))
    return _isotopes(sulfate)["cap_delta17_permil"]


def exact_sulfate_grid(
    *, air_cap_delta17_permil, air_delta18_permil, sulfate_delta18_permil,
    fraction, background_cap_delta17_permil, alpha18_air_to_sulfate: float,
    theta_air_to_sulfate: float | None,
    alpha17_air_to_sulfate: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched exact forward roots; return anomaly and physical-support mask.

    Uses safeguarded Newton iteration on the same monotone atom-balance
    residual as the independent scalar Brent solver. Invalid process
    combinations are masked, not projected onto the physical boundary.
    """
    alpha17, _ = air_fractionation_factors(alpha18_air_to_sulfate, theta_air_to_sulfate,
                                         alpha17_air_to_sulfate)
    a17, a18, s18, f, background = np.broadcast_arrays(*(
        np.asarray(v, dtype=float) for v in (air_cap_delta17_permil, air_delta18_permil,
                                           sulfate_delta18_permil, fraction, background_cap_delta17_permil)))
    if not all(np.all(np.isfinite(v)) for v in (a17, a18, s18, f, background)):
        raise ValueError("sulfate grid inputs must be finite")
    if np.any(a18 <= -1000.0) or np.any(s18 <= -1000.0) or np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("invalid isotope ratio or incorporation fraction")
    r18 = R18_VSMOW * (1.0+a18/1000.0) * alpha18_air_to_sulfate
    r17 = R17_VSMOW * np.exp(a17/1000.0) * (1.0+a18/1000.0)**OXYTIB_SLOPE
    r17 *= alpha17
    total = 1.0+r17+r18
    air16, air17, air18 = 1.0/total, r17/total, r18/total
    q = R18_VSMOW * (1.0+s18/1000.0)
    c = R17_VSMOW * np.exp(background/1000.0) / R18_VSMOW**OXYTIB_SLOPE
    if not all(np.all(np.isfinite(v)) for v in (r17, r18, c)):
        raise ValueError("isotope grid exceeds floating-point ratio limits")
    low = f*air17
    high = np.minimum(1.0-(1.0+q)*f*air16, 1.0-(1.0+q)*f*air18/q)
    valid = (low < high) & (f < 1.0)
    active = valid & (f > 0.0)
    z = np.clip(low+(1.0-f)*R17_VSMOW/(1.0+R17_VSMOW+R18_VSMOW), low, np.maximum(low, high))
    for _ in range(64):
        if not np.any(active):
            break
        b16 = np.maximum((1.0-z)/(1.0+q)-f*air16, np.finfo(float).tiny)
        b18 = np.maximum(q*(1.0-z)/(1.0+q)-f*air18, np.finfo(float).tiny)
        expected17 = c*b16**(1.0-OXYTIB_SLOPE)*b18**OXYTIB_SLOPE
        residual = z-f*air17-expected17
        # At a vanishing non-air inventory the residual is ill-conditioned,
        # although the sulfate isotope ratio is already resolved. A bracket
        # narrower than eight floating-point spacings bounds the anomaly error
        # to about 2e-12 per mil, independently of that residual's slope.
        converged = (np.abs(residual) <= 2e-18) | (high-low <= 8*np.spacing(np.abs(z)))
        active &= ~converged
        low = np.where(active & (residual < 0.0), z, low)
        high = np.where(active & (residual >= 0.0), z, high)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            derivative = 1.0+expected17/(1.0+q)*((1.0-OXYTIB_SLOPE)/b16+OXYTIB_SLOPE*q/b18)
            proposal = z-residual/derivative
        proposal = np.where((proposal > low) & (proposal < high) & np.isfinite(proposal),
                            proposal, (low+high)/2.0)
        z = np.where(active, proposal, z)
    if np.any(active):
        raise RuntimeError("batched sulfate atom balance did not converge")
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = 1000.0*np.log(z*(1.0+q)/(1.0-z)/R17_VSMOW)-OXYTIB_SLOPE*1000.0*np.log1p(s18/1000.0)
    predicted = np.where(f == 0.0, background, predicted)
    pure = (f == 1.0) & np.isclose(r18, q, rtol=1e-12, atol=0.0)
    pure_anomaly = a17+1000.*(log(alpha17)-OXYTIB_SLOPE*log(alpha18_air_to_sulfate))
    predicted = np.where(pure, pure_anomaly, predicted)
    valid |= pure | (f == 0.0)
    return np.where(valid, predicted, np.nan), valid
