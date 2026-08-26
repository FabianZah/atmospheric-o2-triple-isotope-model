"""Fast local R7 tendency operator in physically linear isotope coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from global_o2_isotope_reservoir import GlobalO2Reservoir, IsotopologueTendency


def prime_state_to_ratio_coordinates(state_permil: np.ndarray) -> np.ndarray:
    """Map [delta-prime-18O, Delta-prime-17O] to relative 17O/18O ratios."""

    state = np.asarray(state_permil, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("prime isotope state must contain two finite values")
    delta18_prime, cap_delta17_prime = state
    delta17_prime = cap_delta17_prime + 0.528 * delta18_prime
    return np.exp(np.asarray([delta17_prime, delta18_prime]) / 1000.0)


def prime_tendency_rates_permil_per_year(
    reservoir: GlobalO2Reservoir,
    tendency: IsotopologueTendency,
) -> np.ndarray:
    """Return [delta-prime-18O, Delta-prime-17O] tendency rates."""

    d_major, d17, d18 = tendency.values
    delta18_rate = 1000.0 * (
        d18 / reservoir.o16o18 - d_major / reservoir.o16o16
    )
    delta17_rate = 1000.0 * (
        d17 / reservoir.o16o17 - d_major / reservoir.o16o16
    )
    return np.asarray(
        [delta18_rate, delta17_rate - 0.528 * delta18_rate], dtype=float
    )


def isotopologue_tendency_from_prime_rates(
    reservoir: GlobalO2Reservoir,
    rates_permil_per_year: np.ndarray,
    *,
    source: str,
) -> IsotopologueTendency:
    """Recover a molecule-conserving tendency from two prime-isotope rates."""

    rates = np.asarray(rates_permil_per_year, dtype=float)
    if rates.shape != (2,) or not np.all(np.isfinite(rates)):
        raise ValueError("prime-isotope tendency requires two finite rates")
    if not source.strip():
        raise ValueError("reconstructed isotope tendency requires provenance")
    delta18_rate, cap_delta17_rate = rates
    fractional18 = delta18_rate / 1000.0
    fractional17 = (cap_delta17_rate + 0.528 * delta18_rate) / 1000.0
    total = reservoir.o16o16 + reservoir.o16o17 + reservoir.o16o18
    d_major = -reservoir.o16o16 * (
        reservoir.o16o17 * fractional17
        + reservoir.o16o18 * fractional18
    ) / total
    return IsotopologueTendency(
        d_major,
        reservoir.o16o17 * (fractional17 + d_major / reservoir.o16o16),
        reservoir.o16o18 * (fractional18 + d_major / reservoir.o16o16),
        source=source.strip(),
    )


@dataclass(frozen=True)
class LocalR7ResponseOperator:
    """Affine R7 isotopologue tendency in relative isotope-ratio coordinates."""

    reference_ratio_coordinates: np.ndarray
    reference_tendency_mol_per_year: np.ndarray
    ratio_jacobian_mol_per_year: np.ndarray
    source: str
    sample_condition: float
    maximum_sample_residual_mol_per_year: float

    def __post_init__(self) -> None:
        reference_coordinates = np.asarray(
            self.reference_ratio_coordinates, dtype=float
        )
        reference_tendency = np.asarray(
            self.reference_tendency_mol_per_year, dtype=float
        )
        jacobian = np.asarray(self.ratio_jacobian_mol_per_year, dtype=float)
        if (
            reference_coordinates.shape != (2,)
            or reference_tendency.shape != (3,)
            or jacobian.shape != (3, 2)
        ):
            raise ValueError(
                "R7 response requires 2 ratio coordinates, a 3-vector, and a "
                "3x2 Jacobian"
            )
        if (
            not np.all(np.isfinite(reference_coordinates))
            or not np.all(np.isfinite(reference_tendency))
            or not np.all(np.isfinite(jacobian))
        ):
            raise ValueError("R7 response coefficients must be finite")
        if not self.source.strip():
            raise ValueError("R7 response operator requires provenance")
        if not np.isfinite(self.sample_condition) or self.sample_condition <= 0.0:
            raise ValueError("R7 response sample condition must be finite and positive")
        if (
            not np.isfinite(self.maximum_sample_residual_mol_per_year)
            or self.maximum_sample_residual_mol_per_year < 0.0
        ):
            raise ValueError("R7 response fit residual must be finite and non-negative")
        object.__setattr__(
            self, "reference_ratio_coordinates", reference_coordinates.copy()
        )
        object.__setattr__(
            self, "reference_tendency_mol_per_year", reference_tendency.copy()
        )
        object.__setattr__(self, "ratio_jacobian_mol_per_year", jacobian.copy())

    def evaluate_values(self, state_permil: np.ndarray) -> np.ndarray:
        coordinates = prime_state_to_ratio_coordinates(state_permil)
        return (
            self.reference_tendency_mol_per_year
            + self.ratio_jacobian_mol_per_year
            @ (coordinates - self.reference_ratio_coordinates)
        )

    def evaluate(self, state_permil: np.ndarray) -> IsotopologueTendency:
        return IsotopologueTendency(
            *map(float, self.evaluate_values(state_permil)),
            source=f"local isotope-ratio R7 response: {self.source}",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinate_definition": (
                "x17=exp((DeltaPrime17O+0.528*deltaPrime18O)/1000); "
                "x18=exp(deltaPrime18O/1000)"
            ),
            "reference_ratio_coordinates": self.reference_ratio_coordinates.tolist(),
            "reference_tendency_mol_per_year": (
                self.reference_tendency_mol_per_year.tolist()
            ),
            "ratio_jacobian_mol_per_year": (
                self.ratio_jacobian_mol_per_year.tolist()
            ),
            "source": self.source,
            "sample_condition": self.sample_condition,
            "maximum_sample_residual_mol_per_year": (
                self.maximum_sample_residual_mol_per_year
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LocalR7ResponseOperator":
        return cls(
            reference_ratio_coordinates=np.asarray(
                data["reference_ratio_coordinates"], dtype=float
            ),
            reference_tendency_mol_per_year=np.asarray(
                data["reference_tendency_mol_per_year"], dtype=float
            ),
            ratio_jacobian_mol_per_year=np.asarray(
                data["ratio_jacobian_mol_per_year"], dtype=float
            ),
            source=str(data["source"]),
            sample_condition=float(data["sample_condition"]),
            maximum_sample_residual_mol_per_year=float(
                data["maximum_sample_residual_mol_per_year"]
            ),
        )


def fit_local_r7_response_operator(
    evaluate_tendency: Callable[[np.ndarray], np.ndarray],
    reference_state_permil: np.ndarray,
    *,
    finite_difference_step_permil: float | np.ndarray = 0.5,
    source: str,
) -> LocalR7ResponseOperator:
    """Fit an affine ratio-coordinate operator from five native evaluations."""

    reference = np.asarray(reference_state_permil, dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("R7 response reference state must contain two finite values")
    steps = np.broadcast_to(
        np.asarray(finite_difference_step_permil, dtype=float), reference.shape
    ).copy()
    if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
        raise ValueError("R7 response finite-difference steps must be positive")
    if not source.strip():
        raise ValueError("R7 response fit requires provenance")

    states = [reference.copy()]
    for index in range(2):
        lower = reference.copy()
        upper = reference.copy()
        lower[index] -= steps[index]
        upper[index] += steps[index]
        states.extend((lower, upper))
    coordinates = np.asarray(
        [prime_state_to_ratio_coordinates(state) for state in states]
    )
    reference_coordinates = prime_state_to_ratio_coordinates(reference)
    design = np.column_stack(
        (np.ones(len(states)), coordinates - reference_coordinates)
    )
    tendencies = np.asarray(
        [np.asarray(evaluate_tendency(state), dtype=float) for state in states]
    )
    if tendencies.shape != (5, 3) or not np.all(np.isfinite(tendencies)):
        raise ValueError("native R7 evaluator must return one finite 3-vector per state")
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, tendencies, rcond=None
    )
    if rank != 3:
        raise np.linalg.LinAlgError("R7 response samples do not span ratio coordinates")
    predicted = design @ coefficients
    maximum_residual = float(np.max(np.abs(predicted - tendencies)))
    condition = float(singular_values[0] / singular_values[-1])
    return LocalR7ResponseOperator(
        reference_ratio_coordinates=reference_coordinates,
        reference_tendency_mol_per_year=coefficients[0],
        ratio_jacobian_mol_per_year=coefficients[1:].T,
        source=source.strip(),
        sample_condition=condition,
        maximum_sample_residual_mol_per_year=maximum_residual,
    )


def interpolate_local_r7_response_operators(
    axis_values: np.ndarray,
    operators: tuple[LocalR7ResponseOperator, ...],
    target_value: float,
    *,
    log10_axis: bool,
    source: str,
) -> LocalR7ResponseOperator:
    """Shape-preserving interpolation of mechanistic operator coefficients."""

    axis = np.asarray(axis_values, dtype=float)
    if axis.ndim != 1 or len(axis) < 3 or not np.all(np.isfinite(axis)):
        raise ValueError("operator interpolation needs at least three finite nodes")
    if len(operators) != len(axis):
        raise ValueError("operator count must match interpolation nodes")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("operator interpolation nodes must be strictly increasing")
    if not np.isfinite(target_value):
        raise ValueError("operator interpolation target must be finite")
    if target_value < axis[0] or target_value > axis[-1]:
        raise ValueError("operator interpolation does not permit extrapolation")
    if log10_axis and (np.any(axis <= 0.0) or target_value <= 0.0):
        raise ValueError("log-coordinate operator interpolation requires positive values")
    if not source.strip():
        raise ValueError("interpolated R7 response requires provenance")

    reference_coordinates = np.asarray(
        [operator.reference_ratio_coordinates for operator in operators]
    )
    if float(np.max(np.ptp(reference_coordinates, axis=0))) > 1.0e-12:
        raise ValueError("operators use incompatible isotope reference coordinates")
    transformed_axis = np.log10(axis) if log10_axis else axis
    transformed_target = np.log10(target_value) if log10_axis else target_value
    tendencies = np.asarray(
        [operator.reference_tendency_mol_per_year for operator in operators]
    )
    jacobians = np.asarray(
        [operator.ratio_jacobian_mol_per_year for operator in operators]
    )
    interpolated_tendency = PchipInterpolator(
        transformed_axis, tendencies, axis=0
    )(transformed_target)
    interpolated_jacobian = PchipInterpolator(
        transformed_axis, jacobians, axis=0
    )(transformed_target)
    return LocalR7ResponseOperator(
        reference_ratio_coordinates=reference_coordinates[0],
        reference_tendency_mol_per_year=interpolated_tendency,
        ratio_jacobian_mol_per_year=interpolated_jacobian,
        source=source.strip(),
        sample_condition=max(operator.sample_condition for operator in operators),
        maximum_sample_residual_mol_per_year=max(
            operator.maximum_sample_residual_mol_per_year for operator in operators
        ),
    )


def _interpolate_signed_rate_on_log_axis(
    axis: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float:
    """Interpolate a rate while retaining log accuracy away from zero.

    R7 Delta-prime-17O tendencies normally retain one sign, where
    log-magnitude interpolation best preserves their relative response. At
    extreme atmospheric isotope states, operators at weak-forcing nodes can
    legitimately cross zero. Signed PCHIP interpolation is then required;
    rejecting the whole surface would also reject valid strong-forcing nodes.
    """

    coordinates = np.log10(np.asarray(axis, dtype=float))
    rates = np.asarray(values, dtype=float)
    target_coordinate = np.log10(float(target))
    if np.all(rates > 0.0) or np.all(rates < 0.0):
        sign = float(np.sign(rates[0]))
        return float(
            sign
            * np.exp(
                PchipInterpolator(coordinates, np.log(np.abs(rates)))(
                    target_coordinate
                )
            )
        )
    return float(PchipInterpolator(coordinates, rates)(target_coordinate))


@dataclass(frozen=True)
class LocalR7ResponseSurface:
    """Rectangular pO2-pCO2 surface of native local R7 response operators."""

    po2_nodes_pal: np.ndarray
    pco2_nodes_ppm: np.ndarray
    operators: tuple[tuple[LocalR7ResponseOperator, ...], ...]
    source: str

    def __post_init__(self) -> None:
        po2 = np.asarray(self.po2_nodes_pal, dtype=float)
        pco2 = np.asarray(self.pco2_nodes_ppm, dtype=float)
        if (
            po2.ndim != 1
            or pco2.ndim != 1
            or len(po2) < 3
            or len(pco2) < 3
            or not np.all(np.isfinite(po2))
            or not np.all(np.isfinite(pco2))
            or np.any(po2 <= 0.0)
            or np.any(pco2 <= 0.0)
            or np.any(np.diff(po2) <= 0.0)
            or np.any(np.diff(pco2) <= 0.0)
        ):
            raise ValueError("R7 response surface nodes must be positive and increasing")
        if len(self.operators) != len(po2) or any(
            len(row) != len(pco2) for row in self.operators
        ):
            raise ValueError("R7 response surface operator grid is not rectangular")
        if not self.source.strip():
            raise ValueError("R7 response surface requires provenance")
        object.__setattr__(self, "po2_nodes_pal", po2.copy())
        object.__setattr__(self, "pco2_nodes_ppm", pco2.copy())

    def operator_at(self, *, po2_pal: float, pco2_ppm: float) -> LocalR7ResponseOperator:
        along_pco2 = tuple(
            interpolate_local_r7_response_operators(
                self.pco2_nodes_ppm,
                row,
                pco2_ppm,
                log10_axis=True,
                source=(
                    f"{self.source}; log-pCO2 interpolation at "
                    f"{node:g} PAL, {pco2_ppm:g} ppm"
                ),
            )
            for node, row in zip(self.po2_nodes_pal, self.operators, strict=True)
        )
        return interpolate_local_r7_response_operators(
            self.po2_nodes_pal,
            along_pco2,
            po2_pal,
            log10_axis=True,
            source=(
                f"{self.source}; log-pO2 interpolation at "
                f"{po2_pal:g} PAL, {pco2_ppm:g} ppm"
            ),
        )

    def evaluate_prime_tendency_at(
        self,
        state_permil: np.ndarray,
        *,
        po2_pal: float,
        pco2_ppm: float,
        major_o2_moles_1pal: float,
    ) -> IsotopologueTendency:
        """Interpolate conserved prime-isotope rates and recover atom fluxes."""

        if (
            not np.isfinite(po2_pal)
            or not np.isfinite(pco2_ppm)
            or not np.isfinite(major_o2_moles_1pal)
            or po2_pal <= 0.0
            or pco2_ppm <= 0.0
            or major_o2_moles_1pal <= 0.0
        ):
            raise ValueError("surface evaluation requires positive finite inputs")
        if (
            po2_pal < self.po2_nodes_pal[0]
            or po2_pal > self.po2_nodes_pal[-1]
            or pco2_ppm < self.pco2_nodes_ppm[0]
            or pco2_ppm > self.pco2_nodes_ppm[-1]
        ):
            raise ValueError("R7 response surface does not permit extrapolation")
        state = np.asarray(state_permil, dtype=float)
        rate_grid = np.empty(
            (len(self.po2_nodes_pal), len(self.pco2_nodes_ppm), 2), dtype=float
        )
        for po2_index, (node_po2, operator_row) in enumerate(
            zip(self.po2_nodes_pal, self.operators, strict=True)
        ):
            node_reservoir = GlobalO2Reservoir.from_prime_composition(
                major_o2_moles=major_o2_moles_1pal * node_po2,
                delta18_prime_permil=float(state[0]),
                cap_delta17_prime_permil=float(state[1]),
                source=f"R7 response node at {node_po2:g} PAL",
            )
            for pco2_index, operator in enumerate(operator_row):
                rate_grid[po2_index, pco2_index] = (
                    prime_tendency_rates_permil_per_year(
                        node_reservoir, operator.evaluate(state)
                    )
                )

        # delta-prime-18O contains a small difference of large fluxes and is
        # interpolated without a sign transform. Delta-prime-17O is a stable,
        # one-signed R7 forcing over this domain, so log-magnitude interpolation
        # preserves its relative response across orders of magnitude in pCO2.
        delta18_along_pco2 = np.asarray(
            [
                PchipInterpolator(self.pco2_nodes_ppm, row[:, 0])(pco2_ppm)
                for row in rate_grid
            ]
        )
        delta18_rate = float(
            PchipInterpolator(
                np.log10(self.po2_nodes_pal), delta18_along_pco2
            )(np.log10(po2_pal))
        )
        cap_grid = rate_grid[:, :, 1]
        cap_along_pco2 = np.asarray(
            [
                _interpolate_signed_rate_on_log_axis(
                    self.pco2_nodes_ppm,
                    row,
                    pco2_ppm,
                )
                for row in cap_grid
            ]
        )
        cap_rate = _interpolate_signed_rate_on_log_axis(
            self.po2_nodes_pal,
            cap_along_pco2,
            po2_pal,
        )
        target_reservoir = GlobalO2Reservoir.from_prime_composition(
            major_o2_moles=major_o2_moles_1pal * po2_pal,
            delta18_prime_permil=float(state[0]),
            cap_delta17_prime_permil=float(state[1]),
            source=f"R7 response target at {po2_pal:g} PAL",
        )
        return isotopologue_tendency_from_prime_rates(
            target_reservoir,
            np.asarray([delta18_rate, cap_rate]),
            source=(
                f"prime-tendency R7 response surface at {po2_pal:g} PAL, "
                f"{pco2_ppm:g} ppm: {self.source}"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "po2_nodes_pal": self.po2_nodes_pal.tolist(),
            "pco2_nodes_ppm": self.pco2_nodes_ppm.tolist(),
            "operators": [
                [operator.as_dict() for operator in row] for row in self.operators
            ],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LocalR7ResponseSurface":
        raw_grid = data["operators"]
        if not isinstance(raw_grid, list):
            raise ValueError("serialized R7 response operator grid must be a list")
        return cls(
            po2_nodes_pal=np.asarray(data["po2_nodes_pal"], dtype=float),
            pco2_nodes_ppm=np.asarray(data["pco2_nodes_ppm"], dtype=float),
            operators=tuple(
                tuple(LocalR7ResponseOperator.from_dict(item) for item in row)
                for row in raw_grid
            ),
            source=str(data["source"]),
        )
