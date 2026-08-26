"""Versioned isotope accelerator for the updated molecular model.

The surface is a numerical cache, not a separate physical model.  Central
Delta-prime-17O is interpolated with a tensor-product cubic spline.  Positive
distances from the central prediction to each uncertainty bound are
interpolated in log space, so interval ordering is guaranteed between nodes.
Delta-prime-18O uses its own refined grid because its response has stronger
curvature at low GPP and high pCO2 than the Delta-prime-17O fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
DEFAULT_OUTPUT_SURFACE_PATH = (
    ROOT / "model_data" / "updated_molecular_output_surface_v1.json"
)

SURFACE_FIELDS = (
    "central_delta18_prime_permil",
    "central_cap_delta17_prime_permil",
    "source_isoflux_lower_cap_delta17_permil",
    "source_isoflux_upper_cap_delta17_permil",
    "biological_process_lower_cap_delta17_permil",
    "biological_process_upper_cap_delta17_permil",
    "combined_process_lower_cap_delta17_permil",
    "combined_process_upper_cap_delta17_permil",
    "model_guardrail_lower_cap_delta17_permil",
    "model_guardrail_upper_cap_delta17_permil",
)


@dataclass(frozen=True)
class UpdatedOutputSurfaceInput:
    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.0
    gpp_pgC_per_year: float = 290.0


@dataclass(frozen=True)
class UpdatedOutputSurfacePrediction:
    inputs: UpdatedOutputSurfaceInput
    surface_data_id: str
    upstream_model_data_id: str
    central_delta18_prime_permil: float
    central_delta18_acceleration_validated: bool
    central_cap_delta17_prime_permil: float
    source_isoflux_interval_cap_delta17_permil: tuple[float, float]
    biological_process_interval_cap_delta17_permil: tuple[float, float]
    combined_process_interval_cap_delta17_permil: tuple[float, float]
    interpolated_kernel_guardrail_interval_cap_delta17_permil: tuple[float, float]
    output_surface_interpolation_guardrail_permil: float
    accelerated_model_guardrail_interval_cap_delta17_permil: tuple[float, float]
    interpolation_method: str
    interpolation_coordinates: str
    extrapolation_applied: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class UpdatedMolecularOutputSurface:
    """Validated rectangular output surface around the live updated kernel."""

    def __init__(self, bundle: dict[str, object]):
        if int(bundle.get("schema_version", -1)) != 1:
            raise ValueError("unsupported updated output-surface schema")
        self.bundle = bundle
        self.surface_data_id = str(bundle["surface_data_id"])
        self.upstream_model_data_id = str(bundle["upstream_model_data_id"])
        axes = bundle["axes"]
        self.po2_nodes = self._validate_axis("pO2", axes["po2_pal"])
        self.pco2_nodes = self._validate_axis("pCO2", axes["pco2_ppm"])
        self.gpp_nodes = self._validate_axis("GPP", axes["gpp_pgC_per_year"])
        expected_shape = (
            len(self.po2_nodes),
            len(self.pco2_nodes),
            len(self.gpp_nodes),
        )
        fields = bundle["fields"]
        if set(fields) != set(SURFACE_FIELDS):
            missing = set(SURFACE_FIELDS) - set(fields)
            extra = set(fields) - set(SURFACE_FIELDS)
            raise ValueError(
                f"output surface fields differ from schema; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        self.values: dict[str, np.ndarray] = {}
        coordinates = (self.po2_nodes, self.pco2_nodes, np.log(self.gpp_nodes))
        for key in SURFACE_FIELDS:
            value = np.asarray(fields[key], dtype=float)
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"output field {key!r} must be finite with shape {expected_shape}"
                )
            self.values[key] = value
        self._validate_interval_order()
        self._central_d17_interpolator = RegularGridInterpolator(
            coordinates,
            self.values["central_cap_delta17_prime_permil"],
            method="cubic",
            bounds_error=True,
        )
        delta18_surface = bundle.get("delta18_surface")
        if delta18_surface is None:
            self._delta18_interpolator = RegularGridInterpolator(
                coordinates,
                self.values["central_delta18_prime_permil"],
                method="linear",
                bounds_error=True,
            )
            self._delta18_interpolation_method = "diagnostic trilinear"
            self._delta18_uses_log_pco2 = False
        else:
            delta18_axes = delta18_surface["axes"]
            delta18_po2 = self._validate_axis("delta18 pO2", delta18_axes["po2_pal"])
            delta18_pco2 = self._validate_axis(
                "delta18 pCO2", delta18_axes["pco2_ppm"]
            )
            delta18_gpp = self._validate_axis(
                "delta18 GPP", delta18_axes["gpp_pgC_per_year"]
            )
            delta18_values = np.asarray(delta18_surface["values"], dtype=float)
            delta18_shape = (
                len(delta18_po2),
                len(delta18_pco2),
                len(delta18_gpp),
            )
            if (
                delta18_values.shape != delta18_shape
                or not np.all(np.isfinite(delta18_values))
            ):
                raise ValueError(
                    "delta18 surface values must be finite with shape "
                    f"{delta18_shape}"
                )
            if not (
                delta18_po2[0] == self.po2_nodes[0]
                and delta18_po2[-1] == self.po2_nodes[-1]
                and delta18_pco2[0] == self.pco2_nodes[0]
                and delta18_pco2[-1] == self.pco2_nodes[-1]
                and delta18_gpp[0] == self.gpp_nodes[0]
                and delta18_gpp[-1] == self.gpp_nodes[-1]
            ):
                raise ValueError("delta18 and Delta-prime-17O domains must match")
            self._delta18_interpolator = _LocalTensorQuadraticInterpolator(
                (
                    delta18_po2,
                    np.log(delta18_pco2),
                    np.log(delta18_gpp),
                ),
                delta18_values,
            )
            self._delta18_interpolation_method = (
                "local tensor-product quadratic in pO2, ln(pCO2), and ln(GPP)"
            )
            self._delta18_uses_log_pco2 = True
        central = self.values["central_cap_delta17_prime_permil"]
        self._interval_width_interpolators = {}
        for lower_key, upper_key in self._interval_pairs():
            lower_width = central - self.values[lower_key]
            upper_width = self.values[upper_key] - central
            if np.any(lower_width <= 0.0) or np.any(upper_width <= 0.0):
                raise ValueError(
                    f"stored uncertainty widths must be positive for {lower_key}"
                )
            self._interval_width_interpolators[lower_key] = (
                RegularGridInterpolator(
                    coordinates,
                    np.log(lower_width),
                    method="cubic",
                    bounds_error=True,
                ),
                RegularGridInterpolator(
                    coordinates,
                    np.log(upper_width),
                    method="cubic",
                    bounds_error=True,
                ),
            )
        validation = bundle["validation"]
        self.delta18_acceleration_validated = bool(
            validation.get("delta18_acceleration_validated", False)
        )
        self.maximum_holdout_cap_delta17_residual_permil = float(
            validation["cap_delta17_maximum_absolute_residual_permil"]
        )
        if (
            not np.isfinite(self.maximum_holdout_cap_delta17_residual_permil)
            or self.maximum_holdout_cap_delta17_residual_permil < 0.0
        ):
            raise ValueError("output-surface interpolation guardrail must be non-negative")

    @staticmethod
    def _validate_axis(label: str, values: object) -> np.ndarray:
        axis = np.asarray(values, dtype=float)
        if (
            axis.ndim != 1
            or len(axis) < 4
            or not np.all(np.isfinite(axis))
            or np.any(axis <= 0.0)
            or np.any(np.diff(axis) <= 0.0)
        ):
            raise ValueError(
                f"{label} output-surface nodes must contain at least four "
                "positive, increasing values"
            )
        return axis

    def _validate_interval_order(self) -> None:
        central = self.values["central_cap_delta17_prime_permil"]
        for lower_key, upper_key in self._interval_pairs():
            lower = self.values[lower_key]
            upper = self.values[upper_key]
            if np.any(lower > upper):
                raise ValueError(f"stored interval order fails for {lower_key}")
            if np.any(central <= lower) or np.any(central >= upper):
                raise ValueError(f"central Delta-prime-17O lies outside {lower_key}")

    @staticmethod
    def _interval_pairs() -> tuple[tuple[str, str], ...]:
        return (
            (
                "source_isoflux_lower_cap_delta17_permil",
                "source_isoflux_upper_cap_delta17_permil",
            ),
            (
                "biological_process_lower_cap_delta17_permil",
                "biological_process_upper_cap_delta17_permil",
            ),
            (
                "combined_process_lower_cap_delta17_permil",
                "combined_process_upper_cap_delta17_permil",
            ),
            (
                "model_guardrail_lower_cap_delta17_permil",
                "model_guardrail_upper_cap_delta17_permil",
            ),
        )

    @property
    def domain(self) -> dict[str, tuple[float, float]]:
        return {
            "po2_pal": (float(self.po2_nodes[0]), float(self.po2_nodes[-1])),
            "pco2_ppm": (float(self.pco2_nodes[0]), float(self.pco2_nodes[-1])),
            "gpp_pgC_per_year": (
                float(self.gpp_nodes[0]),
                float(self.gpp_nodes[-1]),
            ),
        }

    def _point(self, request: UpdatedOutputSurfaceInput) -> np.ndarray:
        values = np.asarray(
            (request.p_o2_pal, request.p_co2_ppm, request.gpp_pgC_per_year),
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("pO2, pCO2, and absolute GPP must be finite and positive")
        for label, value, bounds in (
            ("pO2", values[0], self.domain["po2_pal"]),
            ("pCO2", values[1], self.domain["pco2_ppm"]),
            ("GPP", values[2], self.domain["gpp_pgC_per_year"]),
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"{label}={value:g} is outside output-surface domain "
                    f"[{bounds[0]:g}, {bounds[1]:g}]"
                )
        return np.asarray((values[0], values[1], np.log(values[2])), dtype=float)

    def evaluate(
        self, request: UpdatedOutputSurfaceInput
    ) -> UpdatedOutputSurfacePrediction:
        point = self._point(request)
        delta18_point = (
            np.asarray((point[0], np.log(point[1]), point[2]), dtype=float)
            if self._delta18_uses_log_pco2
            else point
        )
        central = float(self._central_d17_interpolator(point)[0])
        delta18_value = np.asarray(
            self._delta18_interpolator(delta18_point), dtype=float
        ).reshape(-1)[0]
        interpolated = {
            "central_delta18_prime_permil": float(delta18_value),
            "central_cap_delta17_prime_permil": central,
        }
        for lower_key, upper_key in self._interval_pairs():
            lower_interpolator, upper_interpolator = (
                self._interval_width_interpolators[lower_key]
            )
            lower_width = float(np.exp(lower_interpolator(point)[0]))
            upper_width = float(np.exp(upper_interpolator(point)[0]))
            interpolated[lower_key] = central - lower_width
            interpolated[upper_key] = central + upper_width
        numerical = self.maximum_holdout_cap_delta17_residual_permil
        kernel_interval = (
            interpolated["model_guardrail_lower_cap_delta17_permil"],
            interpolated["model_guardrail_upper_cap_delta17_permil"],
        )
        return UpdatedOutputSurfacePrediction(
            inputs=request,
            surface_data_id=self.surface_data_id,
            upstream_model_data_id=self.upstream_model_data_id,
            central_delta18_prime_permil=interpolated[
                "central_delta18_prime_permil"
            ],
            central_delta18_acceleration_validated=(
                self.delta18_acceleration_validated
            ),
            central_cap_delta17_prime_permil=interpolated[
                "central_cap_delta17_prime_permil"
            ],
            source_isoflux_interval_cap_delta17_permil=(
                interpolated["source_isoflux_lower_cap_delta17_permil"],
                interpolated["source_isoflux_upper_cap_delta17_permil"],
            ),
            biological_process_interval_cap_delta17_permil=(
                interpolated["biological_process_lower_cap_delta17_permil"],
                interpolated["biological_process_upper_cap_delta17_permil"],
            ),
            combined_process_interval_cap_delta17_permil=(
                interpolated["combined_process_lower_cap_delta17_permil"],
                interpolated["combined_process_upper_cap_delta17_permil"],
            ),
            interpolated_kernel_guardrail_interval_cap_delta17_permil=(
                kernel_interval
            ),
            output_surface_interpolation_guardrail_permil=numerical,
            accelerated_model_guardrail_interval_cap_delta17_permil=(
                kernel_interval[0] - numerical,
                kernel_interval[1] + numerical,
            ),
            interpolation_method=(
                "tensor-product cubic central Delta-prime-17O with cubic "
                "log-width uncertainty bounds; "
                f"{self._delta18_interpolation_method} Delta-prime-18O"
            ),
            interpolation_coordinates="pO2, pCO2, and ln(absolute GPP)",
            extrapolation_applied=False,
        )

    def evaluate_central_cap_delta17_grid(
        self,
        *,
        p_o2_pal: np.ndarray | float,
        p_co2_ppm: np.ndarray | float,
        gpp_pgC_per_year: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate the central isotope field on broadcast-compatible arrays."""

        po2, pco2, gpp = np.broadcast_arrays(
            np.asarray(p_o2_pal, dtype=float),
            np.asarray(p_co2_ppm, dtype=float),
            np.asarray(gpp_pgC_per_year, dtype=float),
        )
        if (
            not np.all(np.isfinite(po2))
            or not np.all(np.isfinite(pco2))
            or not np.all(np.isfinite(gpp))
            or np.any(po2 <= 0.0)
            or np.any(pco2 <= 0.0)
            or np.any(gpp <= 0.0)
        ):
            raise ValueError("pO2, pCO2, and absolute GPP must be finite and positive")
        for label, values, bounds in (
            ("pO2", po2, self.domain["po2_pal"]),
            ("pCO2", pco2, self.domain["pco2_ppm"]),
            ("GPP", gpp, self.domain["gpp_pgC_per_year"]),
        ):
            if np.any(values < bounds[0]) or np.any(values > bounds[1]):
                raise ValueError(
                    f"{label} grid is outside output-surface domain "
                    f"[{bounds[0]:g}, {bounds[1]:g}]"
                )
        points = np.column_stack(
            (po2.reshape(-1), pco2.reshape(-1), np.log(gpp.reshape(-1)))
        )
        evaluated = np.asarray(self._central_d17_interpolator(points), dtype=float)
        return evaluated.reshape(po2.shape)

    def evaluate_central_delta18_prime_grid(
        self,
        *,
        p_o2_pal: np.ndarray | float,
        p_co2_ppm: np.ndarray | float,
        gpp_pgC_per_year: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate central delta-prime-18O on broadcast-compatible arrays."""

        po2, pco2, gpp = np.broadcast_arrays(
            np.asarray(p_o2_pal, dtype=float),
            np.asarray(p_co2_ppm, dtype=float),
            np.asarray(gpp_pgC_per_year, dtype=float),
        )
        if (
            not np.all(np.isfinite(po2))
            or not np.all(np.isfinite(pco2))
            or not np.all(np.isfinite(gpp))
            or np.any(po2 <= 0.0)
            or np.any(pco2 <= 0.0)
            or np.any(gpp <= 0.0)
        ):
            raise ValueError("pO2, pCO2, and absolute GPP must be finite and positive")
        for label, values, bounds in (
            ("pO2", po2, self.domain["po2_pal"]),
            ("pCO2", pco2, self.domain["pco2_ppm"]),
            ("GPP", gpp, self.domain["gpp_pgC_per_year"]),
        ):
            if np.any(values < bounds[0]) or np.any(values > bounds[1]):
                raise ValueError(
                    f"{label} grid is outside output-surface domain "
                    f"[{bounds[0]:g}, {bounds[1]:g}]"
                )
        points = np.column_stack(
            (
                po2.reshape(-1),
                np.log(pco2.reshape(-1)) if self._delta18_uses_log_pco2 else pco2.reshape(-1),
                np.log(gpp.reshape(-1)),
            )
        )
        if isinstance(self._delta18_interpolator, RegularGridInterpolator):
            evaluated = np.asarray(self._delta18_interpolator(points), dtype=float)
        else:
            evaluated = np.fromiter(
                (self._delta18_interpolator(point) for point in points),
                dtype=float,
                count=len(points),
            )
        return evaluated.reshape(po2.shape)


class _LocalTensorQuadraticInterpolator:
    """Exact local three-point tensor interpolation for one scalar field."""

    def __init__(self, axes: tuple[np.ndarray, ...], values: np.ndarray):
        self.axes = axes
        self.values = values

    @staticmethod
    def _indices_and_weights(axis: np.ndarray, query: float):
        interval = int(np.searchsorted(axis, query) - 1)
        start = max(0, min(len(axis) - 3, interval))
        indices = np.arange(start, start + 3)
        nodes = axis[indices]
        weights = np.ones(3, dtype=float)
        for i in range(3):
            for j in range(3):
                if i != j:
                    weights[i] *= (query - nodes[j]) / (nodes[i] - nodes[j])
        return indices, weights

    def __call__(self, point: np.ndarray) -> float:
        selections = [
            self._indices_and_weights(axis, float(query))
            for axis, query in zip(self.axes, point)
        ]
        (i, wi), (j, wj), (k, wk) = selections
        return float(
            np.einsum(
                "a,b,c,abc->",
                wi,
                wj,
                wk,
                self.values[np.ix_(i, j, k)],
            )
        )


@lru_cache(maxsize=4)
def load_updated_output_surface(
    path_text: str = str(DEFAULT_OUTPUT_SURFACE_PATH),
) -> UpdatedMolecularOutputSurface:
    path = Path(path_text).resolve()
    return UpdatedMolecularOutputSurface(
        json.loads(path.read_text(encoding="utf-8"))
    )


def run_updated_accelerated_forward(
    request: UpdatedOutputSurfaceInput,
    *,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
) -> UpdatedOutputSurfacePrediction:
    return load_updated_output_surface(str(Path(surface_path).resolve())).evaluate(request)
