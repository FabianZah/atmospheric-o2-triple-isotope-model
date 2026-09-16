"""Separated uncertainty layers for one updated-model prediction.

No layer is silently converted into another statistical object. In particular,
parameter corner envelopes, numerical guardrails, and structural model-model
differences are not added to analytical measurement sigma.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from updated_output_surface import (
    DEFAULT_OUTPUT_SURFACE_PATH,
    UpdatedOutputSurfaceInput,
    run_updated_accelerated_forward,
)


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
DEFAULT_UNCERTAINTY_CONTRACT_PATH = (
    ROOT / "model_data" / "uncertainty" / "updated_o2_uncertainty_layers_v1.json"
)


@dataclass(frozen=True)
class IntervalLayer:
    lower_permil: float
    central_permil: float
    upper_permil: float
    lower_distance_permil: float
    upper_distance_permil: float
    interpretation: str
    probabilistic: bool


@dataclass(frozen=True)
class MeasurementLayer:
    sigma_permil: float
    interval_one_sigma_permil: tuple[float, float]
    distribution: str | None
    source: str | None
    probabilistic: bool


@dataclass(frozen=True)
class NumericalLayer:
    upstream_response_surface_lower_margin_permil: float
    upstream_response_surface_upper_margin_permil: float
    output_accelerator_margin_permil: float
    total_lower_margin_permil: float
    total_upper_margin_permil: float
    interpretation: str
    probabilistic: bool


@dataclass(frozen=True)
class StructuralEndmemberPoint:
    endmember_id: str
    pressure_convention: str
    cap_delta17_prime_permil: float
    offset_from_central_permil: float
    delta18_offset_from_central_permil: float
    interpolation_applied: bool
    interpretation: str


@dataclass(frozen=True)
class UpdatedUncertaintyDecomposition:
    inputs: UpdatedOutputSurfaceInput
    uncertainty_contract_id: str
    central_cap_delta17_prime_permil: float
    measurement: MeasurementLayer
    source_isoflux_parameter: IntervalLayer
    biological_parameter: IntervalLayer
    crossed_parameter_envelope: IntervalLayer
    numerical: NumericalLayer
    structural_endmember_points: tuple[StructuralEndmemberPoint, ...]
    structural_evidence_ids: tuple[str, ...]
    whole_domain_structural_sigma_available: bool
    combined_public_confidence_interval_available: bool
    probability_scope: str
    export_policy: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interval(
    bounds: tuple[float, float],
    central: float,
    *,
    interpretation: str,
    probabilistic: bool = False,
) -> IntervalLayer:
    lower, upper = map(float, bounds)
    if lower > central or upper < central:
        raise ValueError("uncertainty interval does not contain the central prediction")
    return IntervalLayer(
        lower_permil=lower,
        central_permil=central,
        upper_permil=upper,
        lower_distance_permil=central - lower,
        upper_distance_permil=upper - central,
        interpretation=interpretation,
        probabilistic=probabilistic,
    )


@lru_cache(maxsize=4)
def _load_contract(path_text: str) -> dict[str, object]:
    path = Path(path_text)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported uncertainty contract schema in {path}")
    return contract


def _exact_structural_points(
    contract: dict[str, object],
    request: UpdatedOutputSurfaceInput,
) -> tuple[StructuralEndmemberPoint, ...]:
    structural = contract["layers"]["structural"]
    climates = [structural["climate_endmember"]]
    climates.extend(structural.get("climate_pressure_alternatives", []))
    climates.extend(structural.get("climate_po2_endmembers", []))
    matches = []
    for climate in climates:
        if bool(climate["interpolation_permitted"]):
            raise ValueError("v1 structural lookup expects interpolation to be disabled")
        for row in climate["values"]:
            if (
                np.isclose(float(row["pO2_PAL"]), request.p_o2_pal, atol=1.0e-12)
                and np.isclose(float(row["pCO2_ppm"]), request.p_co2_ppm, atol=1.0e-9)
                and np.isclose(
                    float(row["GPP_PgC_per_year"]),
                    request.gpp_pgC_per_year,
                    atol=1.0e-9,
                )
            ):
                matches.append(
                    StructuralEndmemberPoint(
                        endmember_id=str(climate["id"]),
                        pressure_convention=str(climate["pressure_convention"]),
                        cap_delta17_prime_permil=float(
                            row["climate_cap_delta17_prime_permil"]
                        ),
                        offset_from_central_permil=float(
                            row["climate_minus_fixed_D17O_permil"]
                        ),
                        delta18_offset_from_central_permil=float(
                            row["climate_minus_fixed_delta18_permil"]
                        ),
                        interpolation_applied=False,
                        interpretation=str(climate["role"]),
                    )
                )
    return tuple(matches)


def decompose_updated_uncertainty(
    request: UpdatedOutputSurfaceInput,
    *,
    measurement_sigma_permil: float = 0.0,
    measurement_source: str | None = None,
    surface_path: Path = DEFAULT_OUTPUT_SURFACE_PATH,
    contract_path: Path = DEFAULT_UNCERTAINTY_CONTRACT_PATH,
) -> UpdatedUncertaintyDecomposition:
    """Return uncertainty components without combining unlike layers."""

    if not isfinite(measurement_sigma_permil) or measurement_sigma_permil < 0.0:
        raise ValueError("measurement sigma must be finite and non-negative")
    if measurement_sigma_permil > 0.0 and not (measurement_source or "").strip():
        raise ValueError("positive measurement sigma requires a traceable source")

    prediction = run_updated_accelerated_forward(request, surface_path=surface_path)
    contract = _load_contract(str(Path(contract_path).resolve()))
    policy = contract["policy"]
    central = float(prediction.central_cap_delta17_prime_permil)
    measurement = MeasurementLayer(
        sigma_permil=float(measurement_sigma_permil),
        interval_one_sigma_permil=(
            central - measurement_sigma_permil,
            central + measurement_sigma_permil,
        ),
        distribution=("Gaussian" if measurement_sigma_permil > 0.0 else None),
        source=(
            None
            if measurement_sigma_permil == 0.0
            else str(measurement_source).strip()
        ),
        probabilistic=measurement_sigma_permil > 0.0,
    )
    source = _interval(
        prediction.source_isoflux_interval_cap_delta17_permil,
        central,
        interpretation=(
            "Adnew et al. (2025) minus-one-sigma and plus-one-sigma source-isoflux "
            "endpoints; output marginalization is not yet implemented"
        ),
    )
    biology = _interval(
        prediction.biological_process_interval_cap_delta17_permil,
        central,
        interpretation="source-backed biological literature-corner envelope",
    )
    crossed = _interval(
        prediction.combined_process_interval_cap_delta17_permil,
        central,
        interpretation=(
            "crossed Adnew source-isoflux endpoints and biological corner ensemble; "
            "not a confidence interval"
        ),
    )
    kernel_lower, kernel_upper = (
        prediction.interpolated_kernel_guardrail_interval_cap_delta17_permil
    )
    combined_lower, combined_upper = (
        prediction.combined_process_interval_cap_delta17_permil
    )
    accelerated_lower, accelerated_upper = (
        prediction.accelerated_model_guardrail_interval_cap_delta17_permil
    )
    numerical = NumericalLayer(
        upstream_response_surface_lower_margin_permil=combined_lower - kernel_lower,
        upstream_response_surface_upper_margin_permil=kernel_upper - combined_upper,
        output_accelerator_margin_permil=(
            prediction.output_surface_interpolation_guardrail_permil
        ),
        total_lower_margin_permil=combined_lower - accelerated_lower,
        total_upper_margin_permil=accelerated_upper - combined_upper,
        interpretation="deterministic interpolation and crossed-holdout accuracy guardrail",
        probabilistic=False,
    )
    structural = contract["layers"]["structural"]
    evidence_ids = tuple(str(item["id"]) for item in structural["sources"])
    combined_available = bool(
        policy["combined_public_confidence_interval_available"]
    )
    if combined_available:
        raise ValueError("v1 policy unexpectedly enables a combined confidence interval")
    return UpdatedUncertaintyDecomposition(
        inputs=request,
        uncertainty_contract_id=str(contract["uncertainty_contract_id"]),
        central_cap_delta17_prime_permil=central,
        measurement=measurement,
        source_isoflux_parameter=source,
        biological_parameter=biology,
        crossed_parameter_envelope=crossed,
        numerical=numerical,
        structural_endmember_points=_exact_structural_points(contract, request),
        structural_evidence_ids=evidence_ids,
        whole_domain_structural_sigma_available=(
            structural["whole_domain_sigma_permil"] is not None
        ),
        combined_public_confidence_interval_available=False,
        probability_scope=(
            "Only the declared analytical measurement sigma is probabilistic in "
            "this pointwise decomposition. Parameter endpoints, numerical margins, "
            "and structural end members remain separate."
        ),
        export_policy=(
            "Export every layer and provenance independently; do not label their "
            "union as a confidence or credible interval."
        ),
    )
