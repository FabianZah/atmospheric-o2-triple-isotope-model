"""Framework-neutral service boundary for the public updated model."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from isotope_field_contours import select_isotope_field_contours
from spherule_to_air_d17o import (
    air_d17o_from_spherule,
    analytical_air_d17o_sigma,
    independent_calibration_sensitivity_envelope,
)
from updated_molecular_forward_model import UpdatedForwardInput
from updated_molecular_transient import UpdatedTransientInput, run_updated_transient
from updated_constrained_pco2_posterior import (
    ConstrainedPCO2Input,
    ConstrainedCoordinateInput,
    constrained_coordinate_posterior,
    constrained_pco2_posterior,
)
from updated_output_surface import (
    UpdatedOutputSurfaceInput,
    load_updated_output_surface,
    run_updated_accelerated_forward,
)
from updated_output_surface_inverse import (
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)
from updated_output_surface_joint_posterior import (
    UpdatedJointPosteriorInput,
    joint_updated_posterior,
)
from updated_output_surface_posterior import (
    UpdatedConditionalPosteriorInput,
    conditional_updated_posterior,
)
from updated_photosynthesis_transient import (
    UpdatedPhotosynthesisTransientInput,
    run_updated_photosynthesis_transient,
)
from updated_pco2_trajectory_transient import (
    UpdatedPCO2TrajectoryInput,
    run_updated_pco2_trajectory,
)


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
CONTRACT_PATH = ROOT / "model_data" / "publication_model_contract_v1.json"
API_VERSION = "1.0"


def to_jsonable(value: Any) -> Any:
    """Convert model results to strict JSON-compatible Python values."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


@lru_cache(maxsize=1)
def publication_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def model_metadata() -> dict[str, Any]:
    contract = publication_contract()
    return {
        "api_version": API_VERSION,
        "publication_model_id": contract["publication_model_id"],
        "status": contract["status"],
        "deterministic_model": contract["deterministic_model"],
        "operational_domain": contract["operational_domain"],
        "modern_reference_state": contract["modern_reference_state"],
        "reporting_policy": contract["reporting_policy"],
        "uncertainty": contract["uncertainty"],
        "source_contract": "model_data/publication_model_contract_v1.json",
        "citation": {
            "title": (
                "OXYTIB: Atmospheric oxygen triple-isotope budget and "
                "inference model"
            ),
            "version": "0.1.0",
            "citation_file": "CITATION.cff",
            "citation_files": ("CITATION.cff", "CITATION.bib", "CITATION.ris"),
            "recommended_text": (
                "Zahnow, F. (2026). OXYTIB: Atmospheric Oxygen "
                "Triple-Isotope Budget and Inference Model (Version 0.1.0) "
                "[Computer software]. GitHub."
            ),
            "repository": (
                "https://github.com/FabianZah/"
                "atmospheric-o2-triple-isotope-model"
            ),
        },
    }


def result_envelope(result: Any, *, calculation: str) -> dict[str, Any]:
    contract = publication_contract()
    return {
        "api_version": API_VERSION,
        "publication_model_id": contract["publication_model_id"],
        "calculation": calculation,
        "result": to_jsonable(result),
        "provenance": {
            "contract": "model_data/publication_model_contract_v1.json",
            "central_model_data_id": contract["deterministic_model"]["model_data_id"],
            "accelerated_surface_data_id": contract["deterministic_model"]
            ["numerical_accelerator"]["surface_data_id"],
            "uncertainty_contract_id": contract["uncertainty"]["contract_id"],
            "extrapolation_permitted": False,
        },
    }


def forward(request: UpdatedForwardInput) -> dict[str, Any]:
    result = run_updated_accelerated_forward(
        UpdatedOutputSurfaceInput(
            p_o2_pal=request.p_o2_pal,
            p_co2_ppm=request.p_co2_ppm,
            gpp_pgC_per_year=request.gpp_pgC_per_year,
        )
    )
    return result_envelope(result.as_dict(), calculation="steady_forward")


def inverse(request: UpdatedSurfaceInverseInput) -> dict[str, Any]:
    result = invert_updated_output_surface(request)
    return result_envelope(result.as_dict(), calculation="one_coordinate_inverse")


def conditional_posterior(
    request: UpdatedConditionalPosteriorInput,
) -> dict[str, Any]:
    result = conditional_updated_posterior(request, verify_live_mode=False)
    return result_envelope(result.as_dict(), calculation="conditional_posterior")


def joint_posterior(
    request: UpdatedJointPosteriorInput, *, include_grid: bool = False
) -> dict[str, Any]:
    result = result_envelope(
        joint_updated_posterior(request).as_dict(), calculation="joint_posterior"
    )
    if include_grid:
        return result
    payload = result["result"]
    for key in (
        "posterior_density",
        "posterior_probability_mass",
        "model_cap_delta17_permil",
        "model_delta18_conventional_permil",
        "hpd_mask",
    ):
        payload.pop(key, None)
    payload["grid_included"] = False
    payload["grid_endpoint_hint"] = (
        "Repeat this request with include_grid=true to obtain the flattened "
        "posterior and HPD arrays."
    )
    return result


def constrained_pco2(request: ConstrainedPCO2Input) -> dict[str, Any]:
    result = constrained_pco2_posterior(request)
    return result_envelope(
        result.as_dict(), calculation="constrained_pco2_posterior"
    )


def constrained_coordinate(
    request: ConstrainedCoordinateInput,
) -> dict[str, Any]:
    result = constrained_coordinate_posterior(request)
    return result_envelope(
        result.as_dict(), calculation="constrained_coordinate_posterior"
    )


def isotope_field(
    *,
    p_o2_pal: float,
    pco2_bounds_ppm: tuple[float, float],
    gpp_bounds_pgC_per_year: tuple[float, float],
    pco2_grid_size: int,
    gpp_grid_size: int,
) -> dict[str, Any]:
    """Return the deterministic central isotope field at fixed pO2."""

    surface = load_updated_output_surface()
    pco2_lower, pco2_upper = map(float, pco2_bounds_ppm)
    gpp_lower, gpp_upper = map(float, gpp_bounds_pgC_per_year)
    for label, value, bounds in (
        ("pO2", p_o2_pal, surface.domain["po2_pal"]),
        ("pCO2 lower", pco2_lower, surface.domain["pco2_ppm"]),
        ("pCO2 upper", pco2_upper, surface.domain["pco2_ppm"]),
        ("GPP lower", gpp_lower, surface.domain["gpp_pgC_per_year"]),
        ("GPP upper", gpp_upper, surface.domain["gpp_pgC_per_year"]),
    ):
        if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"{label}={value:g} is outside output-surface domain "
                f"[{bounds[0]:g}, {bounds[1]:g}]"
            )
    if pco2_upper <= pco2_lower or gpp_upper <= gpp_lower:
        raise ValueError("isotope-field bounds must have lower < upper")
    if pco2_grid_size < 17 or gpp_grid_size < 17:
        raise ValueError("isotope-field grid dimensions must be at least 17")

    pco2 = np.geomspace(pco2_lower, pco2_upper, pco2_grid_size)
    gpp = np.geomspace(gpp_lower, gpp_upper, gpp_grid_size)
    pco2_mesh, gpp_mesh = np.meshgrid(pco2, gpp, indexing="ij")
    values = surface.evaluate_central_cap_delta17_grid(
        p_o2_pal=p_o2_pal,
        p_co2_ppm=pco2_mesh,
        gpp_pgC_per_year=gpp_mesh,
    )
    contours = select_isotope_field_contours(values, pco2, gpp)
    result = {
        "fixed_p_o2_pal": float(p_o2_pal),
        "axes": {
            "pCO2": tuple(map(float, pco2)),
            "GPP": tuple(map(float, gpp)),
        },
        "field_shape": tuple(values.shape),
        "central_cap_delta17_permil": tuple(map(float, values.reshape(-1))),
        "minimum_cap_delta17_permil": float(np.min(values)),
        "maximum_cap_delta17_permil": float(np.max(values)),
        "contour_levels_permil": contours.levels_permil,
        "contour_label_decimals": contours.label_decimals,
        "contour_reference_step_permil": contours.reference_step_permil,
        "contour_selection_strategy": contours.strategy,
        "surface_data_id": surface.surface_data_id,
        "upstream_model_data_id": surface.upstream_model_data_id,
        "field_scope": (
            "deterministic central OXYTIB Delta-prime-17O field; measurement "
            "likelihoods and uncertainty are handled by inference endpoints"
        ),
    }
    return result_envelope(result, calculation="deterministic_isotope_field")


def spherule_to_air(
    *,
    cap_delta17_spherule_permil: float,
    delta18_spherule_permil: float,
    cap_delta17_sigma_permil: float = 0.0,
    delta18_sigma_permil: float = 0.0,
    include_calibration_sensitivity: bool = True,
) -> dict[str, Any]:
    air = air_d17o_from_spherule(
        cap_delta17_spherule_permil, delta18_spherule_permil
    )
    analytical_sigma = analytical_air_d17o_sigma(
        cap_delta17_sigma_permil, delta18_sigma_permil
    )
    result: dict[str, Any] = {
        "input": {
            "cap_delta17_spherule_permil": cap_delta17_spherule_permil,
            "delta18_spherule_permil": delta18_spherule_permil,
            "cap_delta17_sigma_permil": cap_delta17_sigma_permil,
            "delta18_sigma_permil": delta18_sigma_permil,
        },
        "cap_delta17_air_o2_permil": air,
        "analytical_sigma_permil": analytical_sigma,
        "conversion": "Zahnow et al. (2025), Eq. 3",
        "calibration": "Fischer et al. (2021)",
    }
    if include_calibration_sensitivity:
        result["independent_calibration_sensitivity_envelope_permil"] = (
            independent_calibration_sensitivity_envelope(
                cap_delta17_spherule_permil, delta18_spherule_permil
            )
        )
        result["calibration_sensitivity_is_confidence_interval"] = False
    return result_envelope(result, calculation="spherule_to_air_conversion")


def state_step_transient(request: UpdatedTransientInput) -> dict[str, Any]:
    return result_envelope(
        run_updated_transient(request).as_dict(),
        calculation="state_step_transient",
    )


def photosynthesis_step_transient(
    request: UpdatedPhotosynthesisTransientInput,
) -> dict[str, Any]:
    return result_envelope(
        run_updated_photosynthesis_transient(request).as_dict(),
        calculation="photosynthesis_step_transient",
    )


def pco2_trajectory_transient(
    request: UpdatedPCO2TrajectoryInput,
) -> dict[str, Any]:
    return result_envelope(
        run_updated_pco2_trajectory(request).as_dict(),
        calculation="pco2_trajectory_transient",
    )
