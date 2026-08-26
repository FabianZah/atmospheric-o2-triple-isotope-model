"""Fast-slow forward kernel for the updated atmospheric O2 isotope model.

This candidate engine combines a reduced native-photochemistry R7 response
surface with a globally mixed biological O2 reservoir. The photochemical
forcing is normalized by the observed Adnew et al. (2025) molecular CO2
anomaly isoflux using an equal-and-opposite material balance. No output offset
or fit to atmospheric O2 Delta-prime-17O is applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from biological_o2_ensemble import (
    BiologicalEnsembleMember,
    biological_ensemble_members,
    central_biological_member,
    compact_biological_envelope_members,
    fixed_po2_partitioned_biological_budget,
)
from global_o2_isotope_reservoir import (
    IsotopologueTendency,
    PartitionedBiologicalO2Budget,
    frozen_photochemical_steady_state,
)
from local_r7_response_operator import LocalR7ResponseSurface
from modern_isotope_column import modern_reference_isotope_compositions
from modern_reference_constraints import RECENT_REFERENCE_CONSTRAINTS
from self_consistent_isotope_fixed_point import (
    solve_mechanistic_fixed_point_with_hybr_fallback,
)
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
DEFAULT_BUNDLE_PATH = ROOT / "model_data" / "updated_r7_response_surface_v1.json"
FORCING_SAMPLES = ("minus_1sigma", "mean", "plus_1sigma")


@dataclass(frozen=True)
class UpdatedForwardInput:
    """Physical inputs for one updated-model forward calculation."""

    p_o2_pal: float = 1.0
    p_co2_ppm: float = 294.0
    gpp_pgC_per_year: float = 290.0


@dataclass(frozen=True)
class UpdatedCentralState:
    """Central molecular-model state without the uncertainty ensemble."""

    inputs: UpdatedForwardInput
    model_data_id: str
    delta18_prime_permil: float
    cap_delta17_prime_permil: float
    numerically_converged: bool
    maximum_fixed_point_residual_permil: float
    target_evaluations: int
    solver_method: str


@dataclass(frozen=True)
class UpdatedCentralForcing:
    """Central updated-model forcing shared by steady and transient solvers."""

    request: UpdatedForwardInput
    model_data_id: str
    transfer_convention: str
    surface: LocalR7ResponseSurface
    forcing_scale: float
    biological_member: BiologicalEnsembleMember

    def photochemical_tendency(
        self, state_permil: np.ndarray
    ) -> IsotopologueTendency:
        state = np.asarray(state_permil, dtype=float)
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError(
                "updated forcing state must contain finite delta-prime-18O "
                "and Delta-prime-17O"
            )
        native = self.surface.evaluate_prime_tendency_at(
            state,
            po2_pal=self.request.p_o2_pal,
            pco2_ppm=self.request.p_co2_ppm,
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        return _scaled_tendency(
            native,
            self.forcing_scale,
            sample="mean",
        )

    def biological_budget(
        self, photochemical: IsotopologueTendency
    ) -> PartitionedBiologicalO2Budget:
        return fixed_po2_partitioned_biological_budget(
            self.biological_member,
            po2_pal=self.request.p_o2_pal,
            gpp_pgC_per_year=self.request.gpp_pgC_per_year,
            photochemical=photochemical,
        )


@dataclass(frozen=True)
class UpdatedForwardResult:
    """Central prediction, uncertainty guardrails, and numerical provenance."""

    inputs: UpdatedForwardInput
    model_data_id: str
    transfer_convention: str
    central_delta18_prime_permil: float
    central_cap_delta17_prime_permil: float
    source_isoflux_interval_cap_delta17_permil: tuple[float, float]
    biological_process_interval_cap_delta17_permil: tuple[float, float]
    combined_process_interval_cap_delta17_permil: tuple[float, float]
    interpolation_guardrail_permil: float
    model_guardrail_interval_cap_delta17_permil: tuple[float, float]
    pack_observed_cap_delta17_permil: float
    pack_observed_uncertainty_permil: float
    pack_observed_interval_permil: tuple[float, float]
    central_residual_to_pack_permil: float
    model_guardrail_overlaps_pack_observation: bool
    numerically_converged: bool
    maximum_fixed_point_residual_permil: float
    maximum_target_evaluations: int
    numerical_solver_methods_used: tuple[str, ...]
    forcing_scale_samples: dict[str, float]
    sample_states: dict[str, dict[str, float]]
    biological_envelope_extrema: dict[str, dict[str, float | str]]
    uncertainty_policy: dict[str, str | bool]
    biological_convention: str
    biological_pathway_uncertainty_included: bool
    biological_full_ensemble_member_count: int
    biological_compact_envelope_member_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pack_constraint():
    return next(
        item
        for item in RECENT_REFERENCE_CONSTRAINTS
        if item.key == "modern_o2_delta17o_pack_2021"
    )


@lru_cache(maxsize=4)
def _load_bundle(path_text: str) -> tuple[dict[str, object], LocalR7ResponseSurface]:
    path = Path(path_text)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1:
        raise ValueError(f"unsupported updated-model data schema in {path}")
    surface = LocalR7ResponseSurface.from_dict(bundle["surface"])
    return bundle, surface


def _validate_input(request: UpdatedForwardInput, bundle: dict[str, object]) -> None:
    values = np.asarray(
        (request.p_o2_pal, request.p_co2_ppm, request.gpp_pgC_per_year),
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("pO2, pCO2, and GPP must be finite and positive")
    domain = bundle["domain"]
    checks = (
        ("pO2", request.p_o2_pal, domain["po2_pal"], "PAL"),
        ("pCO2", request.p_co2_ppm, domain["pco2_ppm"], "ppm"),
        ("GPP", request.gpp_pgC_per_year, domain["gpp_pgC_per_year"], "PgC yr-1"),
    )
    for label, value, bounds, units in checks:
        lower, upper = map(float, bounds)
        if not lower <= value <= upper:
            raise ValueError(
                f"{label}={value:g} {units} is outside the mechanistic "
                f"response domain [{lower:g}, {upper:g}] {units}"
            )


def _scaled_tendency(
    tendency: IsotopologueTendency,
    scale: float,
    *,
    sample: str,
) -> IsotopologueTendency:
    return IsotopologueTendency(
        *map(float, scale * tendency.values),
        source=(
            f"{tendency.source}; {sample} Adnew molecular-isoflux "
            f"normalization={scale:.12g}"
        ),
    )


def _solve_sample(
    surface: LocalR7ResponseSurface,
    request: UpdatedForwardInput,
    *,
    scale: float,
    sample: str,
    biological_member: BiologicalEnsembleMember | None = None,
    marine_accessible_fraction: float = 1.0,
    marine_accessibility_source: str | None = None,
):
    reference_o2, _reference_co2 = modern_reference_isotope_compositions()
    member = central_biological_member() if biological_member is None else biological_member

    def evaluate_target(state: np.ndarray) -> np.ndarray:
        native_photo = surface.evaluate_prime_tendency_at(
            state,
            po2_pal=request.p_o2_pal,
            pco2_ppm=request.p_co2_ppm,
            major_o2_moles_1pal=GLOBAL_MAJOR_O2_MOLES_1PAL,
        )
        photo = _scaled_tendency(native_photo, scale, sample=sample)
        biology = fixed_po2_partitioned_biological_budget(
            member,
            po2_pal=request.p_o2_pal,
            gpp_pgC_per_year=request.gpp_pgC_per_year,
            photochemical=photo,
            marine_accessible_fraction=marine_accessible_fraction,
            marine_accessibility_source=marine_accessibility_source,
        )
        target = frozen_photochemical_steady_state(
            biology,
            photo,
            source=(
                "updated molecular-bridge global O2 fixed-point target; "
                f"source-backed partitioned biological member={member.key}"
            ),
        )
        return np.asarray(
            [target.delta18_prime_permil, target.cap_delta17_prime_permil]
        )

    result = solve_mechanistic_fixed_point_with_hybr_fallback(
        evaluate_target,
        np.asarray(
            [
                reference_o2.delta18_prime_permil,
                reference_o2.cap_delta17_prime_permil,
            ]
        ),
        # The reduced response surface is smooth enough for smaller isotope-
        # coordinate perturbations than the expensive direct-column solver.
        # These steps avoid a coarse-Jacobian convergence stall at low pO2/GPP.
        finite_difference_step=np.asarray([0.001, 0.0001]),
        tolerance=1.0e-8,
        maximum_iterations=36,
    )
    if not result.converged:
        raise RuntimeError(
            "updated molecular forward model failed to converge: "
            f"residual={result.maximum_absolute_residual:.6g} per mil at "
            f"pO2={request.p_o2_pal:g} PAL, pCO2={request.p_co2_ppm:g} ppm, "
            f"GPP={request.gpp_pgC_per_year:g} PgC yr-1, sample={sample}"
        )
    return result


def run_updated_central_state(
    request: UpdatedForwardInput,
    *,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    marine_accessible_fraction: float = 1.0,
    marine_accessibility_source: str | None = None,
) -> UpdatedCentralState:
    """Solve only the central molecular state used by deterministic caches."""

    resolved_path = Path(bundle_path).resolve()
    bundle, surface = _load_bundle(str(resolved_path))
    _validate_input(request, bundle)
    normalization = bundle["transfer_normalization"]
    scale = float(
        normalization["native_surface_molecular_forcing_scale"]["mean"]
    )
    result = _solve_sample(
        surface,
        request,
        scale=scale,
        sample="mean",
        marine_accessible_fraction=marine_accessible_fraction,
        marine_accessibility_source=marine_accessibility_source,
    )
    return UpdatedCentralState(
        inputs=request,
        model_data_id=str(bundle["model_data_id"]),
        delta18_prime_permil=float(result.state[0]),
        cap_delta17_prime_permil=float(result.state[1]),
        numerically_converged=bool(result.converged),
        maximum_fixed_point_residual_permil=float(
            result.maximum_absolute_residual
        ),
        target_evaluations=int(result.target_evaluations),
        solver_method=result.solver_method,
    )


def build_updated_central_forcing(
    request: UpdatedForwardInput,
    *,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
) -> UpdatedCentralForcing:
    """Build the central physical forcing used by updated-model transients."""

    resolved_path = Path(bundle_path).resolve()
    bundle, surface = _load_bundle(str(resolved_path))
    _validate_input(request, bundle)
    normalization = bundle["transfer_normalization"]
    forcing_scale = float(
        normalization["native_surface_molecular_forcing_scale"]["mean"]
    )
    return UpdatedCentralForcing(
        request=request,
        model_data_id=str(bundle["model_data_id"]),
        transfer_convention=str(normalization["convention"]),
        surface=surface,
        forcing_scale=forcing_scale,
        biological_member=central_biological_member(),
    )


def run_updated_forward(
    request: UpdatedForwardInput,
    *,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
) -> UpdatedForwardResult:
    """Run the updated fast-slow model without observation-referencing output."""

    resolved_path = Path(bundle_path).resolve()
    bundle, surface = _load_bundle(str(resolved_path))
    _validate_input(request, bundle)
    normalization = bundle["transfer_normalization"]
    scales = {
        key: float(value)
        for key, value in normalization["native_surface_molecular_forcing_scale"].items()
    }
    if set(scales) != set(FORCING_SAMPLES):
        raise ValueError("updated-model bundle lacks the three Adnew forcing samples")

    solved = {
        sample: _solve_sample(surface, request, scale=scales[sample], sample=sample)
        for sample in FORCING_SAMPLES
    }
    states = {
        sample: {
            "delta18_prime_permil": float(result.state[0]),
            "cap_delta17_prime_permil": float(result.state[1]),
            "fixed_point_residual_permil": float(result.maximum_absolute_residual),
            "target_evaluations": int(result.target_evaluations),
            "solver_method": result.solver_method,
        }
        for sample, result in solved.items()
    }
    source_values = np.asarray(
        [states[sample]["cap_delta17_prime_permil"] for sample in FORCING_SAMPLES]
    )
    source_interval = (float(np.min(source_values)), float(np.max(source_values)))

    compact_members = compact_biological_envelope_members()
    combined_solved = {
        (sample, member.key): _solve_sample(
            surface,
            request,
            scale=scales[sample],
            sample=sample,
            biological_member=member,
        )
        for sample in FORCING_SAMPLES
        for member in compact_members
    }
    combined_values = {
        key: float(result.state[1]) for key, result in combined_solved.items()
    }
    biological_values = np.asarray(
        [combined_values[("mean", member.key)] for member in compact_members]
    )
    biological_interval = (
        float(np.min(biological_values)),
        float(np.max(biological_values)),
    )
    lower_key = min(combined_values, key=combined_values.get)
    upper_key = max(combined_values, key=combined_values.get)
    combined_interval = (
        combined_values[lower_key],
        combined_values[upper_key],
    )
    interpolation_guardrail = float(
        bundle["validation"]["cap_delta17_maximum_absolute_residual_permil"]
    )
    model_interval = (
        combined_interval[0] - interpolation_guardrail,
        combined_interval[1] + interpolation_guardrail,
    )

    pack = _pack_constraint()
    if not isinstance(pack.value, float) or not isinstance(pack.uncertainty, float):
        raise TypeError("Pack O2 constraint must have scalar value and uncertainty")
    pack_interval = (
        pack.value - pack.uncertainty,
        pack.value + pack.uncertainty,
    )
    overlap = max(model_interval[0], pack_interval[0]) <= min(
        model_interval[1], pack_interval[1]
    )
    central = states["mean"]

    return UpdatedForwardResult(
        inputs=request,
        model_data_id=str(bundle["model_data_id"]),
        transfer_convention=str(normalization["convention"]),
        central_delta18_prime_permil=central["delta18_prime_permil"],
        central_cap_delta17_prime_permil=central["cap_delta17_prime_permil"],
        source_isoflux_interval_cap_delta17_permil=source_interval,
        biological_process_interval_cap_delta17_permil=biological_interval,
        combined_process_interval_cap_delta17_permil=combined_interval,
        interpolation_guardrail_permil=interpolation_guardrail,
        model_guardrail_interval_cap_delta17_permil=model_interval,
        pack_observed_cap_delta17_permil=pack.value,
        pack_observed_uncertainty_permil=pack.uncertainty,
        pack_observed_interval_permil=pack_interval,
        central_residual_to_pack_permil=(
            central["cap_delta17_prime_permil"] - pack.value
        ),
        model_guardrail_overlaps_pack_observation=bool(overlap),
        numerically_converged=all(
            result.converged
            for result in (*solved.values(), *combined_solved.values())
        ),
        maximum_fixed_point_residual_permil=max(
            float(result.maximum_absolute_residual)
            for result in (*solved.values(), *combined_solved.values())
        ),
        maximum_target_evaluations=max(
            int(result.target_evaluations)
            for result in (*solved.values(), *combined_solved.values())
        ),
        numerical_solver_methods_used=tuple(
            sorted(
                {
                    result.solver_method
                    for result in (*solved.values(), *combined_solved.values())
                }
            )
        ),
        forcing_scale_samples=scales,
        sample_states=states,
        biological_envelope_extrema={
            "lower": {
                "Adnew_forcing_sample": lower_key[0],
                "biological_member": lower_key[1],
                "cap_delta17_prime_permil": combined_values[lower_key],
            },
            "upper": {
                "Adnew_forcing_sample": upper_key[0],
                "biological_member": upper_key[1],
                "cap_delta17_prime_permil": combined_values[upper_key],
            },
        },
        uncertainty_policy={
            "Adnew_source_isoflux_propagated": True,
            "biological_literature_corner_envelope_propagated": True,
            "response_surface_holdout_error_propagated_as_guardrail": True,
            "Pack_analytical_uncertainty_used_only_for_observation_overlap": True,
            "output_offset_applied": False,
            "interval_interpretation": (
                "Guardrail envelope, not a formal posterior or confidence interval"
            ),
        },
        biological_convention=(
            "Liang et al. (2023) central land-ocean partition; Barkan-Luz marine "
            "source as interpreted by Young et al. (2014); proportional Young "
            "respiration pathways; midpoint of Bender et al. (1994) terrestrial "
            "source-water interval; no atmospheric-O2 isotope calibration"
        ),
        biological_pathway_uncertainty_included=True,
        biological_full_ensemble_member_count=len(biological_ensemble_members()),
        biological_compact_envelope_member_keys=tuple(
            item.key for item in compact_members
        ),
    )
