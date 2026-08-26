"""FastAPI application for the independently hosted public model."""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
import uvicorn

from public_model_service import (
    API_VERSION,
    conditional_posterior,
    constrained_coordinate,
    constrained_pco2,
    forward,
    inverse,
    isotope_field,
    joint_posterior,
    model_metadata,
    photosynthesis_step_transient,
    spherule_to_air,
    state_step_transient,
)
from updated_molecular_forward_model import UpdatedForwardInput
from updated_constrained_pco2_posterior import (
    ConstrainedPCO2Input,
    ConstrainedCoordinateInput,
    CoordinateConstraint,
)
from updated_molecular_transient import UpdatedTransientInput
from updated_output_surface_inverse import UpdatedSurfaceInverseInput
from updated_output_surface_joint_posterior import UpdatedJointPosteriorInput
from updated_output_surface_posterior import UpdatedConditionalPosteriorInput
from updated_photosynthesis_transient import UpdatedPhotosynthesisTransientInput
from model_result_workbook import (
    build_coordinate_inference_workbook,
    build_transient_workbook,
)


Coordinate = Literal["pCO2", "GPP", "pO2"]
Prior = Literal["uniform", "log_uniform", "normal"]
ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
WEB_ROOT = ROOT / "web"


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForwardRequest(StrictRequest):
    p_o2_pal: float = Field(default=1.0, gt=0.0)
    p_co2_ppm: float = Field(default=294.0, gt=0.0)
    gpp_pgC_per_year: float = Field(default=290.0, gt=0.0)

    def solver_input(self) -> UpdatedForwardInput:
        return UpdatedForwardInput(**self.model_dump())


class InverseRequest(StrictRequest):
    target_air_cap_delta17_permil: float
    solve_for: Coordinate = "pCO2"
    measurement_uncertainty_permil: float = Field(default=0.0, ge=0.0)
    p_o2_pal: float = Field(default=1.0, gt=0.0)
    p_co2_ppm: float = Field(default=294.0, gt=0.0)
    gpp_pgC_per_year: float = Field(default=290.0, gt=0.0)
    solve_bounds: tuple[float, float] | None = None

    def solver_input(self) -> UpdatedSurfaceInverseInput:
        return UpdatedSurfaceInverseInput(**self.model_dump())


class ConditionalPosteriorRequest(StrictRequest):
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float = Field(gt=0.0)
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = Field(default=None, gt=0.0)
    solve_for: Coordinate = "pCO2"
    prior: Prior = "log_uniform"
    credible_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    p_o2_pal: float = Field(default=1.0, gt=0.0)
    p_co2_ppm: float = Field(default=294.0, gt=0.0)
    gpp_pgC_per_year: float = Field(default=290.0, gt=0.0)
    solve_bounds: tuple[float, float] | None = None
    grid_size: int = Field(default=4097, ge=257, le=20001)

    def solver_input(self) -> UpdatedConditionalPosteriorInput:
        return UpdatedConditionalPosteriorInput(**self.model_dump())


class JointPosteriorRequest(StrictRequest):
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float = Field(gt=0.0)
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = Field(default=None, gt=0.0)
    free_coordinates: tuple[Coordinate, ...] = ("pCO2", "GPP")
    model_discrepancy_sigma_permil: float = Field(default=0.0, ge=0.0)
    model_discrepancy_source: str | None = None
    credible_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    p_o2_pal: float = Field(default=1.0, gt=0.0)
    p_co2_ppm: float = Field(default=294.0, gt=0.0)
    gpp_pgC_per_year: float = Field(default=290.0, gt=0.0)
    pco2_bounds_ppm: tuple[float, float] | None = None
    gpp_bounds_pgC_per_year: tuple[float, float] | None = None
    po2_bounds_pal: tuple[float, float] | None = None
    pco2_prior: Prior = "log_uniform"
    gpp_prior: Prior = "log_uniform"
    po2_prior: Prior = "uniform"
    pco2_prior_mean: float | None = None
    pco2_prior_sigma: float | None = Field(default=None, gt=0.0)
    gpp_prior_mean: float | None = None
    gpp_prior_sigma: float | None = Field(default=None, gt=0.0)
    po2_prior_mean: float | None = None
    po2_prior_sigma: float | None = Field(default=None, gt=0.0)
    pco2_grid_size: int = Field(default=161, ge=17)
    gpp_grid_size: int = Field(default=121, ge=17)
    po2_grid_size: int = Field(default=81, ge=17)

    @model_validator(mode="after")
    def bounded_public_grid(self) -> "JointPosteriorRequest":
        coordinates = tuple(self.free_coordinates)
        if len(coordinates) not in (1, 2, 3) or len(set(coordinates)) != len(coordinates):
            raise ValueError("free_coordinates must contain 1 to 3 unique coordinates")
        sizes = {
            "pCO2": self.pco2_grid_size,
            "GPP": self.gpp_grid_size,
            "pO2": self.po2_grid_size,
        }
        cells = 1
        for coordinate in coordinates:
            cells *= sizes[coordinate]
        if cells > 250_000:
            raise ValueError("joint posterior grid exceeds the 250,000-cell API limit")
        return self

    def solver_input(self) -> UpdatedJointPosteriorInput:
        return UpdatedJointPosteriorInput(**self.model_dump())


class CoordinateConstraintRequest(StrictRequest):
    kind: Literal["fixed", "normal", "range"] = "fixed"
    center: float | None = None
    sigma: float | None = Field(default=None, gt=0.0)
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def complete_constraint(self) -> "CoordinateConstraintRequest":
        if self.kind == "fixed" and self.center is None:
            raise ValueError("fixed constraint requires center")
        if self.kind == "normal" and (self.center is None or self.sigma is None):
            raise ValueError("normal constraint requires center and sigma")
        if self.kind == "range" and (
            self.lower is None or self.upper is None or self.upper <= self.lower
        ):
            raise ValueError("range constraint requires lower < upper")
        return self

    def solver_input(self) -> CoordinateConstraint:
        return CoordinateConstraint(**self.model_dump())


class ConstrainedPCO2Request(StrictRequest):
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float = Field(gt=0.0)
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = Field(default=None, gt=0.0)
    gpp_constraint: CoordinateConstraintRequest
    po2_constraint: CoordinateConstraintRequest
    pco2_bounds_ppm: tuple[float, float] = (50.0, 60000.0)
    credible_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    pco2_grid_size: int = Field(default=181, ge=17, le=401)
    gpp_grid_size: int = Field(default=81, ge=17, le=201)
    po2_grid_size: int = Field(default=17, ge=17, le=81)

    @model_validator(mode="after")
    def bounded_grid(self) -> "ConstrainedPCO2Request":
        cells = self.pco2_grid_size
        if self.gpp_constraint.kind != "fixed":
            cells *= self.gpp_grid_size
        if self.po2_constraint.kind != "fixed":
            cells *= self.po2_grid_size
        if cells > 250_000:
            raise ValueError("constrained pCO2 grid exceeds the 250,000-cell API limit")
        return self

    def solver_input(self) -> ConstrainedPCO2Input:
        payload = self.model_dump(exclude={"gpp_constraint", "po2_constraint"})
        return ConstrainedPCO2Input(
            **payload,
            gpp_constraint=self.gpp_constraint.solver_input(),
            po2_constraint=self.po2_constraint.solver_input(),
        )


class ConstrainedCoordinateRequest(StrictRequest):
    solve_for: Coordinate
    target_air_cap_delta17_permil: float
    measurement_sigma_permil: float = Field(gt=0.0)
    target_air_delta18_conventional_permil: float | None = None
    delta18_measurement_sigma_permil: float | None = Field(default=None, gt=0.0)
    pco2_constraint: CoordinateConstraintRequest | None = None
    gpp_constraint: CoordinateConstraintRequest | None = None
    po2_constraint: CoordinateConstraintRequest | None = None
    credible_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    pco2_grid_size: int = Field(default=181, ge=17, le=401)
    gpp_grid_size: int = Field(default=81, ge=17, le=201)
    po2_grid_size: int = Field(default=17, ge=17, le=81)

    @model_validator(mode="after")
    def complete_and_bounded(self) -> "ConstrainedCoordinateRequest":
        fields = {
            "pCO2": self.pco2_constraint,
            "GPP": self.gpp_constraint,
            "pO2": self.po2_constraint,
        }
        if fields[self.solve_for] is not None:
            raise ValueError("the solved coordinate must not also have a constraint")
        missing = [
            coordinate
            for coordinate, constraint in fields.items()
            if coordinate != self.solve_for and constraint is None
        ]
        if missing:
            raise ValueError(f"missing constraint(s): {', '.join(missing)}")
        sizes = {
            "pCO2": self.pco2_grid_size,
            "GPP": self.gpp_grid_size,
            "pO2": self.po2_grid_size,
        }
        cells = sizes[self.solve_for]
        for coordinate, constraint in fields.items():
            if coordinate != self.solve_for and constraint.kind != "fixed":
                cells *= sizes[coordinate]
        if cells > 250_000:
            raise ValueError(
                "constrained coordinate grid exceeds the 250,000-cell API limit"
            )
        return self

    def solver_input(self) -> ConstrainedCoordinateInput:
        fields = {
            "pCO2": self.pco2_constraint,
            "GPP": self.gpp_constraint,
            "pO2": self.po2_constraint,
        }
        constraints = {
            coordinate: constraint.solver_input()
            for coordinate, constraint in fields.items()
            if coordinate != self.solve_for
        }
        return ConstrainedCoordinateInput(
            solve_for=self.solve_for,
            target_air_cap_delta17_permil=self.target_air_cap_delta17_permil,
            measurement_sigma_permil=self.measurement_sigma_permil,
            constraints=constraints,
            target_air_delta18_conventional_permil=(
                self.target_air_delta18_conventional_permil
            ),
            delta18_measurement_sigma_permil=self.delta18_measurement_sigma_permil,
            credible_mass=self.credible_mass,
            pco2_grid_size=self.pco2_grid_size,
            gpp_grid_size=self.gpp_grid_size,
            po2_grid_size=self.po2_grid_size,
        )


class SpheruleExportContextRequest(StrictRequest):
    cap_delta17_permil: float
    cap_delta17_sigma_permil: float = Field(ge=0.0)
    delta18_permil: float
    delta18_sigma_permil: float = Field(ge=0.0)


class CoordinateWorkbookContextRequest(StrictRequest):
    isotope_source: Literal["Direct air O2", "I-type cosmic spherule"]
    spherule: SpheruleExportContextRequest | None = None

    @model_validator(mode="after")
    def source_context_is_complete(self) -> "CoordinateWorkbookContextRequest":
        if self.isotope_source == "I-type cosmic spherule" and self.spherule is None:
            raise ValueError("spherule export context is required for a spherule source")
        if self.isotope_source == "Direct air O2" and self.spherule is not None:
            raise ValueError("spherule export context is not valid for direct air")
        return self


class ConstrainedCoordinateWorkbookRequest(StrictRequest):
    inference: ConstrainedCoordinateRequest
    context: CoordinateWorkbookContextRequest


class IsotopeFieldRequest(StrictRequest):
    p_o2_pal: float = Field(default=1.0, gt=0.0)
    pco2_bounds_ppm: tuple[float, float] = (50.0, 60000.0)
    gpp_bounds_pgC_per_year: tuple[float, float] = (18.256264, 435.0)
    pco2_grid_size: int = Field(default=121, ge=17, le=401)
    gpp_grid_size: int = Field(default=101, ge=17, le=401)

    @model_validator(mode="after")
    def bounded_grid(self) -> "IsotopeFieldRequest":
        if self.pco2_grid_size * self.gpp_grid_size > 50_000:
            raise ValueError("isotope-field grid exceeds the 50,000-cell API limit")
        return self


class SpheruleRequest(StrictRequest):
    cap_delta17_spherule_permil: float
    delta18_spherule_permil: float
    cap_delta17_sigma_permil: float = Field(default=0.0, ge=0.0)
    delta18_sigma_permil: float = Field(default=0.0, ge=0.0)
    include_calibration_sensitivity: bool = True


class StateStepRequest(StrictRequest):
    initial: ForwardRequest
    final: ForwardRequest
    duration_years: float = Field(default=12000.0, gt=0.0)
    sample_count: int = Field(default=161, ge=2, le=2001)
    equilibrium_search_max_years: float = Field(default=100000.0, gt=0.0)

    def solver_input(self) -> UpdatedTransientInput:
        return UpdatedTransientInput(
            initial=self.initial.solver_input(),
            final=self.final.solver_input(),
            duration_years=self.duration_years,
            sample_count=self.sample_count,
            equilibrium_search_max_years=self.equilibrium_search_max_years,
        )


class PhotosynthesisStepRequest(StrictRequest):
    initial: ForwardRequest
    photosynthesis_fraction: float = Field(default=0.5, gt=0.0)
    duration_years: float = Field(default=12000.0, gt=0.0)
    sample_count: int = Field(default=161, ge=2, le=2001)
    equilibrium_search_max_years: float = Field(default=100000.0, gt=0.0)

    def solver_input(self) -> UpdatedPhotosynthesisTransientInput:
        return UpdatedPhotosynthesisTransientInput(
            initial=self.initial.solver_input(),
            photosynthesis_fraction=self.photosynthesis_fraction,
            duration_years=self.duration_years,
            sample_count=self.sample_count,
            equilibrium_search_max_years=self.equilibrium_search_max_years,
        )


class TransientWorkbookRequest(StrictRequest):
    experiment_type: Literal["pCO2", "pO2", "GPP", "photosynthesis"]
    state_step: StateStepRequest | None = None
    photosynthesis_step: PhotosynthesisStepRequest | None = None

    @model_validator(mode="after")
    def matching_experiment(self) -> "TransientWorkbookRequest":
        if self.experiment_type == "photosynthesis":
            if self.photosynthesis_step is None or self.state_step is not None:
                raise ValueError(
                    "photosynthesis export requires only photosynthesis_step"
                )
        elif self.state_step is None or self.photosynthesis_step is not None:
            raise ValueError("state-step export requires only state_step")
        return self


@lru_cache(maxsize=16)
def _cached_state_step(request_json: str) -> dict:
    request = StateStepRequest.model_validate_json(request_json)
    return state_step_transient(request.solver_input())


@lru_cache(maxsize=8)
def _cached_photosynthesis_step(request_json: str) -> dict:
    request = PhotosynthesisStepRequest.model_validate_json(request_json)
    return photosynthesis_step_transient(request.solver_input())


def _cors_origins() -> list[str]:
    configured = os.environ.get("O2_MODEL_CORS_ORIGINS", "")
    if configured.strip():
        return [item.strip() for item in configured.split(",") if item.strip()]
    return []


app = FastAPI(
    title="Atmospheric O2 Triple-Isotope Model API",
    version=API_VERSION,
    description=(
        "Typed calculation API for the single accepted updated model. "
        "Numerical extrapolation outside the published surface is rejected."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="web-assets")


@app.exception_handler(ValueError)
async def invalid_model_input(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/citation/model.bib")
def citation_bibtex() -> FileResponse:
    return FileResponse(
        ROOT / "CITATION.bib",
        media_type="application/x-bibtex",
        filename="atmospheric_o2_triple_isotope_model.bib",
    )


@app.get("/citation/model.ris")
def citation_ris() -> FileResponse:
    return FileResponse(
        ROOT / "CITATION.ris",
        media_type="application/x-research-info-systems",
        filename="atmospheric_o2_triple_isotope_model.ris",
    )


@app.get("/citation/CITATION.cff")
def citation_cff() -> FileResponse:
    return FileResponse(
        ROOT / "CITATION.cff",
        media_type="text/yaml",
        filename="CITATION.cff",
    )


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    metadata = model_metadata()
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "publication_model_id": metadata["publication_model_id"],
    }


@app.get("/api/v1/model")
def model() -> dict:
    return model_metadata()


@app.post("/api/v1/forward")
def steady_forward(request: ForwardRequest) -> dict:
    return forward(request.solver_input())


@app.post("/api/v1/inverse")
def one_coordinate_inverse(request: InverseRequest) -> dict:
    return inverse(request.solver_input())


@app.post("/api/v1/posterior/conditional")
def one_coordinate_posterior(request: ConditionalPosteriorRequest) -> dict:
    return conditional_posterior(request.solver_input())


@app.post("/api/v1/posterior/joint")
def solution_surface_posterior(
    request: JointPosteriorRequest,
    include_grid: bool = Query(default=False),
) -> dict:
    return joint_posterior(request.solver_input(), include_grid=include_grid)


@app.post("/api/v1/inference/pco2")
def constrained_pco2_inference(request: ConstrainedPCO2Request) -> dict:
    return constrained_pco2(request.solver_input())


@app.post("/api/v1/inference/coordinate")
def constrained_coordinate_inference(request: ConstrainedCoordinateRequest) -> dict:
    return constrained_coordinate(request.solver_input())


@app.post("/api/v1/export/coordinate.xlsx")
def constrained_coordinate_workbook(
    request: ConstrainedCoordinateWorkbookRequest,
) -> Response:
    envelope = constrained_coordinate(request.inference.solver_input())
    content = build_coordinate_inference_workbook(
        envelope,
        request.context.model_dump(exclude_none=True),
    )
    coordinate = request.inference.solve_for.lower()
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": (
                f'attachment; filename="o2_model_{coordinate}_solution.xlsx"'
            )
        },
    )


@app.post("/api/v1/field/isotope")
def deterministic_isotope_field(request: IsotopeFieldRequest) -> dict:
    return isotope_field(**request.model_dump())


@app.post("/api/v1/proxy/spherule-to-air")
def convert_spherule(request: SpheruleRequest) -> dict:
    return spherule_to_air(**request.model_dump())


@app.post("/api/v1/transients/state-step")
def state_step(request: StateStepRequest) -> dict:
    return _cached_state_step(request.model_dump_json())


@app.post("/api/v1/transients/photosynthesis-step")
def photosynthesis_step(request: PhotosynthesisStepRequest) -> dict:
    return _cached_photosynthesis_step(request.model_dump_json())


@app.post("/api/v1/export/transient.xlsx")
def transient_workbook(request: TransientWorkbookRequest) -> Response:
    if request.experiment_type == "photosynthesis":
        envelope = _cached_photosynthesis_step(
            request.photosynthesis_step.model_dump_json()
        )
    else:
        envelope = _cached_state_step(request.state_step.model_dump_json())
    content = build_transient_workbook(envelope, request.experiment_type)
    name = request.experiment_type.lower()
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": (
                f'attachment; filename="o2_model_{name}_time_response.xlsx"'
            )
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
