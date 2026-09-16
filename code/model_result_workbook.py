"""Publication-grade XLSX export for constrained coordinate inference."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO
import json
from typing import Any

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.xml import LXML
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


TITLE_FILL = PatternFill("solid", fgColor="123238")
HEADER_FILL = PatternFill("solid", fgColor="006F71")
TITLE_FONT = Font(color="FFFFFF", bold=True, size=14)
HEADER_FONT = Font(color="FFFFFF", bold=True)
SOFTWARE_NAME = "OXYTIB"
SOFTWARE_VERSION = "0.1.0"
REPOSITORY_URL = "https://github.com/FabianZah/atmospheric-o2-triple-isotope-model"
DATA_ALIGNMENT = Alignment(vertical="top", wrap_text=True)


class _StreamingTable:
    """Style each row before serializing it, retaining no worksheet cell grid."""

    def __init__(
        self, workbook: Workbook, name: str, *, title: str,
        headers: tuple[str, ...], widths: tuple[float, ...],
        number_formats: dict[int, str] | None = None,
    ) -> None:
        self.sheet = workbook.create_sheet(name)
        self.number_formats = number_formats or {}
        self.columns = len(headers)
        self.row_count = 3
        self.sheet.freeze_panes = "A4"
        self.sheet.row_dimensions[1].height = 24
        self.sheet.merged_cells.add(f"A1:{get_column_letter(self.columns)}1")
        for index, width in enumerate(widths, start=1):
            self.sheet.column_dimensions[get_column_letter(index)].width = width
        cell = WriteOnlyCell(self.sheet, title)
        cell.fill = TITLE_FILL
        cell.font = TITLE_FONT
        cell.alignment = Alignment(vertical="center")
        self.sheet.append([cell])
        self.sheet.append([])
        cells = []
        for header in headers:
            cell = WriteOnlyCell(self.sheet, header)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(vertical="center")
            cells.append(cell)
        self.sheet.append(cells)

    def append(self, values: tuple[Any, ...]) -> None:
        cells = []
        for column, value in enumerate(values, start=1):
            cell = WriteOnlyCell(self.sheet, value)
            # Metadata is literal text, including strings beginning with '='.
            if isinstance(value, str):
                cell.data_type = "s"
            cell.alignment = DATA_ALIGNMENT
            if column in self.number_formats:
                cell.number_format = self.number_formats[column]
            cells.append(cell)
        self.sheet.append(cells)
        self.row_count += 1

    def finish(self) -> None:
        self.sheet.auto_filter.ref = f"A3:{get_column_letter(self.columns)}{self.row_count}"
        self.sheet.close()


@contextmanager
def _streaming_workbook() -> Iterator[Workbook]:
    if not LXML:
        raise RuntimeError(
            "Streaming XLSX export requires lxml; install code/requirements-api.txt "
            "and ensure OPENPYXL_LXML is enabled."
        )
    workbook = Workbook(write_only=True)
    try:
        yield workbook
    finally:
        # openpyxl only removes its XML files after a successful save. On an
        # aborted export, clean up this workbook's writers, never shared temp files.
        for sheet in workbook.worksheets:
            writer = sheet._writer
            if writer is None:
                continue
            with suppress(Exception):
                if not sheet.closed:
                    sheet.close()
            with suppress(FileNotFoundError):
                writer.cleanup()
        workbook.close()


def build_forward_isotope_workbook(envelope: dict[str, Any]) -> bytes:
    result = envelope["result"]
    with _streaming_workbook() as workbook:
        summary = _StreamingTable(workbook, "Summary", title="Atmospheric O2 steady-state prediction",
                                  headers=("Quantity", "Value", "Units"), widths=(42, 90, 24))
        for row in (
            ("generated_utc", datetime.now(timezone.utc).isoformat(), ""),
            ("software", SOFTWARE_NAME, ""), ("software_version", SOFTWARE_VERSION, ""),
            ("repository", REPOSITORY_URL, ""), ("calculation", envelope["calculation"], ""),
            ("isotope_convention", result["isotope_convention"], ""),
            ("constraint_convention", result["constraint_convention"], ""),
            ("interval_convention", result["interval_convention"], ""),
        ):
            summary.append(row)
        for isotope, values in result["isotopes"].items():
            for statistic, value in values.items():
                if statistic == "interval95":
                    if value is not None:
                        summary.append((f"{isotope}: 2.5 percentile", value[0], "per mil"))
                        summary.append((f"{isotope}: 97.5 percentile", value[1], "per mil"))
                else:
                    summary.append((f"{isotope}: {statistic}", value, "per mil"))
        summary.finish()
        inputs = _StreamingTable(workbook, "Input constraints", title="Independent input constraints",
                                 headers=("Parameter", "Constraint", "Center", "1 sigma", "Lower", "Upper", "Units"),
                                 widths=(20, 20, 18, 18, 18, 18, 20))
        for key, coordinate in (("pco2_constraint", "pCO2"), ("gpp_constraint", "GPP"), ("po2_constraint", "pO2")):
            c = result["inputs"][key]
            inputs.append((coordinate, c["kind"], c["center"], c["sigma"], c["lower"], c["upper"], _unit(coordinate)))
        inputs.finish()
        metadata = _StreamingTable(workbook, "Metadata", title="Reproducibility metadata",
                                   headers=("Property", "Value"), widths=(40, 110))
        for key, value in {
            "effective_bounds": result["effective_bounds"],
            "numerical_checks": result["numerical_checks"],
            "publication_model_id": envelope["publication_model_id"],
            "provenance": envelope["provenance"],
        }.items():
            metadata.append((key, json.dumps(value, ensure_ascii=True)))
        metadata.finish()
        buffer = BytesIO()
        workbook.save(buffer)
        return buffer.getvalue()


def _unit(coordinate: str) -> str:
    return {"pCO2": "ppm", "GPP": "PgC yr-1", "pO2": "PAL"}[coordinate]


def _prepare_sheet(
    sheet: Worksheet,
    *,
    title: str,
    headers: tuple[str, ...],
) -> None:
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    title_cell = sheet.cell(1, 1, title)
    title_cell.fill = TITLE_FILL
    title_cell.font = TITLE_FONT
    title_cell.alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 24
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(3, column, header)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    sheet.freeze_panes = "A4"


def _finish_table(sheet: Worksheet, widths: tuple[float, ...]) -> None:
    sheet.auto_filter.ref = f"A3:{sheet.cell(3, len(widths)).column_letter}{sheet.max_row}"
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=4):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def _append_summary_rows(
    sheet: _StreamingTable,
    envelope: dict[str, Any],
    context: dict[str, Any],
) -> None:
    result = envelope["result"]
    inputs = result["inputs"]
    low, high = result["equal_tailed_credible_interval"]
    coordinate = result["solve_for"]
    unit = _unit(coordinate)
    rows: list[tuple[str, Any, str]] = [
        ("generated_utc", datetime.now(timezone.utc).isoformat(), ""),
        ("software", SOFTWARE_NAME, ""),
        ("software_version", SOFTWARE_VERSION, ""),
        ("repository", REPOSITORY_URL, ""),
        ("calculation", envelope["calculation"], ""),
        ("isotope_source", context["isotope_source"], ""),
        (
            "target_air_Delta_prime_17O_0.528",
            inputs["target_air_cap_delta17_permil"],
            "permil",
        ),
        (
            "Delta_prime_17O_analytical_sigma",
            inputs["measurement_sigma_permil"],
            "permil (1 sigma)",
        ),
        (
            "target_air_delta18O_VSMOW",
            inputs.get("target_air_delta18_conventional_permil"),
            "permil",
        ),
        (
            "delta18O_analytical_sigma",
            inputs.get("delta18_measurement_sigma_permil"),
            "permil (1 sigma)",
        ),
        ("solved_coordinate", coordinate, ""),
        ("posterior_median", result["posterior_median"], unit),
        ("credible_interval_lower", low, unit),
        ("credible_interval_upper", high, unit),
        ("credible_interval_mass", inputs["credible_mass"], "probability"),
        ("boundary_sensitive", result["solve_boundary_sensitive"], ""),
        ("boundary_direction", result.get("solve_boundary_direction"), ""),
        (
            "boundary_probability_mass",
            result.get("solve_boundary_probability_mass"),
            "probability",
        ),
        ("posterior_mode_at_boundary", result.get("solve_mode_at_boundary"), ""),
        ("probability_scope", result["probability_scope"], ""),
    ]
    sulfate = inputs.get("sulfate")
    if sulfate is not None:
        air_fields = {"target_air_Delta_prime_17O_0.528", "Delta_prime_17O_analytical_sigma",
                      "target_air_delta18O_VSMOW", "delta18O_analytical_sigma"}
        rows = [row for row in rows if row[0] not in air_fields]
        rows.extend([
            ("sulfate_Delta_prime_17O_0.528", sulfate["measured_cap_delta17_permil"], "permil"),
            ("sulfate_Delta_prime_17O_analytical_sigma", sulfate["cap_delta17_sigma_permil"], "permil (1 sigma)"),
            ("sulfate_delta18O_VSMOW", sulfate["measured_delta18_permil"], "permil"),
            ("sulfate_delta18O_analytical_sigma", sulfate["delta18_sigma_permil"], "permil (1 sigma)"),
            ("primary_sulfate_interpretation_assumed", True, "conditional on stated process assumptions"),
        ])
    spherule = context.get("spherule")
    if spherule:
        rows.extend(
            [
                (
                    "spherule_Delta_prime_17O_0.528",
                    spherule["cap_delta17_permil"],
                    "permil",
                ),
                (
                    "spherule_Delta_prime_17O_analytical_sigma",
                    spherule["cap_delta17_sigma_permil"],
                    "permil (1 sigma)",
                ),
                ("spherule_delta18O_VSMOW", spherule["delta18_permil"], "permil"),
                (
                    "spherule_delta18O_analytical_sigma",
                    spherule["delta18_sigma_permil"],
                    "permil (1 sigma)",
                ),
            ]
        )
    for name, constraint in inputs["constraints"].items():
        rows.append((f"{name}_constraint_kind", constraint["kind"], ""))
        for key in ("center", "sigma", "lower", "upper"):
            value = constraint.get(key)
            if value is not None:
                rows.append((f"{name}_constraint_{key}", value, _unit(name)))
        effective = result["effective_constraint_bounds"].get(name)
        if effective:
            rows.extend(
                [
                    (f"{name}_effective_lower", effective[0], _unit(name)),
                    (f"{name}_effective_upper", effective[1], _unit(name)),
                ]
            )
    for key, value in _flatten_mapping({"coordinate_integration": result.get("coordinate_integration_diagnostics"),
                                      "field_refinement": result.get("field_refinement_diagnostics")}):
        if value is not None:
            rows.append((key, _cell_value(value), "numerical integration"))
    if result.get("field_probability_mass") is not None:
        rows.extend([
            ("field_hpd_density_threshold", result.get("field_hpd_density_threshold"), "reported coordinate units"),
            ("field_hpd_probability_mass", result.get("field_hpd_probability_mass"), "probability"),
            ("joint_probability_storage", "Zero-density, zero-mass grid nodes are omitted except axis anchors; unlisted combinations are zero.", "Both complete axes are retained"),
        ])
    for row in rows:
        sheet.append(row)


def build_coordinate_inference_workbook(
    envelope: dict[str, Any],
    context: dict[str, Any],
) -> bytes:
    """Return a self-contained XLSX export for one constrained inference."""

    with _streaming_workbook() as workbook:
        return _write_coordinate_inference_workbook(workbook, envelope, context)


def _write_coordinate_inference_workbook(
    workbook: Workbook, envelope: dict[str, Any], context: dict[str, Any],
) -> bytes:
    result = envelope["result"]
    workbook.properties.creator = "OXYTIB"
    workbook.properties.title = "Constrained atmospheric O2 isotope inference"
    workbook.properties.subject = f"{SOFTWARE_NAME} {SOFTWARE_VERSION}"

    summary = _StreamingTable(
        workbook, "Summary", title="Constrained model solution",
        headers=("Field", "Value", "Unit or note"), widths=(42, 70, 24),
    )
    _append_summary_rows(summary, envelope, context)
    summary.finish()

    if result["inputs"].get("sulfate") is not None:
        process = _StreamingTable(
            workbook, "Sulfate transfer", title="Conditional sulfate transfer",
            headers=("Field", "Value"), widths=(60, 95),
        )
        metadata = {
            "observation_and_process": result["inputs"]["sulfate"],
            "isotope_coordinate": "logarithmic Delta-prime-17O, slope 0.528; conventional delta18O, VSMOW; per mil",
            "incorporation_units": "fraction of all oxygen atoms, 0-1; after formation-stage exchange",
            "background_units": "effective non-air Delta-prime-17O, per mil; after formation",
            "likelihood_diagnostics": result.get("sulfate_likelihood_diagnostics"),
        }
        # Observation inputs are already tabulated above; keep diagnostics distinct.
        if metadata["likelihood_diagnostics"] is not None:
            metadata["likelihood_diagnostics"] = {
                key: value for key, value in metadata["likelihood_diagnostics"].items() if key != "observation"
            }
        for key, value in _flatten_mapping(metadata):
            process.append((key, _cell_value(value)))
        process.finish()

    coordinate = result["solve_for"]
    posterior = _StreamingTable(
        workbook, "Posterior",
        title=f"{coordinate} marginal posterior",
        headers=(coordinate, "Unit", "Probability mass", "Probability density"),
        widths=(22, 16, 22, 22),
        number_formats={1: "0.0000000000", 3: "0.0000000000E+00", 4: "0.0000000000E+00"},
    )
    for axis, mass, density in zip(
        result["solve_axis"],
        result["solve_marginal_probability_mass"],
        result["solve_marginal_density"],
        strict=True,
    ):
        posterior.append((axis, _unit(coordinate), mass, density))
    posterior.finish()

    if result.get("field_probability_mass") is not None:
        x_name = result["field_x_coordinate"]
        y_name = result["field_y_coordinate"]
        field = _StreamingTable(
            workbook, "Joint probability",
            title=f"{x_name}-{y_name} joint probability field",
            headers=(
                x_name,
                f"{x_name} unit",
                y_name,
                f"{y_name} unit",
                "Probability mass",
                "Probability density",
                "Inside 95% HPD",
            ),
            widths=(20, 16, 20, 16, 22, 22, 18),
            number_formats={1: "0.0000000000", 3: "0.0000000000",
                            5: "0.0000000000E+00", 6: "0.0000000000E+00"},
        )
        y_axis = result["field_y_axis"]
        for x_index, x_value in enumerate(result["field_x_axis"]):
            for y_index, y_value in enumerate(y_axis):
                flat_index = x_index * len(y_axis) + y_index
                if (x_index > 0 and y_index > 0
                        and result["field_probability_mass"][flat_index] == 0.0
                        and result["field_density"][flat_index] == 0.0):
                    continue
                field.append(
                    (
                        x_value,
                        _unit(x_name),
                        y_value,
                        _unit(y_name),
                        result["field_probability_mass"][flat_index],
                        result["field_density"][flat_index],
                        result["field_hpd_mask"][flat_index],
                    )
                )
        field.finish()

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _cell_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_mapping(child, path))
        return rows
    rows.append((prefix, _cell_value(value)))
    return rows


def build_transient_workbook(
    envelope: dict[str, Any],
    experiment_type: str,
) -> bytes:
    """Return an XLSX record of a public time-response experiment."""

    result = envelope["result"]
    request = result["request"]
    equilibrium = result.get("operational_equilibrium", {})
    workbook = Workbook()
    workbook.properties.creator = "OXYTIB"
    workbook.properties.title = "Atmospheric O2 isotope time-response experiment"
    workbook.properties.subject = f"{SOFTWARE_NAME} {SOFTWARE_VERSION}"

    summary = workbook.active
    summary.title = "Summary"
    _prepare_sheet(
        summary,
        title="Atmospheric isotope time response",
        headers=("Field", "Value", "Unit or note"),
    )
    summary_rows = [
        ("generated_utc", datetime.now(timezone.utc).isoformat(), ""),
        ("software", SOFTWARE_NAME, ""),
        ("software_version", SOFTWARE_VERSION, ""),
        ("repository", REPOSITORY_URL, ""),
        ("calculation", envelope["calculation"], ""),
        ("experiment_type", experiment_type, ""),
        ("display_duration", request["duration_years"], "years"),
        ("sample_count", request["sample_count"], ""),
        (
            "equilibrium_search_horizon",
            request["equilibrium_search_max_years"],
            "years",
        ),
        (
            "operational_equilibrium_time",
            equilibrium.get("time_years"),
            "years",
        ),
        (
            "operational_equilibrium_tolerance",
            equilibrium.get("tolerance_permil"),
            "permil",
        ),
    ]
    if experiment_type == "pCO2_trajectory":
        transition = result["transition_end_state"]
        summary_rows.extend(
            (
                ("transition_end_time", transition["time_years"], "years"),
                ("transition_end_pCO2", transition["pco2_ppm"], "ppm"),
                (
                    "transition_end_O2_Delta_prime_17O_0.528",
                    transition["cap_delta17_prime_permil"],
                    "permil",
                ),
                (
                    "transition_end_O2_delta_prime_18O",
                    transition["delta18_prime_permil"],
                    "permil",
                ),
            )
        )
    for row in summary_rows:
        summary.append(row)
    _finish_table(summary, (42, 72, 24))

    inputs = workbook.create_sheet("Inputs")
    _prepare_sheet(
        inputs,
        title="Experiment inputs",
        headers=("Parameter", "Value"),
    )
    for path, value in _flatten_mapping(request):
        inputs.append((path, value))
    _finish_table(inputs, (48, 72))

    timeseries = workbook.create_sheet("Time series")
    headers = (
        "time_years",
        "pO2_PAL",
        "pCO2_ppm",
        "GPP_PgC_per_year",
        "photosynthesis_fraction_of_initial",
        "carbon_driver_pO2_PAL",
        "O2_Delta_prime_17O_0.528_permil",
        "O2_delta_prime_18O_permil",
        "O16O16_mol",
        "O16O17_mol",
        "O16O18_mol",
    )
    _prepare_sheet(timeseries, title="Model time series", headers=headers)
    states = result["states"]
    is_photosynthesis = experiment_type == "photosynthesis"
    is_trajectory = experiment_type == "pCO2_trajectory"
    if is_photosynthesis:
        pco2_values = result["pco2_ppm"]
        carbon_po2_values = result["carbon_driver_po2_pal"]
        gpp_value = request["initial"]["gpp_pgC_per_year"]
        photosynthesis_fraction = request["photosynthesis_fraction"]
    elif is_trajectory:
        pco2_values = result["pco2_ppm"]
        carbon_po2_values = [None] * len(states)
        gpp_value = request["initial"]["gpp_pgC_per_year"]
        photosynthesis_fraction = None
    else:
        pco2_values = [request["final"]["p_co2_ppm"]] * len(states)
        carbon_po2_values = [None] * len(states)
        gpp_value = request["final"]["gpp_pgC_per_year"]
        photosynthesis_fraction = None
    for index, (time, state) in enumerate(
        zip(result["time_years"], states, strict=True)
    ):
        timeseries.append(
            (
                time,
                state["o16o16_mol"] / GLOBAL_MAJOR_O2_MOLES_1PAL,
                pco2_values[index],
                gpp_value,
                photosynthesis_fraction,
                carbon_po2_values[index],
                state["cap_delta17_prime_permil"],
                state["delta18_prime_permil"],
                state["o16o16_mol"],
                state["o16o17_mol"],
                state["o16o18_mol"],
            )
        )
    for row in timeseries.iter_rows(min_row=4, max_col=len(headers)):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.0000000000E+00"
    _finish_table(
        timeseries,
        (16, 16, 18, 22, 28, 22, 30, 27, 22, 22, 22),
    )

    provenance = workbook.create_sheet("Provenance")
    _prepare_sheet(
        provenance,
        title="Model and solver provenance",
        headers=("Field", "Value"),
    )
    provenance_values = {
        "software": SOFTWARE_NAME,
        "software_version": SOFTWARE_VERSION,
        "repository": REPOSITORY_URL,
        "calculation": envelope["calculation"],
        "solver": result.get("solver", {}),
        "operational_equilibrium": equilibrium,
    }
    for path, value in _flatten_mapping(provenance_values):
        provenance.append((path, value))
    _finish_table(provenance, (52, 88))

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
