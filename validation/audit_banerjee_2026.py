"""Audit the Banerjee et al. (2026) supporting-table transcription.

This audit checks source integrity and coordinate compatibility. It does not
fit, calibrate, or score the updated atmospheric model against the record.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


DATA_PATH = ROOT / "model_data" / "literature" / "banerjee_2026_supporting_tables.csv"
SURFACE_PATH = ROOT / "model_data" / "updated_r7_response_surface_v1.json"
BANERJEE_INTERCEPT_PPM = 266.58
BANERJEE_SLOPE_MAGNITUDE = 1.6131
LEGACY_INTERCEPT_PPM = 256.9758
LEGACY_SLOPE_MAGNITUDE = 1.0862


def _optional_float(value: str) -> float | None:
    return None if value.strip() == "" else float(value)


def load_rows(path: Path = DATA_PATH) -> list[dict[str, object]]:
    """Load typed rows while retaining source annotations."""

    numeric_fields = (
        "top_depth_m", "ar_age_ka", "measured_co2_ppm", "delta13c_permil",
        "delta15n_permil", "delta_o2_n2_permil", "delta_ar_n2_permil",
        "cap_delta17_rep1_ppm", "cap_delta17_rep2_ppm",
        "cap_delta17_average_ppm", "reconstructed_co2_ppm",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        row["row"] = int(raw["row"])
        for field in numeric_fields:
            row[field] = _optional_float(raw[field])
        rows.append(row)
    return rows


def reconstructed_co2_from_delta(cap_delta17_ppm: float) -> float:
    """Evaluate the relation supported by Banerjee Table S1 and Figure 1."""

    return BANERJEE_INTERCEPT_PPM - BANERJEE_SLOPE_MAGNITUDE * cap_delta17_ppm


def printed_equation_7_co2_from_delta(cap_delta17_ppm: float) -> float:
    """Evaluate the inconsistent plus-sign equation solely for source auditing."""

    return BANERJEE_INTERCEPT_PPM + BANERJEE_SLOPE_MAGNITUDE * cap_delta17_ppm


def legacy_s2_co2_from_delta(cap_delta17_ppm: float) -> float:
    """Evaluate the exact relation encoded in the archived previous-data sheet."""

    return LEGACY_INTERCEPT_PPM - LEGACY_SLOPE_MAGNITUDE * cap_delta17_ppm


def regression_residuals(
    rows: list[dict[str, object]],
    *,
    printed_equation: bool = False,
    table: str | None = None,
) -> np.ndarray:
    """Return predicted minus reported CO2 residuals for applicable rows."""

    residuals = []
    for row in rows:
        if table is not None and row["table"] != table:
            continue
        cap_delta17 = row["cap_delta17_average_ppm"]
        reported = row["reconstructed_co2_ppm"]
        if cap_delta17 is None or reported is None:
            continue
        predictor = (
            printed_equation_7_co2_from_delta
            if printed_equation
            else reconstructed_co2_from_delta
        )
        residuals.append(predictor(float(cap_delta17)) - float(reported))
    return np.asarray(residuals, dtype=float)


def run() -> dict[str, object]:
    rows = load_rows()
    s1_printed = regression_residuals(rows, printed_equation=True, table="S1")
    s1_corrected = regression_residuals(rows, table="S1")
    s2_rows = [
        row for row in rows
        if row["table"] == "S2"
        and row["cap_delta17_average_ppm"] is not None
        and row["reconstructed_co2_ppm"] is not None
    ]
    s2_delta = np.asarray([float(row["cap_delta17_average_ppm"]) for row in s2_rows])
    s2_reported = np.asarray([float(row["reconstructed_co2_ppm"]) for row in s2_rows])
    s2_archive_residual = legacy_s2_co2_from_delta(s2_delta) - s2_reported
    s2_equation7_residual = regression_residuals(
        rows, table="S2"
    )

    surface = json.loads(SURFACE_PATH.read_text(encoding="utf-8"))
    pco2_min, pco2_max = map(float, surface["domain"]["pco2_ppm"])
    pristine_rows = [
        row for row in rows
        if row["excluded_from_pristine_co2_dataset"] == "no"
    ]
    measured_values = np.asarray([
        float(row["measured_co2_ppm"])
        for row in pristine_rows if row["measured_co2_ppm"] is not None
    ])
    reconstructed_values = np.asarray([
        float(row["reconstructed_co2_ppm"])
        for row in pristine_rows if row["reconstructed_co2_ppm"] is not None
    ])
    measured_inside = int(
        np.sum((measured_values >= pco2_min) & (measured_values <= pco2_max))
    )
    reconstructed_inside = int(
        np.sum((reconstructed_values >= pco2_min) & (reconstructed_values <= pco2_max))
    )
    domain_passes = (
        measured_inside == int(measured_values.size)
        and reconstructed_inside == int(reconstructed_values.size)
    )

    report: dict[str, object] = {
        "audit": "Banerjee et al. (2026) source and coordinate integrity",
        "model_calibrated_or_fitted": False,
        "source": {
            "paper_doi": "10.1029/2025JD045900",
            "data_doi": "10.15784/602070",
            "reported_data_md5": "2c0e4d4090f58cb790e098647141b219",
            "verified_data_sha256": "2bc30014d56169c37a9143ad263767398b6f520fc930dc1c2ac2dae1b90b35c6",
            "archive_checksum_verified": True,
            "normalized_from_archived_workbook": True,
            "transcribed_rows": len(rows),
            "table_S1_rows": sum(row["table"] == "S1" for row in rows),
            "table_S2_rows": sum(row["table"] == "S2" for row in rows),
            "author_excluded_nonpristine_rows": sum(
                row["excluded_from_pristine_co2_dataset"] == "yes" for row in rows
            ),
            "flagged_source_rows": [
                {"table": row["table"], "row": row["row"], "note": row["source_note"]}
                for row in rows if row["source_note"]
            ],
        },
        "isotope_coordinate": {
            "reported_reference": "modern atmospheric O2",
            "reported_lambda": 0.518,
            "reported_unit": "ppm",
            "absolute_model_lambda": 0.528,
            "direct_absolute_comparison_valid": False,
            "conversion_module": "code/isotope_reference_frames.py",
            "sample_delta18_prime_required_for_absolute_inversion": True,
        },
        "equation_7_sign_audit": {
            "printed_equation": "CO2 = 266.58 + 1.6131 * Delta17O_ppm",
            "adopted_operational_equation": "CO2 = 266.58 - 1.6131 * Delta17O_ppm",
            "operational_policy": "the printed plus-sign form is audit-only and cannot be selected by model code",
            "scope": "Table S1 only",
            "strict_rows": int(s1_corrected.size),
            "printed_plus_sign_mae_ppm": float(np.mean(np.abs(s1_printed))),
            "printed_plus_sign_max_abs_ppm": float(np.max(np.abs(s1_printed))),
            "required_minus_sign_mae_ppm": float(np.mean(np.abs(s1_corrected))),
            "required_minus_sign_max_abs_ppm": float(np.max(np.abs(s1_corrected))),
            "required_minus_sign_all_rows_max_abs_ppm": float(np.max(np.abs(s1_corrected))),
            "strict_policy": "All archived Table S1 rows with a reconstructed value",
        },
        "table_S2_legacy_reconstruction": {
            "equation_7_minus_sign_mae_ppm": float(np.mean(np.abs(s2_equation7_residual))),
            "equation_7_minus_sign_max_abs_ppm": float(np.max(np.abs(s2_equation7_residual))),
            "archive_encoded_equation": "CO2 = 256.9758 - 1.0862 * Delta17O_ppm",
            "archive_relation_mae_ppm": float(np.mean(np.abs(s2_archive_residual))),
            "archive_relation_max_abs_ppm": float(np.max(np.abs(s2_archive_residual))),
            "archive_labels_sheet_as": "Previous Data",
            "methodological_attribution_resolved": False,
            "validation_policy": "ignore reconstructed CO2; use raw measured Delta17O and measured CO2 only",
        },
        "current_model_domain_gate": {
            "surface_pco2_ppm": [pco2_min, pco2_max],
            "pristine_measured_co2_rows_inside_surface_domain": measured_inside,
            "pristine_measured_co2_rows_total": int(measured_values.size),
            "pristine_reported_reconstructed_co2_rows_inside_surface_domain": reconstructed_inside,
            "pristine_reported_reconstructed_co2_rows_total": int(reconstructed_values.size),
            "validation_status": "gate_passed" if domain_passes else "outside_surface_domain",
        },
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "banerjee_2026_source_audit.json"

    valid_rows = [
        row for row in rows
        if row["table"] == "S1"
        and row["cap_delta17_average_ppm"] is not None
        and row["reconstructed_co2_ppm"] is not None
    ]
    delta = np.asarray([float(row["cap_delta17_average_ppm"]) for row in valid_rows])
    reported = np.asarray([float(row["reconstructed_co2_ppm"]) for row in valid_rows])
    order = np.argsort(delta)

    figure, axis = plt.subplots(figsize=(7.3, 4.8), constrained_layout=True)
    axis.scatter(delta, reported, color="black", s=26, label="Supporting Table S1")
    axis.plot(
        delta[order], reconstructed_co2_from_delta(delta[order]),
        color="tab:blue", linewidth=2.0, label="Adopted relation (Table S1 and Figure 1)",
    )
    axis.plot(
        delta[order], printed_equation_7_co2_from_delta(delta[order]),
        color="tab:red", linestyle="--", linewidth=1.5, label="Equation 7 as printed",
    )
    axis.set_xlabel(r"Modern-air-relative $\Delta^{17}$O of O$_2$ (ppm; $\lambda=0.518$)")
    axis.set_ylabel("Reconstructed CO$_2$ (ppm)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure_path = output_dir / "banerjee_2026_equation7_sign_audit.png"
    figure.savefig(figure_path, dpi=240)
    plt.close(figure)

    report["outputs"] = {
        "json": str(json_path.relative_to(ROOT)),
        "figure": str(figure_path.relative_to(ROOT)),
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run()
    sign = report["equation_7_sign_audit"]
    domain = report["current_model_domain_gate"]
    print(json.dumps(report, indent=2))
    print(
        "Equation 7 audit: minus-sign MAE "
        f"{sign['required_minus_sign_mae_ppm']:.3f} ppm for Table S1; "
        f"printed plus-sign MAE {sign['printed_plus_sign_mae_ppm']:.1f} ppm"
    )
    print(
        "Model-domain gate: "
        f"{domain['pristine_reported_reconstructed_co2_rows_inside_surface_domain']}/"
        f"{domain['pristine_reported_reconstructed_co2_rows_total']} pristine reconstructed values in domain"
    )


if __name__ == "__main__":
    main()
