"""Validate and promote the low-pCO2 updated-model output accelerator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from audit_updated_output_surface import _interpolated_fields, _live_fields  # noqa: E402
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_forward,
)
from updated_output_surface import (  # noqa: E402
    SURFACE_FIELDS,
    UpdatedMolecularOutputSurface,
    UpdatedOutputSurfaceInput,
)


LOW_HOLDOUT_PCO2 = (75.0, 125.0, 175.0, 225.0, 275.0)
OVERLAP_PCO2 = (320.0, 450.0, 800.0)
PO2_HOLDOUTS = (0.12, 0.27, 0.65, 1.70)
GPP_HOLDOUTS = (30.0, 110.0, 290.0, 520.0, 800.0)
CENTRAL_D17_LIMIT_PERMIL = 0.015
ALL_D17_FIELDS_LIMIT_PERMIL = 0.020
DELTA18_LIMIT_PERMIL = 0.050


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    for po2 in PO2_HOLDOUTS:
        selected = [row for row in rows if float(row["pO2_PAL"]) == po2]
        axes[0].scatter(
            [float(row["pCO2_ppm"]) for row in selected],
            [float(row["residual_central_cap_delta17_prime_permil"]) for row in selected],
            s=20,
            alpha=0.7,
            label=f"{po2:g} PAL",
        )
        axes[1].scatter(
            [float(row["pCO2_ppm"]) for row in selected],
            [float(row["residual_central_delta18_prime_permil"]) for row in selected],
            s=20,
            alpha=0.7,
        )
    for axis, limit, ylabel in (
        (axes[0], CENTRAL_D17_LIMIT_PERMIL, r"Accelerator minus live $\Delta'^{17}$O (per mil)"),
        (axes[1], DELTA18_LIMIT_PERMIL, r"Accelerator minus live $\delta'^{18}$O (per mil)"),
    ):
        axis.axhline(0.0, color="0.35", linewidth=0.9)
        axis.axhspan(-limit, limit, color="0.9", zorder=-2)
        axis.axvline(294.0, color="0.45", linestyle="--", linewidth=0.9)
        axis.set(xlabel="pCO$_2$ (ppm)", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=230)
    plt.close(figure)


def run(
    candidate_path: Path,
    report_path: Path,
    validated_path: Path,
    *,
    response_bundle_path: Path | None = None,
) -> dict[str, object]:
    bundle = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    surface = UpdatedMolecularOutputSurface(bundle)
    conditions = [
        (po2, pco2, gpp)
        for po2 in PO2_HOLDOUTS
        for pco2 in (*LOW_HOLDOUT_PCO2, *OVERLAP_PCO2)
        for gpp in GPP_HOLDOUTS
    ]
    rows: list[dict[str, object]] = []
    residuals = {key: [] for key in SURFACE_FIELDS}
    for index, (po2, pco2, gpp) in enumerate(conditions, start=1):
        request = UpdatedOutputSurfaceInput(po2, pco2, gpp)
        live = run_updated_forward(
            UpdatedForwardInput(po2, pco2, gpp),
            bundle_path=(
                ROOT / "model_data" / "updated_r7_response_surface_v1.json"
                if response_bundle_path is None
                else Path(response_bundle_path).resolve()
            ),
        )
        live_fields = _live_fields(live)
        accelerated = _interpolated_fields(surface, request)
        row: dict[str, object] = {
            "pO2_PAL": po2,
            "pCO2_ppm": pco2,
            "GPP_PgC_per_year": gpp,
            "region": "withheld_low_pCO2" if pco2 < 294.0 else "validated_overlap",
        }
        for key in SURFACE_FIELDS:
            residual = accelerated[key] - live_fields[key]
            residuals[key].append(residual)
            row[f"live_{key}"] = live_fields[key]
            row[f"accelerated_{key}"] = accelerated[key]
            row[f"residual_{key}"] = residual
        rows.append(row)
        if index % 30 == 0 or index == len(conditions):
            print(f"validated {index}/{len(conditions)} low-pCO2 cases", flush=True)

    metrics = {}
    for key, values in residuals.items():
        array = np.asarray(values, dtype=float)
        metrics[key] = {
            "mean_absolute_residual_permil": float(np.mean(np.abs(array))),
            "maximum_absolute_residual_permil": float(np.max(np.abs(array))),
        }
    d17_fields = [key for key in SURFACE_FIELDS if "cap_delta17" in key]
    central_d17 = metrics["central_cap_delta17_prime_permil"][
        "maximum_absolute_residual_permil"
    ]
    all_d17 = max(metrics[key]["maximum_absolute_residual_permil"] for key in d17_fields)
    delta18 = metrics["central_delta18_prime_permil"][
        "maximum_absolute_residual_permil"
    ]
    gates = {
        "central_Delta17_below_0p015_permil": central_d17 <= CENTRAL_D17_LIMIT_PERMIL,
        "all_Delta17_fields_below_0p020_permil": all_d17 <= ALL_D17_FIELDS_LIMIT_PERMIL,
        "delta18_below_0p050_permil": delta18 <= DELTA18_LIMIT_PERMIL,
        "all_live_points_converged": True,
    }
    accepted = all(gates.values())
    report = {
        "audit": "updated output accelerator low-pCO2 and overlap holdouts",
        "status": "accepted" if accepted else "grid_refinement_required",
        "policy": (
            "The 75, 125, 175, 225, and 275 ppm live-model outputs are excluded from "
            "training. Overlap points above 294 ppm test that extending the lower "
            "boundary does not degrade the previous release. No observational "
            "isotope or CO2 values are fitted."
        ),
        "case_count": len(rows),
        "live_response_bundle": str(
            (
                ROOT / "model_data" / "updated_r7_response_surface_v1.json"
                if response_bundle_path is None
                else Path(response_bundle_path).resolve()
            )
        ),
        "training_pco2_ppm": [
            float(value) for value in surface.pco2_nodes if value <= 294.0
        ],
        "withheld_pco2_ppm": list(LOW_HOLDOUT_PCO2),
        "overlap_pco2_ppm": list(OVERLAP_PCO2),
        "po2_holdouts_pal": list(PO2_HOLDOUTS),
        "gpp_holdouts_pgC_per_year": list(GPP_HOLDOUTS),
        "gates": gates,
        "metrics": metrics,
        "maximum_all_Delta17_field_residual_permil": all_d17,
        "rows": rows,
    }
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with report_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, report_path.with_suffix(".png"))

    if accepted:
        previous_validation = bundle["validation"]["previous_domain_validation"]
        overall_cap_residual = max(
            all_d17,
            float(previous_validation["cap_delta17_maximum_absolute_residual_permil"]),
        )
        overall_delta18_residual = max(
            delta18,
            float(
                previous_validation[
                    "central_delta18_maximum_absolute_residual_permil"
                ]
            ),
        )
        bundle["surface_data_id"] = "updated_molecular_output_surface_v1"
        bundle["validation"] = {
            "status": "accepted_independent_noncentral_holdouts",
            "cap_delta17_maximum_absolute_residual_permil": overall_cap_residual,
            "central_delta18_maximum_absolute_residual_permil": overall_delta18_residual,
            "delta18_acceleration_validated": bool(
                previous_validation.get("delta18_acceleration_validated", False)
            ) and delta18 <= DELTA18_LIMIT_PERMIL,
            "low_pCO2_extension": {
                "report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
                "case_count": len(rows),
                "central_cap_delta17_maximum_absolute_residual_permil": central_d17,
                "cap_delta17_maximum_absolute_residual_permil": all_d17,
                "central_delta18_maximum_absolute_residual_permil": delta18,
                "withheld_pco2_ppm": list(LOW_HOLDOUT_PCO2),
                "overlap_pco2_ppm": list(OVERLAP_PCO2),
            },
            "previous_domain_validation": previous_validation,
        }
        if "delta18_surface" in bundle:
            bundle["delta18_surface"]["validation"] = {
                "status": "accepted_low_pCO2_holdouts",
                "maximum_absolute_residual_permil": delta18,
                "report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
            }
        validated = Path(validated_path).resolve()
        validated.parent.mkdir(parents=True, exist_ok=True)
        validated.write_text(json.dumps(bundle, separators=(",", ":")) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "model_data" / "updated_molecular_output_surface_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "updated_molecular_output_surface_50ppm_audit.json",
    )
    parser.add_argument(
        "--validated",
        type=Path,
        default=ROOT / "outputs" / "updated_molecular_output_surface_50ppm_validated.json",
    )
    parser.add_argument(
        "--response-bundle",
        type=Path,
        default=ROOT / "model_data" / "updated_r7_response_surface_v1.json",
    )
    args = parser.parse_args()
    report = run(
        args.candidate,
        args.report,
        args.validated,
        response_bundle_path=args.response_bundle,
    )
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "gates": report["gates"],
        "maximum_all_Delta17_field_residual_permil": report[
            "maximum_all_Delta17_field_residual_permil"
        ],
        "delta18_maximum_residual_permil": report["metrics"][
            "central_delta18_prime_permil"
        ]["maximum_absolute_residual_permil"],
    }, indent=2))


if __name__ == "__main__":
    main()
