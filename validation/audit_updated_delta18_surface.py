"""Audit accelerated delta-prime-18O in every refined-grid cell."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
)
from updated_output_surface import (  # noqa: E402
    UpdatedMolecularOutputSurface,
    UpdatedOutputSurfaceInput,
)


ACCEPTANCE_PERMIL = 0.050


def _fractional_axis_points(values: np.ndarray, *, phase: int) -> np.ndarray:
    fractions = np.asarray((0.2763932023, 0.4384471872, 0.5615528128, 0.7236067977))
    lower = np.log(values[:-1])
    upper = np.log(values[1:])
    fraction = fractions[(np.arange(len(lower)) + phase) % len(fractions)]
    return np.exp(lower + fraction * (upper - lower))


def run(*, surface_path: Path, report_path: Path) -> dict[str, object]:
    bundle = json.loads(surface_path.read_text(encoding="utf-8"))
    surface = UpdatedMolecularOutputSurface(bundle)
    delta18_surface = bundle.get("delta18_surface")
    if delta18_surface is None:
        raise ValueError("surface has no dedicated delta-prime-18O grid")
    axes = delta18_surface["axes"]
    po2_values = _fractional_axis_points(
        np.asarray(axes["po2_pal"], dtype=float), phase=1
    )
    pco2_values = _fractional_axis_points(
        np.asarray(axes["pco2_ppm"], dtype=float), phase=2
    )
    gpp_values = _fractional_axis_points(
        np.asarray(axes["gpp_pgC_per_year"], dtype=float), phase=3
    )
    case_count = len(po2_values) * len(pco2_values) * len(gpp_values)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    case_index = 0
    maximum_fixed_point_residual = 0.0
    for po2 in po2_values:
        for pco2 in pco2_values:
            for gpp in gpp_values:
                request = UpdatedForwardInput(float(po2), float(pco2), float(gpp))
                live = run_updated_central_state(request)
                accelerated = surface.evaluate(
                    UpdatedOutputSurfaceInput(float(po2), float(pco2), float(gpp))
                )
                residual = (
                    accelerated.central_delta18_prime_permil
                    - live.delta18_prime_permil
                )
                maximum_fixed_point_residual = max(
                    maximum_fixed_point_residual,
                    live.maximum_fixed_point_residual_permil,
                )
                rows.append(
                    {
                        "pO2_PAL": float(po2),
                        "pCO2_ppm": float(pco2),
                        "GPP_PgC_per_year": float(gpp),
                        "live_delta18_prime_permil": live.delta18_prime_permil,
                        "accelerated_delta18_prime_permil": (
                            accelerated.central_delta18_prime_permil
                        ),
                        "residual_delta18_prime_permil": residual,
                    }
                )
                case_index += 1
        print(
            f"validated pO2={po2:.7g} PAL ({case_index}/{case_count})",
            flush=True,
        )
    live_seconds = time.perf_counter() - started
    residuals = np.asarray(
        [row["residual_delta18_prime_permil"] for row in rows], dtype=float
    )
    absolute = np.abs(residuals)
    maximum_index = int(np.argmax(absolute))
    report = {
        "audit": "updated molecular delta-prime-18O every-cell noncentral holdouts",
        "status": "accepted" if float(np.max(absolute)) <= ACCEPTANCE_PERMIL else "failed",
        "surface_data_id": surface.surface_data_id,
        "upstream_model_data_id": surface.upstream_model_data_id,
        "holdout_design": (
            "One deterministic noncentral logarithmic point in every cell of "
            "the dedicated refined delta-prime-18O grid; fractions differ "
            "from the accelerator method-selection audit."
        ),
        "case_count": case_count,
        "acceptance_threshold_permil": ACCEPTANCE_PERMIL,
        "mean_absolute_residual_permil": float(np.mean(absolute)),
        "p95_absolute_residual_permil": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_residual_permil": float(np.max(absolute)),
        "maximum_residual_case": rows[maximum_index],
        "maximum_live_kernel_fixed_point_residual_permil": (
            maximum_fixed_point_residual
        ),
        "live_kernel_seconds": live_seconds,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with report_path.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def promote(
    *,
    surface_path: Path,
    report_path: Path,
    validated_surface_path: Path,
) -> dict[str, object]:
    """Attach an accepted every-cell audit to a versioned runtime surface."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "accepted":
        raise ValueError("cannot promote a failed delta-prime-18O audit")
    maximum = float(report["maximum_absolute_residual_permil"])
    threshold = float(report["acceptance_threshold_permil"])
    if maximum > threshold:
        raise ValueError("delta-prime-18O report contradicts accepted status")
    bundle = json.loads(surface_path.read_text(encoding="utf-8"))
    if "delta18_surface" not in bundle:
        raise ValueError("surface has no dedicated delta-prime-18O grid")
    relative_report = str(report_path.resolve().relative_to(ROOT)).replace("\\", "/")
    validation = bundle["validation"]
    validation["delta18_acceleration_validated"] = True
    validation["central_delta18_maximum_absolute_residual_permil"] = maximum
    validation["delta18_every_cell_holdout_case_count"] = int(report["case_count"])
    validation["delta18_every_cell_report"] = relative_report
    bundle["delta18_surface"]["validation"] = {
        "status": "accepted_every_cell_noncentral_holdouts",
        "holdout_case_count": int(report["case_count"]),
        "holdout_design": report["holdout_design"],
        "mean_absolute_residual_permil": float(
            report["mean_absolute_residual_permil"]
        ),
        "p95_absolute_residual_permil": float(
            report["p95_absolute_residual_permil"]
        ),
        "maximum_absolute_residual_permil": maximum,
        "acceptance_threshold_permil": threshold,
        "report": relative_report,
    }
    validated_surface_path.parent.mkdir(parents=True, exist_ok=True)
    validated_surface_path.write_text(
        json.dumps(bundle, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validated-surface", type=Path)
    parser.add_argument("--reuse-report", action="store_true")
    args = parser.parse_args()
    report = (
        json.loads(args.report.read_text(encoding="utf-8"))
        if args.reuse_report
        else run(surface_path=args.surface, report_path=args.report)
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "accepted":
        raise SystemExit(1)
    if args.validated_surface is not None:
        promote(
            surface_path=args.surface,
            report_path=args.report,
            validated_surface_path=args.validated_surface,
        )


if __name__ == "__main__":
    main()
