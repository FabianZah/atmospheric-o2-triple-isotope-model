"""Compare the updated GPP inversion with Luz et al. (1999) Table 2."""

from __future__ import annotations

import argparse
import csv
import json
from math import log1p
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
from updated_output_surface_inverse import (  # noqa: E402
    UpdatedSurfaceInverseInput,
    invert_updated_output_surface,
)


SOURCE_PATH = ROOT / "model_data" / "literature" / "luz_1999_gisp2_v1.json"
OUTPUT_PATH = ROOT / "outputs" / "luz_1999_productivity_benchmark.json"
MODEL_LAMBDA = 0.528
LUZ_LAMBDA = 0.521
PO2_PAL = 1.0
REFERENCE_CO2_PPM = 280.0
REFERENCE_GPP_PGC_PER_YEAR = 290.0
PACK_CAP_DELTA17_PRIME_PERMIL = -0.432
GPP_BOUNDS = (18.256264, 850.0)


def relative_prime_cap_permil(
    delta17_permil: float,
    delta18_permil: float,
    *,
    lambda_ref: float = MODEL_LAMBDA,
) -> float:
    """Convert conventional deltas relative to HLA into logarithmic Delta-prime."""

    delta17_prime = 1000.0 * log1p(float(delta17_permil) / 1000.0)
    delta18_prime = 1000.0 * log1p(float(delta18_permil) / 1000.0)
    return float(delta17_prime - lambda_ref * delta18_prime)


def luz_equation3_productivity(
    *,
    co2_ppm: float,
    delta17_per_meg_relative_hla: float,
    reference_co2_ppm: float = REFERENCE_CO2_PPM,
    biological_equilibrium_per_meg: float = 155.0,
) -> float:
    """Evaluate Luz Eq. 3 with k_t/k_0=1 in the paper's linear coordinate."""

    reference_star = -float(biological_equilibrium_per_meg)
    sample_star = (
        float(delta17_per_meg_relative_hla) - biological_equilibrium_per_meg
    )
    return float((co2_ppm / reference_co2_ppm) * (reference_star / sample_star))


def _invert_gpp(
    *, target_cap_permil: float, pco2_ppm: float, uncertainty_permil: float
):
    return invert_updated_output_surface(
        UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=target_cap_permil,
            solve_for="GPP",
            measurement_uncertainty_permil=uncertainty_permil,
            p_o2_pal=PO2_PAL,
            p_co2_ppm=pco2_ppm,
            gpp_pgC_per_year=REFERENCE_GPP_PGC_PER_YEAR,
            solve_bounds=GPP_BOUNDS,
        ),
        verify_live_root=True,
    )


def _metrics(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    expected = np.asarray(
        [100.0 * float(row["reported_normalized_gross_production"]) for row in rows]
    )
    predicted = np.asarray([float(row[field]) for row in rows])
    residual = predicted - expected
    return {
        "mean_Luz_GPP_percent": float(np.mean(expected)),
        "mean_updated_GPP_percent": float(np.mean(predicted)),
        "bias_percentage_points": float(np.mean(residual)),
        "mean_absolute_error_percentage_points": float(np.mean(np.abs(residual))),
        "RMSE_percentage_points": float(np.sqrt(np.mean(residual**2))),
        "maximum_absolute_difference_percentage_points": float(
            np.max(np.abs(residual))
        ),
        "Pearson_r": float(np.corrcoef(expected, predicted)[0, 1]),
    }


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    age = np.asarray([float(row["gas_age_kyr"]) for row in rows])
    luz = np.asarray(
        [100.0 * float(row["reported_normalized_gross_production"]) for row in rows]
    )
    response = np.asarray(
        [float(row["updated_response_relative_GPP_percent"]) for row in rows]
    )
    pack = np.asarray([float(row["updated_Pack_anchored_GPP_percent"]) for row in rows])

    figure, axis = plt.subplots(figsize=(8.0, 4.7), constrained_layout=True)
    axis.plot(age, luz, "o-", color="#202020", label="Luz et al. Eq. 3")
    axis.plot(
        age,
        response,
        "s-",
        color="#167D8D",
        label="Updated model, response-relative",
    )
    axis.plot(
        age,
        pack,
        "^--",
        color="#B24B35",
        label="Updated model, Pack anchored",
    )
    axis.axhline(100.0, color="0.55", linewidth=0.8)
    axis.set(xlabel="Gas age (kyr)", ylabel="Inferred GPP (% of reference)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    samples = [row for row in source["rows"] if row["role"] == "historical sample"]
    measurement_uncertainty_permil = (
        float(source["analytical_precision"]["Delta17O_per_meg_standard_error"])
        / 1000.0
    )
    model_reference = run_updated_central_state(
        UpdatedForwardInput(
            p_o2_pal=PO2_PAL,
            p_co2_ppm=REFERENCE_CO2_PPM,
            gpp_pgC_per_year=REFERENCE_GPP_PGC_PER_YEAR,
        )
    )

    rows: list[dict[str, object]] = []
    for source_row in samples:
        delta18 = float(source_row["delta18O_permil_relative_HLA_gravity_corrected"])
        delta17 = float(source_row["delta17O_permil_relative_HLA_gravity_corrected"])
        relative_cap = relative_prime_cap_permil(delta17, delta18)
        printed_linear = 1000.0 * (delta17 - LUZ_LAMBDA * delta18)
        eq3 = luz_equation3_productivity(
            co2_ppm=float(source_row["CO2_ppm"]),
            delta17_per_meg_relative_hla=float(
                source_row["reported_Delta17O_per_meg_relative_HLA"]
            ),
            biological_equilibrium_per_meg=float(
                source["biological_equilibrium_Delta17O_per_meg_relative_HLA"]
            ),
        )
        response_target = (
            float(model_reference.cap_delta17_prime_permil) + relative_cap
        )
        pack_target = PACK_CAP_DELTA17_PRIME_PERMIL + relative_cap
        response_inverse = _invert_gpp(
            target_cap_permil=response_target,
            pco2_ppm=float(source_row["CO2_ppm"]),
            uncertainty_permil=measurement_uncertainty_permil,
        )
        pack_inverse = _invert_gpp(
            target_cap_permil=pack_target,
            pco2_ppm=float(source_row["CO2_ppm"]),
            uncertainty_permil=measurement_uncertainty_permil,
        )
        if response_inverse.central_root is None or pack_inverse.central_root is None:
            raise RuntimeError("Luz benchmark GPP inversion did not return one root")
        rows.append(
            {
                **source_row,
                "recalculated_linear_Delta17O_per_meg_lambda_0p521": printed_linear,
                "linear_Delta17O_minus_reported_per_meg": (
                    printed_linear
                    - float(source_row["reported_Delta17O_per_meg_relative_HLA"])
                ),
                "relative_Delta_prime17O_per_meg_lambda_0p528": (
                    1000.0 * relative_cap
                ),
                "recalculated_Luz_equation3_normalized_GPP": eq3,
                "equation3_minus_reported_normalized_GPP": (
                    eq3 - float(source_row["reported_normalized_gross_production"])
                ),
                "updated_response_relative_target_permil": response_target,
                "updated_response_relative_GPP_PgC_per_year": (
                    response_inverse.central_root
                ),
                "updated_response_relative_GPP_percent": (
                    100.0 * response_inverse.central_root / REFERENCE_GPP_PGC_PER_YEAR
                ),
                "updated_response_relative_guardrail_PgC_per_year": (
                    response_inverse.admissible_interval
                ),
                "updated_Pack_anchored_target_permil": pack_target,
                "updated_Pack_anchored_GPP_PgC_per_year": pack_inverse.central_root,
                "updated_Pack_anchored_GPP_percent": (
                    100.0 * pack_inverse.central_root / REFERENCE_GPP_PGC_PER_YEAR
                ),
                "updated_Pack_anchored_guardrail_PgC_per_year": (
                    pack_inverse.admissible_interval
                ),
                "response_relative_live_root_residual_permil": (
                    response_inverse.live_root_residual_permil
                ),
                "Pack_anchored_live_root_residual_permil": (
                    pack_inverse.live_root_residual_permil
                ),
            }
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, figure_path)

    response_metrics = _metrics(rows, "updated_response_relative_GPP_percent")
    pack_metrics = _metrics(rows, "updated_Pack_anchored_GPP_percent")
    report = {
        "schema_version": 1,
        "benchmark": "Luz et al. (1999) GISP2 productivity inverse comparison",
        "status": "complete",
        "role": (
            "Independent inverse-architecture comparison, not independent GPP "
            "validation: both methods infer GPP from the same O2 isotope and CO2 data."
        ),
        "model_fitted_to_Luz": False,
        "source": source,
        "coordinate_policy": {
            "primary": (
                "Exact log conversion of gravity-corrected HLA-relative delta17O "
                "and delta18O to lambda=0.528, added to the updated model's own "
                "280 ppm, 290 PgC/yr reference state."
            ),
            "absolute_sensitivity": (
                "The same relative isotope observations are separately anchored "
                "to Pack (2021) modern Delta-prime-17O=-0.432 per mil."
            ),
        },
        "fixed_inputs": {
            "pO2_PAL": PO2_PAL,
            "reference_CO2_ppm": REFERENCE_CO2_PPM,
            "reference_GPP_PgC_per_year": REFERENCE_GPP_PGC_PER_YEAR,
            "updated_model_reference_cap_delta17_prime_permil": (
                model_reference.cap_delta17_prime_permil
            ),
            "Pack_modern_cap_delta17_prime_permil": (
                PACK_CAP_DELTA17_PRIME_PERMIL
            ),
            "measurement_standard_error_permil": measurement_uncertainty_permil,
        },
        "response_relative_metrics": response_metrics,
        "Pack_anchored_metrics": pack_metrics,
        "source_reproduction": {
            "maximum_abs_linear_Delta17O_residual_per_meg": max(
                abs(float(row["linear_Delta17O_minus_reported_per_meg"]))
                for row in rows
            ),
            "maximum_abs_equation3_GPP_residual": max(
                abs(float(row["equation3_minus_reported_normalized_GPP"]))
                for row in rows
            ),
        },
        "rows": rows,
        "outputs": {"csv": str(csv_path), "figure": str(figure_path)},
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    report = run(arguments.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "response_relative_metrics": report["response_relative_metrics"],
                "Pack_anchored_metrics": report["Pack_anchored_metrics"],
                "source_reproduction": report["source_reproduction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
