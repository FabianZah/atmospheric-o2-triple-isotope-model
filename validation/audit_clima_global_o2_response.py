"""Propagate Clima-coupled local R7 operators through the global O2 budget.

The biological budget, source-isoflux normalization, isotope coordinates, and
fixed-point solver are identical to the released updated model. Only the native
Photochem parent atmosphere is replaced by the paired Clima-anomaly experiment.
The calculation is restricted to its explicit 1 PAL pO2 nodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from audit_local_r7_response_operator import _fit_operator  # noqa: E402
from biological_o2_ensemble import (  # noqa: E402
    central_biological_member,
    fixed_po2_partitioned_biological_budget,
)
from global_o2_isotope_reservoir import (  # noqa: E402
    IsotopologueTendency,
    frozen_photochemical_steady_state,
)
from local_r7_response_operator import LocalR7ResponseOperator  # noqa: E402
from modern_isotope_reference import modern_reference_isotope_compositions  # noqa: E402
from self_consistent_isotope_fixed_point import (  # noqa: E402
    solve_mechanistic_fixed_point_with_hybr_fallback,
)
from updated_molecular_forward_model import (  # noqa: E402
    UpdatedForwardInput,
    run_updated_central_state,
)


DEFAULT_GPP_PERCENT = (25.0, 50.0, 100.0, 150.0)
UPDATED_MODERN_GPP_PGC_PER_YEAR = 290.0


def _load_manifest_rows(
    manifest_path: Path, artifact_directory: Path, *, po2_pal: float
) -> tuple[list[dict[str, object]], str]:
    report = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"no scenarios in {manifest_path}")
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        if not np.isclose(float(scenario["pO2_PAL"]), po2_pal, atol=1.0e-12):
            continue
        pco2 = float(scenario["pCO2_ppm"])
        artifact_value = scenario.get("artifact")
        if not isinstance(artifact_value, str) or not artifact_value:
            raise ValueError(f"scenario lacks artifact metadata at {pco2:g} ppm")
        artifact = Path(artifact_directory) / Path(artifact_value).name
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != scenario["artifact_sha256"]:
            raise ValueError(f"native atmosphere checksum mismatch: {artifact}")
        rows.append(
            {
                **scenario,
                "artifact": str(artifact.resolve()),
                "pO2_PAL": po2_pal,
                "pCO2_ppm": pco2,
            }
        )
    rows.sort(key=lambda row: float(row["pCO2_ppm"]))
    if len(rows) < 3:
        raise ValueError("global O2 climate audit needs at least three pCO2 nodes")
    return rows, str(report.get("pressure_convention", "unknown"))


def _forcing_scale(bundle_path: Path) -> float:
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    return float(
        bundle["transfer_normalization"]["native_surface_molecular_forcing_scale"][
            "mean"
        ]
    )


def _scale_tendency(
    tendency: IsotopologueTendency, scale: float
) -> IsotopologueTendency:
    return IsotopologueTendency(
        *map(float, scale * tendency.values),
        source=(
            f"{tendency.source}; unchanged updated-model Adnew molecular-isoflux "
            f"normalization={scale:.12g}"
        ),
    )


def _solve_climate_state(
    operator: LocalR7ResponseOperator,
    *,
    gpp_pgC_per_year: float,
    forcing_scale: float,
    po2_pal: float,
) -> tuple[np.ndarray, float, int, str]:
    reference_o2, _co2 = modern_reference_isotope_compositions()
    member = central_biological_member()

    def evaluate_target(state: np.ndarray) -> np.ndarray:
        photochemical = _scale_tendency(operator.evaluate(state), forcing_scale)
        biological = fixed_po2_partitioned_biological_budget(
            member,
            po2_pal=po2_pal,
            gpp_pgC_per_year=gpp_pgC_per_year,
            photochemical=photochemical,
        )
        target = frozen_photochemical_steady_state(
            biological,
            photochemical,
            source=(
                "Clima-parent structural sensitivity with unchanged updated "
                "global O2 and biological budget"
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
        finite_difference_step=np.asarray([0.001, 0.0001]),
        tolerance=1.0e-8,
        maximum_iterations=36,
    )
    if not result.converged:
        raise RuntimeError(
            "Clima-parent global O2 fixed point did not converge: "
            f"residual={result.maximum_absolute_residual:.6g} per mil"
        )
    return (
        result.state,
        float(result.maximum_absolute_residual),
        int(result.target_evaluations),
        result.solver_method,
    )


def _plot(rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    gpp_values = sorted({float(row["GPP_percent_of_updated_modern"]) for row in rows})
    colors = dict(
        zip(
            gpp_values,
            plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(gpp_values))),
            strict=True,
        )
    )
    for gpp in gpp_values:
        selected = sorted(
            (
                row
                for row in rows
                if float(row["GPP_percent_of_updated_modern"]) == gpp
            ),
            key=lambda row: float(row["pCO2_ppm"]),
        )
        pco2 = [float(row["pCO2_ppm"]) for row in selected]
        axes[0].plot(
            pco2,
            [float(row["fixed_cap_delta17_prime_permil"]) for row in selected],
            "o-",
            color=colors[gpp],
            label=f"{gpp:g}% fixed",
        )
        axes[0].plot(
            pco2,
            [float(row["climate_cap_delta17_prime_permil"]) for row in selected],
            "s--",
            color=colors[gpp],
            label=f"{gpp:g}% Clima",
        )
        axes[1].plot(
            pco2,
            [float(row["climate_minus_fixed_D17O_permil"]) for row in selected],
            "o-",
            color=colors[gpp],
            label=f"{gpp:g}% GPP",
        )
        axes[2].plot(
            pco2,
            [float(row["climate_minus_fixed_delta18_permil"]) for row in selected],
            "o-",
            color=colors[gpp],
            label=f"{gpp:g}% GPP",
        )
    axes[0].set_ylabel("Atmospheric O$_2$ Δ′$^{17}$O (‰)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].axhline(0.0, color="0.45", linewidth=0.9)
    axes[1].set_ylabel("Clima − fixed Δ′$^{17}$O (‰)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].axhline(0.0, color="0.45", linewidth=0.9)
    axes[2].set_ylabel("Clima − fixed δ′$^{18}$O (‰)")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("pCO$_2$ (ppm)")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(
    manifest_path: Path,
    artifact_directory: Path,
    bundle_path: Path,
    output_path: Path,
    *,
    po2_pal: float = 1.0,
    gpp_percent: tuple[float, ...] = DEFAULT_GPP_PERCENT,
) -> dict[str, object]:
    scenarios, pressure_convention = _load_manifest_rows(
        manifest_path, artifact_directory, po2_pal=po2_pal
    )
    scale = _forcing_scale(bundle_path)
    operators = {
        float(scenario["pCO2_ppm"]): _fit_operator(scenario)
        for scenario in scenarios
    }

    rows: list[dict[str, object]] = []
    for pco2, operator in operators.items():
        for percent in gpp_percent:
            absolute_gpp = UPDATED_MODERN_GPP_PGC_PER_YEAR * percent / 100.0
            fixed = run_updated_central_state(
                UpdatedForwardInput(po2_pal, pco2, absolute_gpp),
                bundle_path=bundle_path,
            )
            climate, residual, evaluations, method = _solve_climate_state(
                operator,
                gpp_pgC_per_year=absolute_gpp,
                forcing_scale=scale,
                po2_pal=po2_pal,
            )
            rows.append(
                {
                    "pO2_PAL": po2_pal,
                    "pCO2_ppm": pco2,
                    "GPP_percent_of_updated_modern": percent,
                    "GPP_PgC_per_year": absolute_gpp,
                    "fixed_delta18_prime_permil": fixed.delta18_prime_permil,
                    "climate_delta18_prime_permil": float(climate[0]),
                    "climate_minus_fixed_delta18_permil": float(
                        climate[0] - fixed.delta18_prime_permil
                    ),
                    "fixed_cap_delta17_prime_permil": fixed.cap_delta17_prime_permil,
                    "climate_cap_delta17_prime_permil": float(climate[1]),
                    "climate_minus_fixed_D17O_permil": float(
                        climate[1] - fixed.cap_delta17_prime_permil
                    ),
                    "fixed_solver_residual_permil": (
                        fixed.maximum_fixed_point_residual_permil
                    ),
                    "climate_solver_residual_permil": residual,
                    "climate_target_evaluations": evaluations,
                    "climate_solver_method": method,
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

    d17_differences = np.asarray(
        [float(row["climate_minus_fixed_D17O_permil"]) for row in rows]
    )
    d18_differences = np.asarray(
        [float(row["climate_minus_fixed_delta18_permil"]) for row in rows]
    )
    report: dict[str, object] = {
        "audit": "Clima-parent structural sensitivity propagated to global O2",
        "status": "diagnostic_not_promoted",
        "pO2_PAL": po2_pal,
        "pressure_convention": pressure_convention,
        "pCO2_nodes_ppm": list(operators),
        "GPP_percent_of_updated_modern": list(gpp_percent),
        "updated_modern_GPP_PgC_per_year": UPDATED_MODERN_GPP_PGC_PER_YEAR,
        "operator_count": len(operators),
        "scenario_count": len(rows),
        "all_scenarios_converged": True,
        "maximum_absolute_D17O_change_permil": float(
            np.max(np.abs(d17_differences))
        ),
        "maximum_absolute_delta18_change_permil": float(
            np.max(np.abs(d18_differences))
        ),
        "maximum_climate_fixed_point_residual_permil": float(
            max(float(row["climate_solver_residual_permil"]) for row in rows)
        ),
        "controlled_difference": (
            "Only the pCO2-dependent Clima temperature anomaly and the resulting "
            "native parent chemistry differ. The updated biological budget, Adnew "
            "source-isoflux normalization, global O2 inventory, and numerical "
            "fixed-point equations are unchanged."
        ),
        "promotion_decision": {
            "promote_to_released_model": False,
            "reason": (
                f"This is a single-pO2, {pressure_convention} structural "
                "sensitivity using a simplified isothermal-stratosphere climate "
                "profile."
            ),
        },
        "operators": {
            f"{pco2:g}ppm": operator.as_dict()
            for pco2, operator in operators.items()
        },
        "rows": rows,
        "sources": {
            "manifest": str(Path(manifest_path).resolve()),
            "artifact_directory": str(Path(artifact_directory).resolve()),
            "released_bundle": str(Path(bundle_path).resolve()),
        },
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "model_data" / "updated_r7_response_surface_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "clima_global_o2_response.json",
    )
    parser.add_argument("--po2-pal", type=float, default=1.0)
    parser.add_argument(
        "--gpp-percent",
        type=float,
        nargs="+",
        default=list(DEFAULT_GPP_PERCENT),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.manifest,
                args.artifact_directory,
                args.bundle,
                args.output,
                po2_pal=args.po2_pal,
                gpp_percent=tuple(args.gpp_percent),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
