"""Direct fast-column/slow-O2 fixed points for representative end members."""

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
for directory in (ROOT / "code", ROOT / "validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from audit_photochem_endmember_slow_o2_coupling import (  # noqa: E402
    GLOBAL_MAJOR_O2_MOLES_1PAL,
    _fixed_po2_budget,
)
from fixed_boundary_isotope_column import solve_fixed_boundary_column  # noqa: E402
from global_o2_isotope_reservoir import (  # noqa: E402
    frozen_photochemical_steady_state,
    isotopologue_tendency_from_atom_flux,
    r7_global_transfer_from_tendency,
)
from gridded_oxygen_chemistry import local_reaction_tendency_mol_per_year  # noqa: E402
from modern_isotope_reference import (  # noqa: E402
    PrimeIsotopeComposition,
    modern_reference_isotope_compositions,
)
from photochem_endmember_isotope_kernel import (  # noqa: E402
    build_photochem_endmember_isotope_kernel,
)
from self_consistent_isotope_fixed_point import (  # noqa: E402
    solve_mechanistic_fixed_point,
)


GPP_CASES = {
    0.1: (25.0,),
    1.0: (25.0, 100.0),
    2.0: (25.0,),
}
R7_EVALUATION_RELATIVE_CLOSURE_LIMIT = 5.0e-10
R7_FINAL_RELATIVE_CLOSURE_LIMIT = 1.0e-10


def _r7_closure_passes(value: float, *, final_state: bool) -> bool:
    limit = (
        R7_FINAL_RELATIVE_CLOSURE_LIMIT
        if final_state
        else R7_EVALUATION_RELATIVE_CLOSURE_LIMIT
    )
    return bool(np.isfinite(value) and value <= limit)


def _resolve_gpp_cases(
    forcing_lookup: dict[tuple[float, float], dict[str, object]],
    requested_values: tuple[float, ...] | None,
) -> dict[float, tuple[float, ...]]:
    po2_values = sorted({po2 for po2, _ in forcing_lookup})
    if requested_values is None:
        missing = [po2 for po2 in po2_values if po2 not in GPP_CASES]
        if missing:
            raise ValueError(
                "No default GPP cases are defined for pO2 values "
                f"{missing}; pass --gpp-percent-values explicitly."
            )
        return {po2: GPP_CASES[po2] for po2 in po2_values}

    values = tuple(float(value) for value in requested_values)
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("GPP values must be a non-empty sequence of positive numbers.")
    if len(set(values)) != len(values):
        raise ValueError("GPP values must be unique.")
    return {po2: values for po2 in po2_values}


def _target_evaluator(
    *,
    artifact: Path,
    po2_pal: float,
    pco2_ppm: float,
    gpp_percent: float,
    co2_composition: PrimeIsotopeComposition,
):
    records: list[dict[str, object]] = []

    def evaluate(state: np.ndarray) -> np.ndarray:
        o2_composition = PrimeIsotopeComposition(
            delta18_prime_permil=float(state[0]),
            cap_delta17_prime_permil=float(state[1]),
            source=(
                "self-consistent global O2 state supplied to the native-parent "
                "isotope kernel"
            ),
        )
        build = build_photochem_endmember_isotope_kernel(
            artifact,
            o2_composition=o2_composition,
            co2_composition=co2_composition,
        )
        solved = solve_fixed_boundary_column(
            build.column,
            maximum_years=100.0,
            steady_throughput_relative_tolerance=1.0e-8,
            maximum_step_years=1.0,
        )
        if not solved.solver_success or not solved.converged_to_steady:
            raise RuntimeError(
                f"fast isotope column failed at state {state.tolist()}: "
                f"{solved.solver_message}"
            )
        tendency = local_reaction_tendency_mol_per_year(
            solved.inventory_moles,
            species_names=build.column.species_system.species_names,
            air_moles=build.column.species_system.air_moles,
            pressure_pa=build.pressure_pa,
            temperature_k=build.temperature_k,
            reactions=build.r7_reactions,
        )
        transfer = r7_global_transfer_from_tendency(
            tendency,
            species_names=build.column.species_system.species_names,
            source=(
                f"self-consistent native-parent R7 at {po2_pal:g} PAL, "
                f"{pco2_ppm:g} ppm, {gpp_percent:g}% GPP"
            ),
        )
        if not _r7_closure_passes(
            transfer.maximum_relative_closure_residual, final_state=False
        ):
            raise RuntimeError(
                "self-consistent R7 transfer fails the numerical-path atom-"
                "closure limit at state "
                f"{state.tolist()}: relative residual "
                f"{transfer.maximum_relative_closure_residual:.6g}"
            )
        photochemical = isotopologue_tendency_from_atom_flux(transfer.global_o2)
        biological = _fixed_po2_budget(
            po2_pal=po2_pal,
            gpp_percent=gpp_percent,
            photochemical=photochemical,
        )
        target = frozen_photochemical_steady_state(
            biological,
            photochemical,
            source="self-consistent native-parent global O2 target",
        )
        target_state = np.asarray(
            [target.delta18_prime_permil, target.cap_delta17_prime_permil]
        )
        records.append(
            {
                "state": np.asarray(state, dtype=float).copy(),
                "target": target_state.copy(),
                "photochemical": photochemical,
                "respiration_rate_per_year": biological.respiration_rate_per_year,
                "major_O2_relative_residual": (
                    target.o16o16 / (GLOBAL_MAJOR_O2_MOLES_1PAL * po2_pal) - 1.0
                ),
                "R7_atom_closure": transfer.maximum_relative_closure_residual,
                "fast_column_residual": solved.maximum_free_throughput_scaled_residual,
            }
        )
        return target_state

    return evaluate, records


def _history_record(step) -> dict[str, object]:
    return {
        "iteration": step.iteration,
        "input_delta18_prime_permil": float(step.state[0]),
        "input_cap_delta17_prime_permil": float(step.state[1]),
        "target_delta18_prime_permil": float(step.target[0]),
        "target_cap_delta17_prime_permil": float(step.target[1]),
        "residual_delta18_prime_permil": float(step.residual[0]),
        "residual_cap_delta17_prime_permil": float(step.residual[1]),
        "maximum_absolute_residual_permil": step.maximum_absolute_residual,
        "response_jacobian": (
            None if step.response_jacobian is None else step.response_jacobian.tolist()
        ),
        "response_spectral_radius": (
            None
            if step.response_jacobian is None
            else float(np.max(np.abs(np.linalg.eigvals(step.response_jacobian))))
        ),
        "linear_system_condition": step.linear_system_condition,
        "newton_step_permil": (
            None if step.newton_step is None else step.newton_step.tolist()
        ),
    }


def _run_case(
    forcing: dict[str, object],
    frozen: dict[str, object],
    *,
    gpp_percent: float,
) -> dict[str, object]:
    po2 = float(forcing["pO2_PAL"])
    pco2 = float(forcing["pCO2_ppm"])
    artifact = Path(str(forcing["artifact"]))
    o2_reference, co2_composition = modern_reference_isotope_compositions()
    evaluate, evaluations = _target_evaluator(
        artifact=artifact,
        po2_pal=po2,
        pco2_ppm=pco2,
        gpp_percent=gpp_percent,
        co2_composition=co2_composition,
    )
    result = solve_mechanistic_fixed_point(
        evaluate,
        np.asarray(
            [
                o2_reference.delta18_prime_permil,
                o2_reference.cap_delta17_prime_permil,
            ]
        ),
        finite_difference_step=np.asarray([0.5, 0.5]),
        tolerance=1.0e-4,
        maximum_iterations=6,
    )
    final_record = evaluations[-1]
    nonempty_jacobians = [
        step.response_jacobian
        for step in result.history
        if step.response_jacobian is not None
    ]
    spectral_radii = [
        float(np.max(np.abs(np.linalg.eigvals(jacobian))))
        for jacobian in nonempty_jacobians
    ]
    conditions = [
        float(step.linear_system_condition)
        for step in result.history
        if step.linear_system_condition is not None
    ]
    photo = final_record["photochemical"]
    return {
        "pO2_PAL": po2,
        "pCO2_ppm": pco2,
        "GPP_percent_of_Young": gpp_percent,
        "converged": result.converged,
        "self_consistent_delta18_prime_permil": float(result.state[0]),
        "self_consistent_cap_delta17_prime_permil": float(result.state[1]),
        "direct_holdout_target_delta18_prime_permil": float(result.target[0]),
        "direct_holdout_target_cap_delta17_prime_permil": float(result.target[1]),
        "maximum_direct_holdout_residual_permil": (
            result.maximum_absolute_residual
        ),
        "frozen_pack_flux_delta18_prime_permil": float(
            frozen["equilibrium_delta18_prime_permil"]
        ),
        "frozen_pack_flux_cap_delta17_prime_permil": float(
            frozen["equilibrium_cap_delta17_prime_permil"]
        ),
        "self_consistency_shift_cap_delta17_prime_permil": float(
            result.state[1] - float(frozen["equilibrium_cap_delta17_prime_permil"])
        ),
        "target_evaluations": result.target_evaluations,
        "maximum_response_spectral_radius": max(spectral_radii),
        "ordinary_iteration_locally_unstable": max(spectral_radii) >= 1.0,
        "maximum_Newton_system_condition": max(conditions),
        "major_O2_relative_steady_residual": float(
            final_record["major_O2_relative_residual"]
        ),
        "maximum_relative_R7_atom_closure_residual": float(
            max(record["R7_atom_closure"] for record in evaluations)
        ),
        "final_relative_R7_atom_closure_residual": float(
            final_record["R7_atom_closure"]
        ),
        "maximum_fast_column_throughput_residual": float(
            max(record["fast_column_residual"] for record in evaluations)
        ),
        "final_R7_O16O16_tendency_mol_per_year": photo.o16o16,
        "final_R7_O16O17_tendency_mol_per_year": photo.o16o17,
        "final_R7_O16O18_tendency_mol_per_year": photo.o16o18,
        "history": [_history_record(step) for step in result.history],
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    flat = [{key: value for key, value in row.items() if key != "history"} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _plot(rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    po2_values = sorted({float(row["pO2_PAL"]) for row in rows})
    color_positions = np.linspace(0.08, 0.92, len(po2_values))
    colors = {
        po2: plt.get_cmap("viridis")(position)
        for po2, position in zip(po2_values, color_positions, strict=True)
    }
    markers = {25.0: "o", 100.0: "s"}
    for row in rows:
        po2 = float(row["pO2_PAL"])
        gpp = float(row["GPP_percent_of_Young"])
        axes[0].scatter(
            float(row["frozen_pack_flux_cap_delta17_prime_permil"]),
            float(row["self_consistent_cap_delta17_prime_permil"]),
            color=colors[po2],
            marker=markers.get(gpp, "D"),
            s=48,
        )
    limits = np.asarray(
        [
            min(
                min(float(row["frozen_pack_flux_cap_delta17_prime_permil"]) for row in rows),
                min(float(row["self_consistent_cap_delta17_prime_permil"]) for row in rows),
            ),
            max(
                max(float(row["frozen_pack_flux_cap_delta17_prime_permil"]) for row in rows),
                max(float(row["self_consistent_cap_delta17_prime_permil"]) for row in rows),
            ),
        ]
    )
    axes[0].plot(limits, limits, color="0.45", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel=r"Frozen-boundary equilibrium $\Delta'^{17}$O (‰)",
        ylabel=r"Self-consistent equilibrium $\Delta'^{17}$O (‰)",
    )
    for po2, color in colors.items():
        selected = sorted(
            (
                row
                for row in rows
                if float(row["pO2_PAL"]) == po2
                and float(row["GPP_percent_of_Young"]) == 25.0
            ),
            key=lambda row: float(row["pCO2_ppm"]),
        )
        axes[1].plot(
            [float(row["pCO2_ppm"]) for row in selected],
            [float(row["self_consistent_cap_delta17_prime_permil"]) for row in selected],
            "o-",
            color=color,
            label=f"{po2:g} PAL, 25% GPP",
        )
    selected_100 = sorted(
        (row for row in rows if float(row["GPP_percent_of_Young"]) == 100.0),
        key=lambda row: float(row["pCO2_ppm"]),
    )
    axes[1].plot(
        [float(row["pCO2_ppm"]) for row in selected_100],
        [float(row["self_consistent_cap_delta17_prime_permil"]) for row in selected_100],
        "s--",
        color="0.15",
        label="1 PAL, 100% GPP",
    )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="pCO$_2$ (ppm)",
        ylabel=r"Self-consistent O$_2$ $\Delta'^{17}$O (‰)",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(
    forcing_path: Path,
    frozen_path: Path,
    output_path: Path,
    *,
    gpp_percent_values: tuple[float, ...] | None = None,
    pco2_ppm_values: tuple[float, ...] | None = None,
) -> dict[str, object]:
    forcing_report = json.loads(Path(forcing_path).read_text(encoding="utf-8"))
    frozen_report = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    forcing_lookup = {
        (float(row["pO2_PAL"]), float(row["pCO2_ppm"])): row
        for row in forcing_report["scenarios"]
    }
    if pco2_ppm_values is not None:
        requested_pco2 = tuple(float(value) for value in pco2_ppm_values)
        if (
            not requested_pco2
            or len(set(requested_pco2)) != len(requested_pco2)
            or any(not np.isfinite(value) or value <= 0.0 for value in requested_pco2)
        ):
            raise ValueError("pCO2 values must be unique positive finite numbers")
        available_pco2 = {pco2 for _po2, pco2 in forcing_lookup}
        missing = sorted(set(requested_pco2) - available_pco2)
        if missing:
            raise ValueError(f"requested pCO2 values are absent from forcing: {missing}")
        forcing_lookup = {
            key: row for key, row in forcing_lookup.items() if key[1] in requested_pco2
        }
    frozen_lookup = {
        (
            float(row["pO2_PAL"]),
            float(row["pCO2_ppm"]),
            float(row["GPP_percent_of_Young"]),
        ): row
        for row in frozen_report["scenarios"]
    }
    rows: list[dict[str, object]] = []
    gpp_cases = _resolve_gpp_cases(forcing_lookup, gpp_percent_values)
    for po2, gpp_values in gpp_cases.items():
        pco2_values = sorted(
            pco2 for candidate_po2, pco2 in forcing_lookup if candidate_po2 == po2
        )
        for gpp in gpp_values:
            for pco2 in pco2_values:
                rows.append(
                    _run_case(
                        forcing_lookup[(po2, pco2)],
                        frozen_lookup[(po2, pco2, gpp)],
                        gpp_percent=gpp,
                    )
                )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    _write_csv(rows, csv_path)
    _plot(rows, figure_path)
    report = {
        "audit": "self-consistent native-parent fast-column/slow-O2 holdouts",
        "status": (
            "gate_passed"
            if all(bool(row["converged"]) for row in rows)
            and max(float(row["maximum_direct_holdout_residual_permil"]) for row in rows)
            <= 1.0e-4
            and max(abs(float(row["major_O2_relative_steady_residual"])) for row in rows)
            <= 1.0e-12
            and max(
                float(row["maximum_relative_R7_atom_closure_residual"])
                for row in rows
            )
            <= R7_EVALUATION_RELATIVE_CLOSURE_LIMIT
            and max(
                float(row["final_relative_R7_atom_closure_residual"])
                for row in rows
            )
            <= R7_FINAL_RELATIVE_CLOSURE_LIMIT
            else "gate_failed"
        ),
        "solver_policy": (
            "Newton solve of T(x)-x using full mechanistic finite-difference "
            "Jacobians in delta-prime-18O and Delta-prime-17O. No physical or "
            "numerical relaxation factor is applied."
        ),
        "boundary_policy": (
            "Native Photochem parent atmospheres are fixed; CO2 lower-boundary "
            "isotope composition remains the modern Adnew et al. (2025) value "
            "for this architecture holdout."
        ),
        "R7_closure_policy": (
            "Intermediate Newton/finite-difference states must close R7 oxygen "
            f"atoms within {R7_EVALUATION_RELATIVE_CLOSURE_LIMIT:.1e} relative; "
            "every accepted final fixed point retains the stricter "
            f"{R7_FINAL_RELATIVE_CLOSURE_LIMIT:.1e} limit. The separate path "
            "limit accommodates floating-point cancellation only and does not "
            "alter or project any flux."
        ),
        "case_count": len(rows),
        "pCO2_values_ppm": sorted({float(row["pCO2_ppm"]) for row in rows}),
        "ordinary_iteration_unstable_case_count": sum(
            bool(row["ordinary_iteration_locally_unstable"]) for row in rows
        ),
        "maximum_direct_holdout_residual_permil": max(
            float(row["maximum_direct_holdout_residual_permil"]) for row in rows
        ),
        "maximum_relative_R7_atom_closure_residual": max(
            float(row["maximum_relative_R7_atom_closure_residual"]) for row in rows
        ),
        "maximum_final_relative_R7_atom_closure_residual": max(
            float(row["final_relative_R7_atom_closure_residual"]) for row in rows
        ),
        "scenarios": rows,
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forcing",
        type=Path,
        default=ROOT / "outputs" / "photochem_endmember_isotope_kernel.json",
    )
    parser.add_argument(
        "--frozen",
        type=Path,
        default=ROOT / "outputs" / "photochem_endmember_slow_o2_coupling.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "photochem_endmember_self_consistent_o2.json",
    )
    parser.add_argument(
        "--gpp-percent-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "GPP percentages to solve at every available pO2 value. Omit to "
            "retain the predefined representative cases."
        ),
    )
    parser.add_argument(
        "--pco2-ppm-values",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of native pCO2 columns to solve.",
    )
    args = parser.parse_args()
    requested_gpp = (
        None
        if args.gpp_percent_values is None
        else tuple(args.gpp_percent_values)
    )
    print(
        json.dumps(
            run(
                args.forcing,
                args.frozen,
                args.output,
                gpp_percent_values=requested_gpp,
                pco2_ppm_values=(
                    None
                    if args.pco2_ppm_values is None
                    else tuple(args.pco2_ppm_values)
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
