"""Command-line interface for reproducible OXYTIB calculations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from model_result_workbook import build_transient_workbook
from public_model_service import (
    forward,
    inverse,
    pco2_trajectory_transient,
    photosynthesis_step_transient,
    state_step_transient,
)
from updated_molecular_forward_model import UpdatedForwardInput
from updated_molecular_transient import UpdatedTransientInput
from updated_output_surface_inverse import UpdatedSurfaceInverseInput
from updated_photosynthesis_transient import UpdatedPhotosynthesisTransientInput
from updated_pco2_trajectory_transient import (
    UpdatedPCO2TrajectoryInput,
)
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL


def _forward_input(args: argparse.Namespace, prefix: str = "") -> UpdatedForwardInput:
    return UpdatedForwardInput(
        p_o2_pal=getattr(args, f"{prefix}po2"),
        p_co2_ppm=getattr(args, f"{prefix}pco2"),
        gpp_pgC_per_year=getattr(args, f"{prefix}gpp"),
    )


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(content)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output)


def _transient_rows(envelope: dict[str, Any], experiment: str) -> list[dict[str, Any]]:
    result = envelope["result"]
    request = result["request"]
    rows: list[dict[str, Any]] = []
    photosynthesis = experiment == "photosynthesis"
    trajectory = experiment == "pCO2_trajectory"
    for index, (time, state) in enumerate(
        zip(result["time_years"], result["states"], strict=True)
    ):
        if photosynthesis:
            pco2 = result["pco2_ppm"][index]
            gpp = request["initial"]["gpp_pgC_per_year"]
            fraction = request["photosynthesis_fraction"]
        elif trajectory:
            pco2 = result["pco2_ppm"][index]
            gpp = request["initial"]["gpp_pgC_per_year"]
            fraction = None
        else:
            pco2 = request["final"]["p_co2_ppm"]
            gpp = request["final"]["gpp_pgC_per_year"]
            fraction = None
        rows.append(
            {
                "time_years": time,
                "pO2_PAL": state["o16o16_mol"] / GLOBAL_MAJOR_O2_MOLES_1PAL,
                "pCO2_ppm": pco2,
                "GPP_PgC_per_year": gpp,
                "photosynthesis_fraction_of_initial": fraction,
                "O2_Delta_prime_17O_permil": state["cap_delta17_prime_permil"],
                "O2_delta_prime_18O_permil": state["delta18_prime_permil"],
            }
        )
    return rows


def _write_transient(
    envelope: dict[str, Any],
    *,
    experiment: str,
    output: Path | None,
    output_format: str,
) -> None:
    if output_format == "json":
        _write_json(envelope, output)
        return
    if output is None:
        raise ValueError("--output is required for CSV and XLSX exports")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "xlsx":
        output.write_bytes(build_transient_workbook(envelope, experiment))
    else:
        rows = _transient_rows(envelope, experiment)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(output)


def _add_state(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    parser.add_argument(f"--{prefix}po2", type=float, default=1.0, metavar="PAL")
    parser.add_argument(f"--{prefix}pco2", type=float, default=294.0, metavar="PPM")
    parser.add_argument(f"--{prefix}gpp", type=float, default=290.0, metavar="PGC_YR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="calculation", required=True)

    forward_parser = commands.add_parser("forward", help="Run one steady-state forward calculation.")
    _add_state(forward_parser)
    forward_parser.add_argument("--output", type=Path)

    inverse_parser = commands.add_parser(
        "infer", help="Infer one coordinate while holding the other two fixed."
    )
    inverse_parser.add_argument("--target-d17o", type=float, required=True, metavar="PERMIL")
    inverse_parser.add_argument("--sigma", type=float, default=0.0, metavar="PERMIL")
    inverse_parser.add_argument("--solve-for", choices=("pCO2", "GPP", "pO2"), default="pCO2")
    _add_state(inverse_parser)
    inverse_parser.add_argument("--lower", type=float)
    inverse_parser.add_argument("--upper", type=float)
    inverse_parser.add_argument("--output", type=Path)

    transient_parser = commands.add_parser(
        "transient", help="Run a declared atmospheric time-response experiment."
    )
    transient_parser.add_argument(
        "--experiment",
        choices=("pCO2", "pO2", "GPP", "photosynthesis", "pCO2-trajectory"),
        required=True,
    )
    _add_state(transient_parser, "initial-")
    transient_parser.add_argument("--final-value", type=float)
    transient_parser.add_argument("--photosynthesis-fraction", type=float, default=0.5)
    transient_parser.add_argument("--trajectory-duration", type=float, default=174.0, metavar="YEARS")
    transient_parser.add_argument("--interpolation", choices=("linear", "smoothstep"), default="smoothstep")
    transient_parser.add_argument("--duration", type=float, default=12000.0, metavar="YEARS")
    transient_parser.add_argument("--samples", type=int, default=161)
    transient_parser.add_argument("--equilibrium-horizon", type=float, default=100000.0, metavar="YEARS")
    transient_parser.add_argument("--format", choices=("json", "csv", "xlsx"), default="json")
    transient_parser.add_argument("--output", type=Path)
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.calculation == "forward":
        _write_json(forward(_forward_input(args)), args.output)
        return

    if args.calculation == "infer":
        bounds = None
        if args.lower is not None or args.upper is not None:
            if args.lower is None or args.upper is None:
                raise ValueError("--lower and --upper must be supplied together")
            bounds = (args.lower, args.upper)
        request = UpdatedSurfaceInverseInput(
            target_air_cap_delta17_permil=args.target_d17o,
            solve_for=args.solve_for,
            measurement_uncertainty_permil=args.sigma,
            p_o2_pal=args.po2,
            p_co2_ppm=args.pco2,
            gpp_pgC_per_year=args.gpp,
            solve_bounds=bounds,
        )
        _write_json(inverse(request), args.output)
        return

    initial = _forward_input(args, "initial_")
    export_experiment = args.experiment
    if args.experiment == "photosynthesis":
        request = UpdatedPhotosynthesisTransientInput(
            initial=initial,
            photosynthesis_fraction=args.photosynthesis_fraction,
            duration_years=args.duration,
            sample_count=args.samples,
            equilibrium_search_max_years=args.equilibrium_horizon,
        )
        envelope = photosynthesis_step_transient(request)
    elif args.experiment == "pCO2-trajectory":
        if args.final_value is None:
            raise ValueError("--final-value is required for a pCO2 trajectory")
        request = UpdatedPCO2TrajectoryInput(
            initial=initial,
            final_pco2_ppm=args.final_value,
            transition_duration_years=args.trajectory_duration,
            interpolation=args.interpolation,
            duration_years=args.duration,
            sample_count=args.samples,
            equilibrium_search_max_years=args.equilibrium_horizon,
        )
        envelope = pco2_trajectory_transient(request)
        export_experiment = "pCO2_trajectory"
    else:
        if args.final_value is None:
            raise ValueError("--final-value is required for pCO2, pO2, and GPP steps")
        final = UpdatedForwardInput(
            p_o2_pal=args.final_value if args.experiment == "pO2" else initial.p_o2_pal,
            p_co2_ppm=args.final_value if args.experiment == "pCO2" else initial.p_co2_ppm,
            gpp_pgC_per_year=(args.final_value if args.experiment == "GPP" else initial.gpp_pgC_per_year),
        )
        envelope = state_step_transient(
            UpdatedTransientInput(
                initial=initial,
                final=final,
                duration_years=args.duration,
                sample_count=args.samples,
                equilibrium_search_max_years=args.equilibrium_horizon,
            )
        )
    _write_transient(
        envelope,
        experiment=export_experiment,
        output=args.output,
        output_format=args.format,
    )


def main() -> int:
    try:
        _run(build_parser().parse_args())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
