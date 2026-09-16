r"""Integrate Young Fig. 9/Fig. 10 style transient perturbations.

Run this with an environment that has SciPy:

    python code/integrate_fig9_fig10_transients.py

The script starts from the current reconstructed full-atmosphere steady state
and then applies the two perturbations described by Young:

- Fig. 9: halve photosynthesis rp while keeping respiration kr fixed.
- Fig. 10: instantaneously raise atmospheric CO2 from 294.4 to 400 ppm, then
  hold the CO2 budget at the new pCO2 condition so O2 isotopes relax slowly.

This is a diagnostic for model behavior, not a claim that the current closure
terms are Young-exact.
"""

from __future__ import annotations

# --- path bootstrap (direct execution) ---
import sys as _sys
from pathlib import Path as _Path
_root = next((p for p in _Path(__file__).resolve().parents if (p / ".project-root").exists()), None)
if _root is not None:
    for _sub in ("code", "validation"):
        _p = str(_root / _sub)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
# --- end path bootstrap ---

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from calibrated_model import MODEL_VARIANTS, R5_MODES, ModelConfig, build_reactions, run_model
from model_scenarios import (
    CURRENT_YOUNG_REPRODUCTION_PRESET,
    ScenarioInput,
    config_from_scenario,
    effective_rp_o2,
    preset_names,
)
from model_runner import isotope_summaries
from processed_access_reservoir import (
    ACCESS_SPECIES,
    EXTENDED_SPECIES_ORDER,
    AccessStateRelaxation,
    extend_state_with_access,
)
from reactions import derivative
from table3_state import SPECIES_ORDER
from young_model_inventory import PARAMETERS
from young_reactions import TABLE3_O2_TROP
from young_bulk_fig9 import integrate_young_bulk_fig9
from young_bulk_fig10 import (
    YOUNG_FIG10_INITIAL_PCO2_PPM,
    integrate_young_bulk_fig10,
)


HERE = Path(__file__).resolve().parent
MODERN_PCO2 = 294.4
FIG10_PCO2 = 400.0
MODERN_RP_O2 = PARAMETERS["k_respiration_per_year"] * TABLE3_O2_TROP


@dataclass(frozen=True)
class PhotosynthesisCarbonDriverTrajectory:
    """Detailed-box trajectory used to drive the updated isotope reservoir."""

    time_years: np.ndarray
    pco2_ppm: np.ndarray
    po2_pal: np.ndarray
    solver: object

    def __iter__(self):
        """Retain the historical three-value unpacking API."""

        yield self.time_years
        yield self.pco2_ppm
        yield self.solver


def scaled_rhs_factory(reactions, y_scale: np.ndarray, species_order=SPECIES_ORDER):
    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        y = np.maximum(z * y_scale, 1.0e-300)
        return derivative(y, reactions, species_order) / y_scale

    return rhs


def integrate(
    y0: np.ndarray,
    reactions,
    t_end_yr: float,
    sample_count: int,
    rtol: float,
    atol: float,
    species_order=SPECIES_ORDER,
) -> tuple[np.ndarray, np.ndarray, object]:
    y_scale = np.maximum(np.abs(y0), 1.0)
    z0 = y0 / y_scale
    t_eval = np.linspace(0.0, t_end_yr, sample_count)
    # BDF's adaptive finite-difference Jacobian can emit overflow warnings
    # while probing trial perturbations even when the accepted trajectory is
    # finite. Retain those messages as solver provenance and validate the
    # accepted state explicitly instead of leaking warnings into the UI.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = solve_ivp(
            scaled_rhs_factory(reactions, y_scale, species_order),
            (0.0, t_end_yr),
            z0,
            method="BDF",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max(t_end_yr / 2000.0, 1.0),
        )
    result.internal_runtime_warnings = tuple(
        sorted({f"{item.category.__name__}: {item.message}" for item in caught})
    )
    result.internal_runtime_warning_events = len(caught)
    if not np.all(np.isfinite(result.y)):
        raise RuntimeError("BDF transient returned a non-finite accepted state")
    with np.errstate(over="raise", invalid="raise"):
        raw_y = result.y.T * y_scale
    result.minimum_unclipped_inventory_mol = float(np.min(raw_y))
    result.negative_inventory_count_before_clipping = int(np.count_nonzero(raw_y < 0.0))
    y = np.maximum(raw_y, 1.0e-300)
    return result.t, y, result


def summary_row(
    t: float,
    y: np.ndarray,
    experiment: str,
    preset: str,
    model_variant: str,
    r5_mode: str,
    r7_throughput_factor: float,
    r8c_factor: float,
    species_order=SPECIES_ORDER,
) -> dict:
    summaries = {summary.label: summary for summary in isotope_summaries(y[: len(SPECIES_ORDER)])}
    idx = {name: i for i, name in enumerate(species_order)}
    row = {
        "experiment": experiment,
        "preset": preset,
        "model_variant": model_variant,
        "r5_mode": r5_mode,
        "r7_throughput_factor": r7_throughput_factor,
        "r8c_factor": r8c_factor,
        "time_yr": t,
        "O2_trop_mol": y[idx["O2_trop"]],
        "CO2_trop_mol": y[idx["CO2_trop"]],
        "O2_trop_D17O_permil": summaries["O2_trop"].cap_delta17,
        "O2_trop_d18p_permil": summaries["O2_trop"].delta18_prime,
        "CO2_trop_D17O_permil": summaries["CO2_trop"].cap_delta17,
        "CO2_strat_D17O_permil": summaries["CO2_strat"].cap_delta17,
        "O3_strat_D17O_permil": summaries["O3_strat"].cap_delta17,
    }
    if ACCESS_SPECIES in idx:
        row[ACCESS_SPECIES] = float(y[idx[ACCESS_SPECIES]])
    return row


def co2_pulse_state(y0: np.ndarray, target_pco2_ppm: float) -> np.ndarray:
    idx = {name: i for i, name in enumerate(SPECIES_ORDER)}
    y = y0.copy()
    factor = target_pco2_ppm / MODERN_PCO2
    for name in ("CO2_trop", "CO18O_trop", "CO17O_trop", "CO2_strat", "CO18O_strat", "CO17O_strat"):
        y[idx[name]] *= factor
    return y


def config_from_args(args, *, solve_mode: str, p_co2_ppm: float = MODERN_PCO2) -> ModelConfig:
    if args.preset:
        return config_from_scenario(
            ScenarioInput(
                preset=args.preset,
                p_co2_ppm=p_co2_ppm,
                solve_mode=solve_mode,
                co2_sink_factor=args.co2_sink_factor,
                co2_ocean_infusion_factor=args.co2_ocean_infusion_factor,
                alpha_respiration_18=getattr(args, "alpha_respiration_18", None),
                beta_respiration_17=getattr(args, "beta_respiration_17", None),
                evapotranspiration_alpha_18=getattr(args, "evapotranspiration_alpha_18", None),
                evapotranspiration_beta_17=getattr(args, "evapotranspiration_beta_17", None),
                r8_rate_factor=getattr(args, "r8_rate_factor", None),
                r8c_factor=getattr(args, "r8c_factor_override", None),
                a_mif=getattr(args, "a_mif", None),
            )
        )
    return ModelConfig(
        p_co2_ppm=p_co2_ppm,
        model_variant=args.model_variant,
        solve_mode=solve_mode,
        r5_mode=args.r5_mode,
        young_steady_co2_source_isotope_mode="smow",
        r7_throughput_factor=args.r7_throughput_factor,
        r8c_factor=args.r8c_factor,
        co2_sink_factor=args.co2_sink_factor,
        co2_ocean_infusion_factor=args.co2_ocean_infusion_factor
        if args.co2_ocean_infusion_factor is not None
        else 1.0,
    )


def integrate_photosynthesis_carbon_driver(
    base_config: ModelConfig,
    *,
    photosynthesis_fraction: float,
    duration_years: float,
    sample_count: int,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> PhotosynthesisCarbonDriverTrajectory:
    """Return the detailed CO2 response to photosynthesis at fixed respiration."""

    if not 0.0 < photosynthesis_fraction:
        raise ValueError("photosynthesis fraction must be positive")

    no_finite = replace(
        base_config,
        r7_finite_exposure_exchange_time_modern_yr=None,
        r7_finite_exposure_residence_time_yr=None,
    )
    initial_run = run_model(no_finite)
    if no_finite.processed_access_reservoir_mode == "lagged_state":
        species_order = EXTENDED_SPECIES_ORDER
        initial = extend_state_with_access(
            initial_run.y, no_finite.processed_access_reservoir_initial
        )
        perturbed = replace(
            no_finite,
            rp_o2=photosynthesis_fraction * effective_rp_o2(no_finite),
            processed_column_low_gpp_recovery_po2_mode="access_state",
        )
        reactions = [
            *build_reactions(perturbed),
            AccessStateRelaxation(
                tau_yr=no_finite.processed_access_reservoir_tau_yr,
                half_pal=no_finite.processed_access_reservoir_half_pal,
                hill=no_finite.processed_access_reservoir_hill,
            ),
        ]
    else:
        species_order = SPECIES_ORDER
        initial = initial_run.y
        perturbed = replace(
            no_finite,
            rp_o2=photosynthesis_fraction * effective_rp_o2(no_finite),
        )
        reactions = build_reactions(perturbed)

    # Fifty-year output spacing captures the early CO2 maximum that drives
    # the bulk isotope response; later behavior remains inexpensive.
    driver_samples = max(sample_count, int(np.ceil(duration_years / 50.0)) + 1)
    times, states, solved = integrate(
        initial,
        reactions,
        duration_years,
        driver_samples,
        rtol,
        atol,
        species_order=species_order,
    )
    index = {name: position for position, name in enumerate(species_order)}
    pco2 = (
        base_config.p_co2_ppm
        * states[:, index["CO2_trop"]]
        / states[0, index["CO2_trop"]]
    )
    po2 = (
        base_config.p_o2_pal
        * states[:, index["O2_trop"]]
        / states[0, index["O2_trop"]]
    )
    return PhotosynthesisCarbonDriverTrajectory(
        time_years=times,
        pco2_ppm=pco2,
        po2_pal=po2,
        solver=solved,
    )


def _source_derived_young_fig9_driver(
    base_config: ModelConfig,
    args,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Return the detailed carbon-cycle driver without the old R7 exposure term."""

    return integrate_photosynthesis_carbon_driver(
        base_config,
        photosynthesis_fraction=0.5,
        duration_years=args.fig9_years,
        sample_count=args.samples,
        rtol=args.rtol,
        atol=args.atol,
    )


def _source_derived_young_transient_rows(
    base_config: ModelConfig,
    modern_row: dict,
    args,
) -> tuple[list[dict], list[dict]]:
    """Calculate Young Fig. 9/Fig. 10 from printed bulk R7 and biology."""

    fig9_time, fig9_pco2, driver = _source_derived_young_fig9_driver(
        base_config, args
    )
    fig9 = integrate_young_bulk_fig9(
        fig9_time, fig9_pco2, rtol=args.rtol, atol=args.atol
    )
    fig10_time = np.linspace(0.0, args.fig10_years, args.samples)
    fig10 = integrate_young_bulk_fig10(
        fig10_time, rtol=args.rtol, atol=args.atol
    )

    rows: list[dict] = []
    for index, time in enumerate(fig9.time_years):
        rows.append(
            {
                **modern_row,
                "experiment": "fig9_half_photosynthesis",
                "model_variant": "young_printed_bulk_r7_live_co2_reduction",
                "r7_throughput_factor": 1.0,
                "time_yr": float(time),
                "O2_trop_mol": fig9.states[index].o16o16,
                "CO2_trop_mol": 5.29e16 * float(fig9.pco2_ppm[index]) / MODERN_PCO2,
                "O2_trop_D17O_permil": float(fig9.cap_delta17_permil[index]),
                "O2_trop_d18p_permil": float(fig9.delta18_prime_permil[index]),
            }
        )
    for index, time in enumerate(fig10.time_years):
        rows.append(
            {
                **modern_row,
                "experiment": "fig10_co2_step",
                "model_variant": "young_printed_bulk_r7_reduction",
                "r7_throughput_factor": 1.0,
                "time_yr": float(time),
                "O2_trop_mol": fig10.states[index].o16o16,
                "CO2_trop_mol": 5.29e16 * FIG10_PCO2 / MODERN_PCO2,
                "O2_trop_D17O_permil": float(
                    fig10.states[index].cap_delta17_prime_permil
                ),
                "O2_trop_d18p_permil": float(
                    fig10.states[index].delta18_prime_permil
                ),
            }
        )
    diagnostics = [
        {
            "experiment": "fig9_half_photosynthesis",
            "success": bool(driver.success),
            "message": (
                "source-derived Young bulk R7 scaled by the detailed "
                "no-finite-exposure CO2 driver; " + driver.message
            ),
            "nfev": int(driver.nfev),
            "njev": int(driver.njev),
            "nlu": int(driver.nlu),
            "final_time_yr": float(fig9.time_years[-1]),
        },
        {
            "experiment": "fig10_co2_step",
            "success": True,
            "message": (
                "source-derived Young Table 2/Table 3 bulk R7 plus printed "
                "biological turnover; no finite-exposure adjustment"
            ),
            "nfev": "",
            "njev": "",
            "nlu": "",
            "final_time_yr": float(fig10.time_years[-1]),
        },
    ]
    return rows, diagnostics


def run_experiments(args) -> tuple[list[dict], list[dict]]:
    base_config = config_from_args(args, solve_mode="full_atmosphere")
    base_run = run_model(base_config)
    y0 = base_run.y
    preset_label = args.preset or "manual"

    rows = [
        summary_row(
            0.0,
            y0,
            "modern_initial",
            preset_label,
            base_config.model_variant,
            base_config.r5_mode,
            base_config.r7_throughput_factor,
            base_config.r8c_factor,
        )
    ]
    diagnostics = [
        {
            "experiment": "modern_initial",
            "success": base_run.full_result.converged,
            "message": "full_atmosphere steady solve",
            "nfev": "",
            "njev": "",
            "nlu": "",
            "final_time_yr": 0.0,
        }
    ]

    if args.preset == CURRENT_YOUNG_REPRODUCTION_PRESET:
        source_rows, source_diagnostics = _source_derived_young_transient_rows(
            base_config, rows[0], args
        )
        rows.extend(source_rows)
        diagnostics.extend(source_diagnostics)
        return rows, diagnostics

    if base_config.processed_access_reservoir_mode == "lagged_state":
        fig9_species_order = EXTENDED_SPECIES_ORDER
        fig9_y0 = extend_state_with_access(y0, base_config.processed_access_reservoir_initial)
        fig9_config = replace(
            base_config,
            rp_o2=0.5 * MODERN_RP_O2,
            processed_column_low_gpp_recovery_po2_mode="access_state",
        )
        fig9_reactions = [
            *build_reactions(fig9_config),
            AccessStateRelaxation(
                tau_yr=base_config.processed_access_reservoir_tau_yr,
                half_pal=base_config.processed_access_reservoir_half_pal,
                hill=base_config.processed_access_reservoir_hill,
            ),
        ]
        fig9_message_prefix = (
            f"lagged processed-access state tau={base_config.processed_access_reservoir_tau_yr:g} yr; "
        )
    else:
        fig9_species_order = SPECIES_ORDER
        fig9_y0 = y0
        fig9_config = replace(
            base_config,
            rp_o2=0.5 * MODERN_RP_O2,
        )
        fig9_reactions = build_reactions(fig9_config)
        fig9_message_prefix = ""
    t, y_series, result = integrate(
        fig9_y0,
        fig9_reactions,
        args.fig9_years,
        args.samples,
        args.rtol,
        args.atol,
        species_order=fig9_species_order,
    )
    rows.extend(
        summary_row(
            float(tt),
            yy,
            "fig9_half_photosynthesis",
            preset_label,
            fig9_config.model_variant,
            fig9_config.r5_mode,
            fig9_config.r7_throughput_factor,
            fig9_config.r8c_factor,
            species_order=fig9_species_order,
        )
        for tt, yy in zip(t, y_series)
    )
    diagnostics.append(
        {
            "experiment": "fig9_half_photosynthesis",
            "success": result.success,
            "message": fig9_message_prefix + result.message,
            "nfev": result.nfev,
            "njev": result.njev,
            "nlu": result.nlu,
            "final_time_yr": float(result.t[-1]) if len(result.t) else 0.0,
        }
    )

    fig10_config = config_from_args(args, solve_mode="young_steady", p_co2_ppm=FIG10_PCO2)
    fig10_reactions = build_reactions(fig10_config)
    y_pulse = co2_pulse_state(y0, FIG10_PCO2)
    t, y_series, result = integrate(y_pulse, fig10_reactions, args.fig10_years, args.samples, args.rtol, args.atol)
    rows.extend(
        summary_row(
            float(tt),
            yy,
            "fig10_co2_step",
            preset_label,
            fig10_config.model_variant,
            fig10_config.r5_mode,
            fig10_config.r7_throughput_factor,
            fig10_config.r8c_factor,
        )
        for tt, yy in zip(t, y_series)
    )
    diagnostics.append(
        {
            "experiment": "fig10_co2_step",
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": result.njev,
            "nlu": result.nlu,
            "final_time_yr": float(result.t[-1]) if len(result.t) else 0.0,
        }
    )

    return rows, diagnostics


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    by_exp: dict[str, list[dict]] = {}
    for row in rows:
        by_exp.setdefault(row["experiment"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, experiment, title in (
        (axes[0], "fig9_half_photosynthesis", "Fig. 9-style rp/2 transient"),
        (axes[1], "fig10_co2_step", "Fig. 10-style CO2 step"),
    ):
        exp_rows = by_exp[experiment]
        time = np.array([float(row["time_yr"]) for row in exp_rows])
        d17 = np.array([float(row["O2_trop_D17O_permil"]) for row in exp_rows])
        ax.plot(time, d17, color="#1f6f8b", lw=2.0)
        ax.axhline(float(by_exp["modern_initial"][0]["O2_trop_D17O_permil"]), color="#777777", lw=1.0, ls="--")
        if experiment == "fig9_half_photosynthesis":
            ax.axhline(-0.539, color="#b23a48", lw=1.0, ls=":", label="Young final")
        ax.set_title(title)
        ax.set_xlabel("time (yr)")
        ax.set_ylabel("O$_2$ Δ$'^{17}$O (‰)")
        ax.grid(True, color="#d9d9d9", lw=0.6)
        ax.legend(loc="best", frameon=False) if experiment == "fig9_half_photosynthesis" else None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_short_summary(rows: list[dict], diagnostics: list[dict]) -> None:
    by_exp: dict[str, list[dict]] = {}
    for row in rows:
        by_exp.setdefault(row["experiment"], []).append(row)
    modern = by_exp["modern_initial"][0]
    print("Transient diagnostics")
    print("---------------------")
    print(f"modern O2 D17O: {float(modern['O2_trop_D17O_permil']):.4f} permil")
    for experiment in ("fig9_half_photosynthesis", "fig10_co2_step"):
        exp_rows = by_exp[experiment]
        final = exp_rows[-1]
        min_d17 = min(float(row["O2_trop_D17O_permil"]) for row in exp_rows)
        max_d17 = max(float(row["O2_trop_D17O_permil"]) for row in exp_rows)
        print(
            f"{experiment}: final O2 D17O={float(final['O2_trop_D17O_permil']):.4f} "
            f"range=[{min_d17:.4f}, {max_d17:.4f}] "
            f"final O2={float(final['O2_trop_mol']):.4e} mol "
            f"final CO2={float(final['CO2_trop_mol']):.4e} mol"
        )
    print()
    for item in diagnostics:
        print(f"{item['experiment']}: success={item['success']} final_time={item['final_time_yr']} message={item['message']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("",) + preset_names(), default="")
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="table3_diagnostic")
    parser.add_argument("--r5-mode", choices=R5_MODES, default="table3_balanced")
    parser.add_argument("--r7-throughput-factor", type=float, default=2.25)
    parser.add_argument("--r8c-factor", type=float, default=1.0)
    parser.add_argument("--co2-sink-factor", type=float, default=None)
    parser.add_argument("--co2-ocean-infusion-factor", type=float, default=None)
    parser.add_argument("--fig9-years", type=float, default=50000.0)
    parser.add_argument("--fig10-years", type=float, default=10000.0)
    parser.add_argument("--samples", type=int, default=251)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--output", default="outputs/fig9_fig10_transients.csv")
    parser.add_argument("--diagnostics-output", default="outputs/fig9_fig10_transient_solver_diagnostics.csv")
    parser.add_argument("--plot-output", default="")
    args = parser.parse_args()

    rows, diagnostics = run_experiments(args)
    write_csv(rows, HERE / args.output)
    write_csv(diagnostics, HERE / args.diagnostics_output)
    if args.plot_output:
        write_plot(rows, HERE / args.plot_output)
    print_short_summary(rows, diagnostics)
    print()
    print(f"Wrote {HERE / args.output}")
    if args.plot_output:
        print(f"Wrote {HERE / args.plot_output}")


if __name__ == "__main__":
    main()
