"""Score named scenario presets against the core Young constraints.

The older global scorecard works on low-level model variants. This script
scores the public scenario presets that are now used by the plotting and
spherule-inversion workflows.
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
import json
import sys
import numpy as np
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_scenarios import (
    CURRENT_UPDATED_PHYSICAL_PRESET,
    CURRENT_YOUNG_PRINTED_PRESET,
    CURRENT_YOUNG_REPRODUCTION_PRESET,
    ScenarioInput,
    preset_names,
    run_scenario,
)
from integrate_fig9_fig10_transients import MODERN_PCO2, run_experiments
from young_fig8_reconstruction import d17o_from_pco2, digitize_fig8
from young_model_inventory import TABLE3_TARGETS
from young_validation_targets import fig7_digitized_targets, fig8_pco2_points


YOUNG_FIG9_VISUAL_MIN_D17O = -0.73
YOUNG_FIG9_VISUAL_MIN_SCALE = 0.08
YOUNG_FIG9_PEAK_CO2_PPM = 1000.0
YOUNG_FIG10_TEXT_SHIFT = -0.006
YOUNG_FIG10_TEXT_SCALE = 0.006

PRESETS_TO_SCORE = (
    CURRENT_YOUNG_PRINTED_PRESET,
    CURRENT_YOUNG_REPRODUCTION_PRESET,
    CURRENT_UPDATED_PHYSICAL_PRESET,
)


PREFERRED_PROCESSED_SCORE_MODE = "column_fixed"
PROCESSED_SCORE_MODES = ("preferred", "exact", "exact_fixed", "column_extra", "column_fixed", "raw")


@lru_cache(maxsize=4096)
def cached_run_scenario(scenario: ScenarioInput):
    """Run deterministic scorecard scenarios with in-process caching.

    The scorecard evaluates the same pCO2/GPP grid for multiple processed CO2
    reporting modes. Those modes change modern CO2 scoring only, not the O2
    Fig. 7/Fig. 8 surface, so caching avoids hundreds of duplicate explicit
    lower-box solves during tests and audits.
    """

    return run_scenario(scenario)


def context_row(
    *,
    preset: str,
    constraint: str,
    model: float,
    young: float,
    scale: float,
    note: str,
) -> dict:
    residual = float(model) - float(young)
    return {
        "preset": preset,
        "constraint": constraint,
        "group": "modern_table3_context",
        "model": float(model),
        "young": float(young),
        "residual": residual,
        "scale": scale,
        "weight": 0.0,
        "score": 0.0,
        "warnings": note,
    }


def parse_preset_list(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(values) - set(preset_names()))
    if unknown:
        raise ValueError(f"unknown preset(s): {', '.join(unknown)}")
    return values


def parse_processed_modes(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(values) - set(PROCESSED_SCORE_MODES))
    if unknown:
        raise ValueError(f"unknown processed mode(s): {', '.join(unknown)}")
    return values


def score_label(preset: str, processed_mode: str) -> str:
    return preset if processed_mode == "preferred" else f"{preset} [{processed_mode}]"


def effective_processed_mode(outputs: dict, processed_mode: str) -> str:
    if processed_mode != "preferred":
        return processed_mode
    if "column_processed_mixed_D17O_fixed_reported_permil" in outputs:
        return PREFERRED_PROCESSED_SCORE_MODE
    if "processed_parallel_flux_for_young_anomaly_mol_per_year" in outputs:
        return "exact_fixed"
    return "raw"


def scalar_residual_rows(preset: str, *, processed_mode: str = "preferred") -> list[dict]:
    result = cached_run_scenario(ScenarioInput(preset=preset))
    outputs = result.outputs
    label = score_label(preset, processed_mode)
    selected_mode = effective_processed_mode(outputs, processed_mode)
    co2_strat_model = outputs["CO2_strat_D17O_permil"]
    co2_flux_model = outputs["CO2_strat_D17O_flux_permil_mol_per_year"]
    processed_note = ""
    if "processed_parallel_flux_for_young_anomaly_mol_per_year" in outputs:
        if selected_mode == "exact":
            co2_strat_model = outputs["processed_parallel_mixed_D17O_with_extra_major_flux_permil"]
            co2_flux_model = (
                outputs["CO2_strat_D17O_flux_permil_mol_per_year"]
                + outputs["processed_parallel_flux_for_young_anomaly_mol_per_year"]
                * outputs["processed_parallel_CO2_D17O_permil"]
            )
            processed_note = (
                "exact Young-target parallel processed-export diagnostic scored for modern CO2 constraints; "
                f"processed/lower flux={outputs['processed_parallel_fraction_for_young_anomaly']:.4f}"
            )
        elif selected_mode == "exact_fixed":
            co2_strat_model = (
                outputs["CO2_export_D17O_permil"]
                + outputs["processed_parallel_fraction_for_young_fixed_reported_D17O"]
                * outputs["processed_parallel_CO2_D17O_permil"]
            )
            co2_flux_model = (
                outputs["CO2_strat_D17O_flux_permil_mol_per_year"]
                + outputs["processed_parallel_flux_for_young_fixed_reported_D17O_mol_per_year"]
                * outputs["processed_parallel_CO2_D17O_permil"]
            )
            processed_note = (
                "exact Young-target parallel processed diagnostic scored at fixed reported denominator; "
                f"processed/lower flux={outputs['processed_parallel_fraction_for_young_fixed_reported_D17O']:.4f}"
            )
        elif selected_mode == "column_extra":
            co2_strat_model = outputs["column_processed_mixed_D17O_extra_major_permil"]
            co2_flux_model = outputs["column_processed_anomaly_flux_permil_mol_per_year"]
            processed_note = (
                "source-derived Young-column processed law scored as extra major export; "
                f"activation={outputs['column_processed_upper_activation']:.3f}, "
                f"processed/lower={outputs['column_processed_fraction']:.4f}"
            )
        elif selected_mode == "column_fixed":
            co2_strat_model = outputs["column_processed_mixed_D17O_fixed_reported_permil"]
            co2_flux_model = outputs["column_processed_anomaly_flux_permil_mol_per_year"]
            processed_note = (
                "source-derived Young-column processed law scored at fixed reported denominator; "
                f"activation={outputs['column_processed_upper_activation']:.3f}, "
                f"processed/lower={outputs['column_processed_fraction']:.4f}"
            )
        elif selected_mode == "raw":
            processed_note = "raw lower-box CO2 diagnostics scored; processed alternatives retained as context"
        else:
            raise ValueError(f"unknown processed mode: {processed_mode}")
        if processed_mode == "preferred":
            processed_note = f"preferred processed mode resolved to {selected_mode}; {processed_note}"
    checks = [
        ("modern_O2_D17O", outputs["O2_trop_D17O_permil"], TABLE3_TARGETS["D17_O2_trop_permil"], 0.10, 2.0),
        ("modern_CO2_trop_D17O", outputs["CO2_trop_D17O_permil"], TABLE3_TARGETS["D17_CO2_trop_permil"], 0.10, 2.0),
        ("modern_CO2_strat_D17O", co2_strat_model, TABLE3_TARGETS["D17_CO2_strat_permil"], 0.20, 2.0),
        ("modern_O3_D17O", outputs["O3_strat_D17O_permil"], TABLE3_TARGETS["D17_O3_permil"], 0.20, 1.5),
        ("modern_O2_d18p", outputs["O2_trop_d18_prime_permil"], TABLE3_TARGETS["d18_O2_trop_permil"], 0.20, 1.0),
        ("modern_O2_d17p", outputs["O2_trop_d17_prime_permil"], TABLE3_TARGETS["d17_O2_trop_permil"], 0.20, 1.0),
        (
            "co2_d17_flux",
            co2_flux_model,
            TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"],
            8.55e15,
            1.5,
        ),
    ]
    rows = []
    for key, model, young, scale, weight in checks:
        residual = float(model) - float(young)
        norm = abs(residual) / scale
        rows.append(
            {
                "preset": label,
                "constraint": key,
                "group": "modern_table3",
                "model": float(model),
                "young": float(young),
                "residual": residual,
                "scale": scale,
                "weight": weight,
                "score": max(0.0, 1.0 - norm),
                "warnings": " | ".join([*result.warnings, processed_note] if processed_note else result.warnings),
            }
        )
    if processed_note:
        rows.extend(
            [
                context_row(
                    preset=label,
                    constraint="modern_CO2_strat_D17O_raw_lower_box",
                    model=outputs["CO2_strat_D17O_permil"],
                    young=TABLE3_TARGETS["D17_CO2_strat_permil"],
                    scale=0.20,
                    note=processed_note,
                ),
                context_row(
                    preset=label,
                    constraint="co2_d17_flux_raw_lower_box",
                    model=outputs["CO2_strat_D17O_flux_permil_mol_per_year"],
                    young=TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"],
                    scale=8.55e15,
                    note=processed_note,
                ),
                context_row(
                    preset=label,
                    constraint="processed_parallel_fraction_for_young_anomaly",
                    model=outputs["processed_parallel_fraction_for_young_anomaly"],
                    young=0.0,
                    scale=1.0,
                    note=processed_note,
                ),
                context_row(
                    preset=label,
                    constraint="modern_CO2_strat_D17O_exact_target_extra_major",
                    model=outputs["processed_parallel_mixed_D17O_with_extra_major_flux_permil"],
                    young=TABLE3_TARGETS["D17_CO2_strat_permil"],
                    scale=0.20,
                    note="exact Young-target extra-major processed diagnostic retained for QA context",
                ),
                context_row(
                    preset=label,
                    constraint="co2_d17_flux_exact_target",
                    model=(
                        outputs["CO2_strat_D17O_flux_permil_mol_per_year"]
                        + outputs["processed_parallel_flux_for_young_anomaly_mol_per_year"]
                        * outputs["processed_parallel_CO2_D17O_permil"]
                    ),
                    young=TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"],
                    scale=8.55e15,
                    note="exact Young-target extra-major processed diagnostic retained for QA context",
                ),
                context_row(
                    preset=label,
                    constraint="modern_CO2_strat_D17O_exact_target_fixed_reported",
                    model=(
                        outputs["CO2_export_D17O_permil"]
                        + outputs["processed_parallel_fraction_for_young_fixed_reported_D17O"]
                        * outputs["processed_parallel_CO2_D17O_permil"]
                    ),
                    young=TABLE3_TARGETS["D17_CO2_strat_permil"],
                    scale=0.20,
                    note="exact Young-target fixed-reported processed diagnostic retained for QA context",
                ),
                context_row(
                    preset=label,
                    constraint="co2_d17_flux_exact_target_fixed_reported",
                    model=(
                        outputs["CO2_strat_D17O_flux_permil_mol_per_year"]
                        + outputs["processed_parallel_flux_for_young_fixed_reported_D17O_mol_per_year"]
                        * outputs["processed_parallel_CO2_D17O_permil"]
                    ),
                    young=TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"],
                    scale=8.55e15,
                    note="exact Young-target fixed-reported processed diagnostic retained for QA context",
                ),
            ]
        )
        if "column_processed_fraction" in outputs:
            column_note = (
                "source-derived Young-column processed law retained alongside exact target diagnostic; "
                f"activation={outputs['column_processed_upper_activation']:.3f}, "
                f"processed/lower={outputs['column_processed_fraction']:.4f}"
            )
            rows.extend(
                [
                    context_row(
                        preset=label,
                        constraint="modern_CO2_strat_D17O_column_law_extra_major",
                        model=outputs["column_processed_mixed_D17O_extra_major_permil"],
                        young=TABLE3_TARGETS["D17_CO2_strat_permil"],
                        scale=0.20,
                        note=column_note,
                    ),
                    context_row(
                        preset=label,
                        constraint="modern_CO2_strat_D17O_column_law_fixed_reported",
                        model=outputs["column_processed_mixed_D17O_fixed_reported_permil"],
                        young=TABLE3_TARGETS["D17_CO2_strat_permil"],
                        scale=0.20,
                        note=column_note,
                    ),
                    context_row(
                        preset=label,
                        constraint="co2_d17_flux_column_law",
                        model=outputs["column_processed_anomaly_flux_permil_mol_per_year"],
                        young=TABLE3_TARGETS["D17_CO2_flux_permil_mol_per_year"],
                        scale=8.55e15,
                        note=column_note,
                    ),
                    context_row(
                        preset=label,
                        constraint="column_processed_fraction",
                        model=outputs["column_processed_fraction"],
                        young=outputs["processed_parallel_fraction_for_young_anomaly"],
                        scale=1.0,
                        note=column_note,
                    ),
                    context_row(
                        preset=label,
                        constraint="column_processed_upper_activation",
                        model=outputs["column_processed_upper_activation"],
                        young=0.0,
                        scale=1.0,
                        note=column_note,
                    ),
                ]
            )
    return rows


def fig7_residual_rows(preset: str) -> list[dict]:
    rows = []
    residuals = []
    for pco2_ppm, gpp_percent, young in fig7_digitized_targets(samples_per_contour=15):
        result = cached_run_scenario(
            ScenarioInput(preset=preset, p_co2_ppm=float(pco2_ppm), gpp_scale=gpp_percent / 100.0)
        )
        model = float(result.outputs["O2_trop_D17O_permil"])
        residual = model - young
        residuals.append(residual)
        rows.append(
            {
                "preset": preset,
                "constraint": f"fig7_{int(pco2_ppm)}ppm_{gpp_percent:.1f}pct_gpp",
                "group": "figure_behavior",
                "model": model,
                "young": young,
                "residual": residual,
                "scale": 0.10,
                "weight": 0.4,
                "score": max(0.0, 1.0 - abs(residual) / 0.10),
                "warnings": " | ".join(result.warnings),
            }
        )
    mean_abs = sum(abs(value) for value in residuals) / len(residuals)
    rows.append(
        {
            "preset": preset,
            "constraint": "fig7_digitized_mean_abs",
            "group": "figure_behavior",
            "model": mean_abs,
            "young": 0.0,
            "residual": mean_abs,
            "scale": 0.05,
            "weight": 1.5,
            "score": max(0.0, 1.0 - mean_abs / 0.05),
            "warnings": "",
        }
    )
    return rows


def fig8_residual_rows(preset: str) -> list[dict]:
    curves = digitize_fig8()
    rows = []
    residuals = []
    for gpp_percent in (50.0, 100.0):
        for pco2_ppm in fig8_pco2_points(dense=True):
            result = cached_run_scenario(ScenarioInput(preset=preset, p_co2_ppm=pco2_ppm, gpp_scale=gpp_percent / 100.0))
            model = float(result.outputs["O2_trop_D17O_permil"])
            young = float(d17o_from_pco2(pco2_ppm, int(gpp_percent), curves))
            residual = model - young
            residuals.append(residual)
            rows.append(
                {
                    "preset": preset,
                    "constraint": f"fig8_{gpp_percent:.0f}pct_{pco2_ppm:g}ppm",
                    "group": "figure_behavior",
                    "model": model,
                    "young": young,
                    "residual": residual,
                    "scale": 0.50,
                    "weight": 0.35,
                    "score": max(0.0, 1.0 - abs(residual) / 0.50),
                    "warnings": " | ".join(result.warnings),
                }
            )
    mean_abs = sum(abs(value) for value in residuals) / len(residuals)
    max_abs = max(abs(value) for value in residuals)
    rows.extend(
        [
            {
                "preset": preset,
                "constraint": "fig8_mean_abs",
                "group": "figure_behavior",
                "model": mean_abs,
                "young": 0.0,
                "residual": mean_abs,
                "scale": 0.25,
                "weight": 2.0,
                "score": max(0.0, 1.0 - mean_abs / 0.25),
                "warnings": "",
            },
            {
                "preset": preset,
                "constraint": "fig8_max_abs",
                "group": "figure_behavior",
                "model": max_abs,
                "young": 0.0,
                "residual": max_abs,
                "scale": 0.75,
                "weight": 1.0,
                "score": max(0.0, 1.0 - max_abs / 0.75),
                "warnings": "",
            },
        ]
    )
    return rows


def dynamic_residual_rows(preset: str) -> list[dict]:
    args = SimpleNamespace(
        preset=preset,
        model_variant="r7_throughput_diagnostic",
        r5_mode="variant_default",
        r7_throughput_factor=2.25,
        r8c_factor=1.0,
        co2_sink_factor=None,
        co2_ocean_infusion_factor=None,
        fig9_years=12000.0,
        fig10_years=150.0,
        samples=121,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    trace_rows, diagnostics = run_experiments(args)
    warnings = " | ".join(
        f"{row['experiment']}: {row['message']}"
        for row in diagnostics
        if not row["success"]
    )
    modern = next(row for row in trace_rows if row["experiment"] == "modern_initial")
    fig9 = [row for row in trace_rows if row["experiment"] == "fig9_half_photosynthesis"]
    fig10 = [row for row in trace_rows if row["experiment"] == "fig10_co2_step"]
    initial_d17 = float(modern["O2_trop_D17O_permil"])
    if preset == CURRENT_YOUNG_REPRODUCTION_PRESET:
        fig9_d17 = [float(row["O2_trop_D17O_permil"]) for row in fig9]
        fig9_d18 = [float(row["O2_trop_d18p_permil"]) for row in fig9]
        fig9_co2 = [MODERN_PCO2 * float(row["CO2_trop_mol"]) / 5.29e16 for row in fig9]
        fig9_initial_d17 = fig9_d17[0]
        fig10_shift = (
            float(fig10[-1]["O2_trop_D17O_permil"])
            - float(fig10[0]["O2_trop_D17O_permil"])
        )
        warnings = " | ".join(
            item
            for item in (
                warnings,
                "Fig. 9 uses source-derived bulk R7 driven by detailed no-finite-exposure CO2",
                "Fig. 10 uses source-derived Young bulk R7/biology reduction",
            )
            if item
        )
    else:
        fig9_d17 = [float(row["O2_trop_D17O_permil"]) for row in fig9]
        fig9_d18 = [float(row["O2_trop_d18p_permil"]) for row in fig9]
        fig9_co2 = [294.4 * float(row["CO2_trop_mol"]) / 5.29e16 for row in fig9]
        fig9_initial_d17 = initial_d17
        fig10_shift = float(fig10[-1]["O2_trop_D17O_permil"]) - initial_d17
    checks = [
        ("fig9_half_rp_final_O2_D17O", fig9_d17[-1], -0.539, 0.05, 2.0),
        ("fig9_min_O2_D17O", min(fig9_d17), YOUNG_FIG9_VISUAL_MIN_D17O, YOUNG_FIG9_VISUAL_MIN_SCALE, 1.5),
        ("fig9_peak_CO2_ppm", max(fig9_co2), YOUNG_FIG9_PEAK_CO2_PPM, 250.0, 1.0),
        ("fig9_peak_O2_d18p", max(fig9_d18), 28.0, 0.6, 1.0),
        ("fig9_positive_D17O_bump", max(0.0, max(fig9_d17) - fig9_initial_d17), 0.0, 0.02, 1.0),
        # Young Fig. 10 is described in the text/caption as a single-digit
        # per-meg effect after the industrial-era-length CO2 step, not as an
        # exact -10 per-meg target. Use a soft 6 per-meg center with a 6 per-meg
        # scale so the score rewards the right sign/order and does not overfit
        # the digitized-looking endpoint.
        ("fig10_150yr_shift", fig10_shift, YOUNG_FIG10_TEXT_SHIFT, YOUNG_FIG10_TEXT_SCALE, 1.0),
    ]
    rows = []
    for key, model, young, scale, weight in checks:
        residual = float(model) - float(young)
        rows.append(
            {
                "preset": preset,
                "constraint": key,
                "group": "dynamic_behavior",
                "model": float(model),
                "young": float(young),
                "residual": residual,
                "scale": scale,
                "weight": weight,
                "score": max(0.0, 1.0 - abs(residual) / scale),
                "warnings": warnings,
            }
        )
    return rows


def score_preset(preset: str, *, processed_mode: str = "preferred") -> list[dict]:
    label = score_label(preset, processed_mode)
    rows = scalar_residual_rows(preset, processed_mode=processed_mode) + fig7_residual_rows(preset) + fig8_residual_rows(preset)
    for row in rows:
        row["preset"] = label
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for preset in sorted({row["preset"] for row in rows}):
        subset = [row for row in rows if row["preset"] == preset]
        total_weight = sum(float(row["weight"]) for row in subset)
        score = sum(float(row["weight"]) * float(row["score"]) for row in subset) / total_weight
        fig8 = next(row for row in subset if row["constraint"] == "fig8_mean_abs")
        fig7 = next(row for row in subset if row["constraint"] == "fig7_digitized_mean_abs")
        out.append(
            {
                "preset": preset,
                "overall_score_percent": 100.0 * score,
                "fig7_mean_abs_permil": fig7["model"],
                "fig8_mean_abs_permil": fig8["model"],
                "fig8_max_abs_permil": next(row for row in subset if row["constraint"] == "fig8_max_abs")["model"],
            }
        )
    return sorted(out, key=lambda row: row["overall_score_percent"], reverse=True)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], summaries: list[dict], path: Path, *, include_dynamics: bool = False) -> None:
    constraint_text = "Young Table 3, digitized Fig. 7, and digitized Fig. 8"
    if include_dynamics:
        constraint_text += ", plus Fig. 9/Fig. 10 dynamic checks"
    lines = [
        "# Scenario Preset Scorecard",
        "",
        f"Scores compare public scenario presets against {constraint_text}.",
        "This deliberately scores the same presets used by the plotting and spherule-inversion scripts.",
        "When a scenario exposes processed CO2 diagnostics, the selected processed scoring mode is used for modern CO2_strat and CO2 flux.",
        "`preferred` resolves to the source-derived `column_fixed` Young-column law when available; exact-target and raw lower-box alternatives are retained as zero-weight QA context rows.",
        "",
        "| preset | overall % | Fig. 7 mean abs | Fig. 8 mean abs | Fig. 8 max abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['preset']} | {row['overall_score_percent']:.1f} | "
            f"{row['fig7_mean_abs_permil']:.4f} | {row['fig8_mean_abs_permil']:.4f} | "
            f"{row['fig8_max_abs_permil']:.4f} |"
        )
    lines.extend(["", "## Largest Residuals", ""])
    for preset in [row["preset"] for row in summaries]:
        subset = [row for row in rows if row["preset"] == preset]
        lines.append(f"### {preset}")
        lines.append("")
        lines.append("| constraint | group | model | Young | residual |")
        lines.append("|---|---|---:|---:|---:|")
        scored_subset = [row for row in subset if float(row["weight"]) > 0.0]
        for row in sorted(scored_subset, key=lambda item: abs(float(item["residual"])), reverse=True)[:10]:
            lines.append(
                f"| {row['constraint']} | {row['group']} | {row['model']:.6g} | "
                f"{row['young']:.6g} | {row['residual']:+.4g} |"
            )
        context_subset = [row for row in subset if float(row["weight"]) == 0.0]
        if context_subset:
            lines.append("")
            lines.append("Context rows retained outside the score:")
            lines.append("")
            lines.append("| constraint | model | reference | note |")
            lines.append("|---|---:|---:|---|")
            for row in context_subset:
                lines.append(
                    f"| {row['constraint']} | {row['model']:.6g} | "
                    f"{row['young']:.6g} | {row['warnings']} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default=",".join(PRESETS_TO_SCORE))
    parser.add_argument(
        "--processed-modes",
        default="preferred",
        help="Comma-separated processed CO2 scoring modes: preferred,exact,exact_fixed,column_extra,column_fixed,raw.",
    )
    parser.add_argument("--include-dynamics", action="store_true")
    args = parser.parse_args()

    requested = parse_preset_list(args.presets)
    processed_modes = parse_processed_modes(args.processed_modes)
    scored_presets = [preset for preset in requested if preset in preset_names()]
    rows = []
    for preset in scored_presets:
        for processed_mode in processed_modes:
            label = score_label(preset, processed_mode)
            rows.extend(score_preset(preset, processed_mode=processed_mode))
            if args.include_dynamics:
                dynamic_rows = dynamic_residual_rows(preset)
                for row in dynamic_rows:
                    row["preset"] = label
                rows.extend(dynamic_rows)
    summaries = summarize(rows)
    rows_csv = _PROJECT_OUTPUTS / "scenario_preset_scorecard.csv"
    summary_csv = _PROJECT_OUTPUTS / "scenario_preset_score_summary.csv"
    metadata_path = _PROJECT_OUTPUTS / "scenario_preset_scorecard.metadata.json"
    md_path = HERE / "scenario_preset_scorecard.md"
    write_csv(rows, rows_csv)
    write_csv(summaries, summary_csv)
    constraints = "Table 3, Fig. 7, Fig. 8"
    if args.include_dynamics:
        constraints += ", Fig. 9/Fig. 10 dynamics"
    metadata_path.write_text(
        json.dumps(
            {"presets": scored_presets, "processed_modes": processed_modes, "constraints": constraints},
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(rows, summaries, md_path, include_dynamics=args.include_dynamics)

    print("Scenario preset scorecard")
    print("-------------------------")
    for row in summaries:
        print(
            f"{row['preset']:30s} score={row['overall_score_percent']:5.1f}% "
            f"fig7={row['fig7_mean_abs_permil']:.4f} fig8={row['fig8_mean_abs_permil']:.4f}"
        )
    print()
    print(f"Wrote {rows_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
