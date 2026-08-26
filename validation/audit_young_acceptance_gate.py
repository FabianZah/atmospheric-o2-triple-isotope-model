"""Acceptance gate for the Young-anchored reconstruction.

This is not another fitting script. It combines point residuals, textual
checks, and curve-shape diagnostics so the model is judged against the whole
published Young constraint set rather than a few selected values.
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
import csv
import argparse
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_scenarios import CURRENT_YOUNG_REPRODUCTION_PRESET, ScenarioInput, run_scenario
from score_scenario_presets import dynamic_residual_rows, scalar_residual_rows
from young_architecture_candidate import (
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME,
    YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME,
    YOUNG_LIKE_V2_NAME,
)
from young_fig8_reconstruction import d17o_from_pco2, digitize_fig8
from young_validation_targets import fig7_digitized_targets, fig8_pco2_points, load_fig7_digitized_contours


OUTPUTS = _PROJECT_OUTPUTS
DEFAULT_PRESETS = (CURRENT_YOUNG_REPRODUCTION_PRESET,)
PROCESSED_MODE = "preferred"


@dataclass(frozen=True)
class AcceptanceCriterion:
    key: str
    value: float
    limit: float
    relation: str
    group: str
    note: str

    @property
    def passed(self) -> bool:
        if self.relation == "<=":
            return self.value <= self.limit
        if self.relation == ">=":
            return self.value >= self.limit
        raise ValueError(f"unknown relation {self.relation!r}")


@lru_cache(maxsize=2048)
def model_d17o(preset: str, pco2_ppm: float, gpp_percent: float) -> float:
    result = run_scenario(
        ScenarioInput(
            preset=preset,
            p_co2_ppm=float(pco2_ppm),
            gpp_scale=float(gpp_percent) / 100.0,
        )
    )
    return float(result.outputs["O2_trop_D17O_permil"])


def table_criteria(preset: str) -> list[AcceptanceCriterion]:
    rows = scalar_residual_rows(preset, processed_mode=PROCESSED_MODE)
    criteria = []
    limits = {
        "modern_O2_D17O": 0.05,
        "modern_CO2_trop_D17O": 0.10,
        "modern_CO2_strat_D17O": 0.10,
        "modern_O3_D17O": 0.20,
        "modern_O2_d18p": 0.20,
        "modern_O2_d17p": 0.20,
        "co2_d17_flux": 1.0e15,
    }
    for row in rows:
        key = row["constraint"]
        if key not in limits:
            continue
        residual = abs(float(row["residual"]))
        criteria.append(
            AcceptanceCriterion(
                key=key,
                value=residual,
                limit=limits[key],
                relation="<=",
                group="table3",
                note=f"absolute residual; processed mode={PROCESSED_MODE}",
            )
        )
    return criteria


def fig_residual_criteria(preset: str) -> tuple[list[AcceptanceCriterion], list[dict]]:
    rows = []
    criteria = []
    fig7_residuals = []
    for pco2_ppm, gpp_percent, young in fig7_digitized_targets(samples_per_contour=15):
        model = model_d17o(preset, pco2_ppm, gpp_percent)
        residual = model - young
        fig7_residuals.append(residual)
        rows.append(
            {
                "figure": "fig7",
                "pco2_ppm": pco2_ppm,
                "gpp_percent": gpp_percent,
                "model": model,
                "young": young,
                "residual": residual,
            }
        )
    curves = digitize_fig8()
    fig8_residuals = []
    for gpp_percent in (50.0, 100.0):
        for pco2_ppm in fig8_pco2_points(dense=True):
            model = model_d17o(preset, pco2_ppm, gpp_percent)
            young = float(d17o_from_pco2(pco2_ppm, int(gpp_percent), curves))
            residual = model - young
            fig8_residuals.append(residual)
            rows.append(
                {
                    "figure": "fig8",
                    "pco2_ppm": pco2_ppm,
                    "gpp_percent": gpp_percent,
                    "model": model,
                    "young": young,
                    "residual": residual,
                }
            )
    criteria.extend(
        [
            AcceptanceCriterion(
                "fig7_mean_abs",
                float(np.mean(np.abs(fig7_residuals))),
                0.03,
                "<=",
                "published_figures",
                "mean absolute residual against digitized Fig. 7",
            ),
            AcceptanceCriterion(
                "fig7_max_abs",
                float(np.max(np.abs(fig7_residuals))),
                0.08,
                "<=",
                "published_figures",
                "maximum absolute residual against digitized Fig. 7",
            ),
            AcceptanceCriterion(
                "fig8_mean_abs",
                float(np.mean(np.abs(fig8_residuals))),
                0.10,
                "<=",
                "published_figures",
                "mean absolute residual against digitized Fig. 8",
            ),
            AcceptanceCriterion(
                "fig8_max_abs",
                float(np.max(np.abs(fig8_residuals))),
                0.25,
                "<=",
                "published_figures",
                "maximum absolute residual against digitized Fig. 8",
            ),
        ]
    )
    return criteria, rows


def text_criteria(preset: str) -> list[AcceptanceCriterion]:
    modern = model_d17o(preset, 294.4, 100.0)
    plus_100ppm = model_d17o(preset, 394.4, 100.0) - modern
    gpp_70 = model_d17o(preset, 294.4, 70.0) - modern
    contours = load_fig7_digitized_contours()
    fig7_494_70 = float(np.interp(70.0, contours[494][:, 0], contours[494][:, 1]))
    gpp_70_494ppm = model_d17o(preset, 494.4, 70.0)
    return [
        AcceptanceCriterion(
            "text_100ppm_shift_abs",
            abs(plus_100ppm - (-0.050)),
            0.03,
            "<=",
            "text_constraints",
            "Young text: +100 ppm CO2 gives about -50 per meg shift",
        ),
        AcceptanceCriterion(
            "text_70pct_gpp_shift_abs",
            abs(gpp_70 - (-0.050)),
            0.03,
            "<=",
            "text_constraints",
            "Young text: GPP decrease to 70% gives about -50 per meg shift",
        ),
        AcceptanceCriterion(
            "text_70pct_gpp_494ppm_digitized_abs",
            abs(gpp_70_494ppm - fig7_494_70),
            0.06,
            "<=",
            "text_constraints",
            "Young text example cross-checked against digitized Fig. 7 because strict -80 per meg wording conflicts with the contour",
        ),
    ]


def dynamic_criteria(preset: str) -> list[AcceptanceCriterion]:
    rows = dynamic_residual_rows(preset)
    limits = {
        "fig9_half_rp_final_O2_D17O": 0.02,
        "fig9_min_O2_D17O": 0.03,
        "fig9_peak_CO2_ppm": 150.0,
        "fig9_peak_O2_d18p": 0.60,
        "fig9_positive_D17O_bump": 0.02,
        "fig10_150yr_shift": 0.01,
    }
    criteria = []
    for row in rows:
        key = row["constraint"]
        if key not in limits:
            continue
        criteria.append(
            AcceptanceCriterion(
                key=key,
                value=abs(float(row["residual"])),
                limit=limits[key],
                relation="<=",
                group="dynamic_figures",
                note="absolute residual against Young Fig. 9/Fig. 10 dynamic check",
            )
        )
    return criteria


def slope_correlation(model_values: np.ndarray, young_values: np.ndarray, pco2_grid: np.ndarray) -> float:
    x = np.log(pco2_grid)
    model_slopes = np.diff(model_values) / np.diff(x)
    young_slopes = np.diff(young_values) / np.diff(x)
    if np.std(model_slopes) <= 0.0 or np.std(young_slopes) <= 0.0:
        return 0.0
    return float(np.corrcoef(model_slopes, young_slopes)[0, 1])


def shape_criteria(preset: str) -> tuple[list[AcceptanceCriterion], list[dict]]:
    curves = digitize_fig8()
    rows = []
    criteria = []
    published_grid = np.array(fig8_pco2_points(dense=True), dtype=float)
    extrap_grid = np.array([294.4, 1000.0, 5000.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0])
    hook_tolerance = 0.01
    for gpp_percent in (50.0, 100.0):
        model_published = np.array([model_d17o(preset, pco2, gpp_percent) for pco2 in published_grid])
        young_published = np.array([d17o_from_pco2(pco2, int(gpp_percent), curves) for pco2 in published_grid])
        corr = slope_correlation(model_published, young_published, published_grid)
        model_extrap = np.array([model_d17o(preset, pco2, gpp_percent) for pco2 in extrap_grid])
        increases = np.diff(model_extrap)
        hook_count = int(np.sum(increases > hook_tolerance))
        max_hook = float(np.max(increases)) if len(increases) else 0.0
        criteria.extend(
            [
                AcceptanceCriterion(
                    f"fig8_{int(gpp_percent)}pct_slope_correlation",
                    corr,
                    0.85,
                    ">=",
                    "shape",
                    "slope correlation with digitized Young Fig. 8 over the published domain",
                ),
                AcceptanceCriterion(
                    f"fig8_{int(gpp_percent)}pct_extrap_hook_count",
                    float(hook_count),
                    0.0,
                    "<=",
                    "shape",
                    "number of >0.01 per mil positive D17O increases from 294-50000 ppm",
                ),
            ]
        )
        for pco2, value in zip(extrap_grid, model_extrap):
            rows.append(
                {
                    "figure": "fig8_extrapolation",
                    "pco2_ppm": float(pco2),
                    "gpp_percent": gpp_percent,
                    "model": float(value),
                    "young": np.nan,
                    "residual": np.nan,
                    "slope_corr_published": corr,
                    "max_hook_permil": max_hook,
                }
            )

    spacing_violations = 0
    for pco2 in extrap_grid:
        d50 = model_d17o(preset, float(pco2), 50.0)
        d100 = model_d17o(preset, float(pco2), 100.0)
        if d50 > d100 + hook_tolerance:
            spacing_violations += 1
    criteria.append(
        AcceptanceCriterion(
            "fig8_50pct_below_100pct_spacing_violations",
            float(spacing_violations),
            0.0,
            "<=",
            "shape",
            "50% GPP curve should remain lower/more negative than 100% GPP in the inspected domain",
        )
    )
    shape_failures = sum(1 for item in criteria if not item.passed)
    criteria.append(
        AcceptanceCriterion(
            "shape_first_model_acceptance",
            float(shape_failures),
            0.0,
            "<=",
            "acceptance_policy",
            "model cannot be accepted for extrapolating pCO2/GPP behavior if any Fig. 8 shape guard fails",
        )
    )
    return criteria, rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_criteria_csv(criteria: list[AcceptanceCriterion], path: Path) -> None:
    rows = [
        {
            "key": item.key,
            "group": item.group,
            "value": item.value,
            "relation": item.relation,
            "limit": item.limit,
            "passed": item.passed,
            "note": item.note,
        }
        for item in criteria
    ]
    write_csv(rows, path)


def plot_fig8_shape(rows: list[dict], path: Path, *, preset: str) -> None:
    fig_rows = [row for row in rows if row["figure"] == "fig8"]
    extrap_rows = [row for row in rows if row["figure"] == "fig8_extrapolation"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)
    colors = {50.0: "#7b6aa6", 100.0: "#2d7f72"}
    for gpp_percent in (50.0, 100.0):
        subset = [row for row in fig_rows if float(row["gpp_percent"]) == gpp_percent]
        axes[0].plot(
            [row["pco2_ppm"] for row in subset],
            [row["young"] for row in subset],
            color=colors[gpp_percent],
            linestyle="--",
            linewidth=1.4,
            label=f"Young {gpp_percent:.0f}% GPP",
        )
        axes[0].plot(
            [row["pco2_ppm"] for row in subset],
            [row["model"] for row in subset],
            color=colors[gpp_percent],
            linewidth=2.0,
            label=f"model {gpp_percent:.0f}% GPP",
        )
        ex = [row for row in extrap_rows if float(row["gpp_percent"]) == gpp_percent]
        axes[1].plot(
            [row["pco2_ppm"] for row in ex],
            [row["model"] for row in ex],
            color=colors[gpp_percent],
            linewidth=2.0,
            marker="o",
            label=f"model {gpp_percent:.0f}% GPP",
        )
    for ax in axes:
        ax.set_xlabel(r"pCO$_2$ (ppm)")
        ax.set_ylabel(r"O$_2$ $\Delta'^{17}$O (per mil)")
        ax.grid(True, color="#dddddd", linewidth=0.8)
        ax.legend(frameon=False)
    axes[0].set_title("Published Fig. 8 Domain")
    axes[1].set_title("Extrapolation Shape Check")
    fig.suptitle(preset, fontsize=11)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".svg"))


def write_markdown(criteria: list[AcceptanceCriterion], path: Path, *, preset: str, output_prefix: str) -> None:
    by_group: dict[str, list[AcceptanceCriterion]] = {}
    for item in criteria:
        by_group.setdefault(item.group, []).append(item)
    passed = sum(item.passed for item in criteria)
    lines = [
        "# Young Acceptance Gate",
        "",
        f"Preset: `{preset}`. Processed CO2 scoring mode: `{PROCESSED_MODE}`.",
        "",
        f"Passed `{passed}` of `{len(criteria)}` criteria.",
        "",
        "| group | criterion | value | requirement | pass | note |",
        "|---|---|---:|---:|---|---|",
    ]
    for group in sorted(by_group):
        for item in by_group[group]:
            status = "yes" if item.passed else "no"
            lines.append(
                f"| {group} | {item.key} | {item.value:.5g} | "
                f"{item.relation} {item.limit:.5g} | {status} | {item.note} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Table/text/figure residuals test whether the model reproduces the published Young constraints.",
            "- Dynamic Fig. 9/Fig. 10 checks are included for the current Young-like O2 architecture.",
            "- Shape criteria are acceptance guards, not cosmetic diagnostics. A term that fixes one point but creates wrong curvature, hooks, or curve crossing is rejected as a physical model function.",
            "- Failed shape criteria mean the branch may still be useful as a diagnostic locator for missing behavior, but it cannot be used for extrapolated pCO2/GPP/pO2 scenarios or as the publication model.",
            "",
            f"Plot: `outputs/{output_prefix}_fig8_shape.png`",
            f"Criteria: `outputs/{output_prefix}_criteria.csv`",
            f"Point data: `outputs/{output_prefix}_points.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_name(preset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", preset)


def run_gate(preset: str) -> dict:
    output_prefix = f"young_acceptance_gate_{safe_name(preset)}"
    table = table_criteria(preset)
    fig_criteria, fig_rows = fig_residual_criteria(preset)
    text = text_criteria(preset)
    dynamics = dynamic_criteria(preset)
    shape, shape_rows = shape_criteria(preset)
    criteria = [*table, *fig_criteria, *text, *dynamics, *shape]
    point_rows = [
        *fig_rows,
        *[
            {
                **row,
                "young": "" if np.isnan(row["young"]) else row["young"],
                "residual": "" if np.isnan(row["residual"]) else row["residual"],
            }
            for row in shape_rows
        ],
    ]
    write_criteria_csv(criteria, OUTPUTS / f"{output_prefix}_criteria.csv")
    write_csv(point_rows, OUTPUTS / f"{output_prefix}_points.csv")
    plot_fig8_shape(point_rows, OUTPUTS / f"{output_prefix}_fig8_shape", preset=preset)
    write_markdown(criteria, HERE / f"{output_prefix}.md", preset=preset, output_prefix=output_prefix)
    if preset == CURRENT_YOUNG_REPRODUCTION_PRESET:
        write_markdown(criteria, HERE / "young_acceptance_gate.md", preset=preset, output_prefix=output_prefix)
    return {
        "preset": preset,
        "passed": sum(item.passed for item in criteria),
        "total": len(criteria),
        "failed": len([item for item in criteria if not item.passed]),
        "markdown": HERE / f"{output_prefix}.md",
        "criteria": OUTPUTS / f"{output_prefix}_criteria.csv",
        "plot": OUTPUTS / f"{output_prefix}_fig8_shape.png",
    }


def write_summary(results: list[dict], path: Path) -> None:
    lines = [
        "# Young Acceptance Gate Summary",
        "",
        "| preset | passed | failed | report | plot |",
        "|---|---:|---:|---|---|",
    ]
    for result in results:
        report = Path(result["markdown"]).name
        plot = f"outputs/{Path(result['plot']).name}"
        lines.append(
            f"| `{result['preset']}` | {result['passed']}/{result['total']} | {result['failed']} | "
            f"`{report}` | `{plot}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_presets(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    return values or DEFAULT_PRESETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--presets",
        default=",".join(DEFAULT_PRESETS),
        help="Comma-separated scenario presets to validate.",
    )
    parser.add_argument(
        "--compare-current",
        action="store_true",
        help="Validate instantaneous v2 and the tau=30 access-reservoir candidate.",
    )
    parser.add_argument(
        "--compare-access",
        action="store_true",
        help="Validate instantaneous v2 plus tau=1 and tau=30 access-reservoir candidates.",
    )
    args = parser.parse_args()

    if args.compare_access:
        presets = (YOUNG_LIKE_V2_NAME, YOUNG_LIKE_V2_ACCESS_RESERVOIR_TAU1_NAME, YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME)
    elif args.compare_current:
        presets = (YOUNG_LIKE_V2_NAME, YOUNG_LIKE_V2_ACCESS_RESERVOIR_NAME)
    else:
        presets = parse_presets(args.presets)
    results = [run_gate(preset) for preset in presets]
    write_summary(results, HERE / "young_acceptance_gate_summary.md")
    for result in results:
        print(
            f"{result['preset']}: passed {result['passed']}/{result['total']} "
            f"criteria; wrote {result['markdown']}"
        )
        print(f"Wrote {result['criteria']}")
        print(f"Wrote {result['plot']}")
    print(f"Wrote {HERE / 'young_acceptance_gate_summary.md'}")


if __name__ == "__main__":
    main()
