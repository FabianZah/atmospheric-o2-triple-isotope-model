"""Unified scorecard for the current young-like architecture candidate.

This scorecard deliberately reads the already generated steady and transient
audit outputs. The transient scan is comparatively expensive, and the purpose
here is to summarize the current state without silently retuning anything.
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
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from young_architecture_candidate import (
    CURRENT_ARCHITECTURE_LABEL,
    CURRENT_ARCHITECTURE_NAME,
    PREREQUISITE_OUTPUTS,
    SCORE_THRESHOLDS,
    STEADY_SOURCE_LAW,
    TRANSIENT_LAW,
    missing_prerequisite_outputs,
)


HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
OUTPUTS = _PROJECT_OUTPUTS

STEADY_ROWS = OUTPUTS / "processed_column_three_term_steady_validation_rows.csv"
STEADY_SUMMARY = OUTPUTS / "processed_column_three_term_steady_validation_summary.csv"
TRANSIENT_SUMMARY = OUTPUTS / "three_term_with_finite_exposure_transients_summary.csv"
TRANSIENT_TIMESERIES = OUTPUTS / "three_term_with_finite_exposure_transients_timeseries.csv"
LITERATURE_CONSTRAINT = OUTPUTS / "finite_exposure_literature_constraint.csv"
THRESHOLDS_BY_METRIC = {threshold.metric: threshold for threshold in SCORE_THRESHOLDS}

OUT_ROWS = OUTPUTS / "current_architecture_scorecard_rows.csv"
OUT_SUMMARY = OUTPUTS / "current_architecture_scorecard_summary.csv"
OUT_FIG = OUTPUTS / "current_architecture_scorecard"
OUT_MD = HERE / "current_architecture_scorecard.md"

YOUNG_FIG9_VISUAL_MIN = -0.730
YOUNG_FIG9_FINAL_TEXT = -0.539
YOUNG_FIG9_PEAK_CO2 = 1000.0
YOUNG_FIG10_SHIFT_TARGET = -0.049


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    return float(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def status(value: float, warn: float, fail: float, lower_is_better: bool = True) -> str:
    if lower_is_better:
        if value <= warn:
            return "pass"
        if value <= fail:
            return "warning"
        return "fail"
    if value >= warn:
        return "pass"
    if value >= fail:
        return "warning"
    return "fail"


def threshold_status(metric_name: str, value: float) -> tuple[str, str]:
    threshold = THRESHOLDS_BY_METRIC[metric_name]
    result = status(value, threshold.pass_limit, threshold.warning_limit)
    threshold_text = (
        f"pass <={threshold.pass_limit:g}; "
        f"warning <={threshold.warning_limit:g} {threshold.units}"
    )
    return result, threshold_text


def metric(
    category: str,
    name: str,
    value: float,
    target: float | str,
    units: str,
    threshold: str,
    result: str,
    note: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "metric": name,
        "value": value,
        "target": target,
        "units": units,
        "threshold": threshold,
        "status": result,
        "note": note,
    }


def load_best_transient() -> dict[str, Any]:
    rows = read_csv(TRANSIENT_SUMMARY)
    return min(rows, key=lambda row: as_float(row, "cost"))


def load_literature_ratios() -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_csv(LITERATURE_CONSTRAINT):
        label = row.get("label", "")
        ratio = as_float(row, "ratio")
        if not math.isnan(ratio):
            out[label] = ratio
    return out


def select_best_timeseries(best: dict[str, Any]) -> list[dict[str, Any]]:
    branch = best["branch_group"]
    exchange = as_float(best, "exchange_time_modern_yr")
    residence = as_float(best, "residence_time_yr")
    selected = []
    for row in read_csv(TRANSIENT_TIMESERIES):
        if row["experiment"] != "fig9_half_photosynthesis":
            continue
        if row["branch_group"] != branch:
            continue
        if not math.isclose(as_float(row, "exchange_time_modern_yr"), exchange):
            continue
        if not math.isclose(as_float(row, "residence_time_yr"), residence):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: as_float(row, "time_yr"))


def build_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    steady_summary = read_csv(STEADY_SUMMARY)
    steady_rows = read_csv(STEADY_ROWS)
    best = load_best_transient()
    ratios = load_literature_ratios()
    metrics: list[dict[str, Any]] = []

    summary_by_figure = {row["figure"]: row for row in steady_summary}
    fig7 = summary_by_figure["fig7_digitized"]
    result, threshold_text = threshold_status("Fig. 7 digitized mean abs residual", as_float(fig7, "mean_abs_residual"))
    metrics.append(
        metric(
            "steady",
            "Fig. 7 digitized mean abs residual",
            as_float(fig7, "mean_abs_residual"),
            "<=0.03",
            "permil",
            threshold_text,
            result,
            "Dense digitized pCO2 contours, not only the printed 294 ppm equation.",
        )
    )
    result, threshold_text = threshold_status("Fig. 7 digitized max abs residual", as_float(fig7, "max_abs_residual"))
    metrics.append(
        metric(
            "steady",
            "Fig. 7 digitized max abs residual",
            as_float(fig7, "max_abs_residual"),
            "<=0.08",
            "permil",
            threshold_text,
            result,
            "Shape guardrail across the Young Fig. 7 domain.",
        )
    )

    fig8 = summary_by_figure["fig8_digitized"]
    result, threshold_text = threshold_status("Fig. 8 digitized mean abs residual", as_float(fig8, "mean_abs_residual"))
    metrics.append(
        metric(
            "steady",
            "Fig. 8 digitized mean abs residual",
            as_float(fig8, "mean_abs_residual"),
            "<=0.05",
            "permil",
            threshold_text,
            result,
            "Main steady high-pCO2 benchmark.",
        )
    )
    result, threshold_text = threshold_status("Fig. 8 digitized max abs residual", as_float(fig8, "max_abs_residual"))
    metrics.append(
        metric(
            "steady",
            "Fig. 8 digitized max abs residual",
            as_float(fig8, "max_abs_residual"),
            "<=0.15",
            "permil",
            threshold_text,
            result,
            "Worst point remains the 50% GPP low/mid-pCO2 part of Fig. 8.",
        )
    )
    fig8_50_worst = summary_by_figure["fig8_50pct_worst_point"]
    result, threshold_text = threshold_status("Fig. 8 50% GPP worst residual", as_float(fig8_50_worst, "max_abs_residual"))
    metrics.append(
        metric(
            "steady",
            "Fig. 8 50% GPP worst residual",
            as_float(fig8_50_worst, "max_abs_residual"),
            "<=0.10 preferred",
            "permil",
            threshold_text,
            result,
            f"Worst at pCO2 {as_float(fig8_50_worst, 'pco2_ppm'):.0f} ppm.",
        )
    )

    fig9_min_resid = abs(as_float(best, "fig9_min_D17O_permil") - YOUNG_FIG9_VISUAL_MIN)
    fig9_final_resid = abs(as_float(best, "fig9_final_D17O_permil") - YOUNG_FIG9_FINAL_TEXT)
    fig9_peak_resid = abs(as_float(best, "fig9_peak_pCO2_ppm") - YOUNG_FIG9_PEAK_CO2)
    fig10_shift_resid = abs(as_float(best, "fig10_shift_from_modern_permil") - YOUNG_FIG10_SHIFT_TARGET)
    metrics.extend(
        [
            metric(
                "transient",
                "Fig. 9 visual minimum residual",
                fig9_min_resid,
                YOUNG_FIG9_VISUAL_MIN,
                "permil abs residual",
                threshold_status("Fig. 9 visual minimum residual", fig9_min_resid)[1],
                threshold_status("Fig. 9 visual minimum residual", fig9_min_resid)[0],
                f"Best model minimum {as_float(best, 'fig9_min_D17O_permil'):.4f} per mil at {as_float(best, 'fig9_time_min_yr'):.0f} yr.",
            ),
            metric(
                "transient",
                "Fig. 9 final residual",
                fig9_final_resid,
                YOUNG_FIG9_FINAL_TEXT,
                "permil abs residual",
                threshold_status("Fig. 9 final residual", fig9_final_resid)[1],
                threshold_status("Fig. 9 final residual", fig9_final_resid)[0],
                f"Best model final {as_float(best, 'fig9_final_D17O_permil'):.4f} per mil.",
            ),
            metric(
                "transient",
                "Fig. 9 peak pCO2 residual",
                fig9_peak_resid,
                YOUNG_FIG9_PEAK_CO2,
                "ppm abs residual",
                threshold_status("Fig. 9 peak pCO2 residual", fig9_peak_resid)[1],
                threshold_status("Fig. 9 peak pCO2 residual", fig9_peak_resid)[0],
                f"Best model peak {as_float(best, 'fig9_peak_pCO2_ppm'):.1f} ppm.",
            ),
            metric(
                "transient",
                "Fig. 10 final shift residual",
                fig10_shift_resid,
                YOUNG_FIG10_SHIFT_TARGET,
                "permil abs residual",
                threshold_status("Fig. 10 final shift residual", fig10_shift_resid)[1],
                threshold_status("Fig. 10 final shift residual", fig10_shift_resid)[0],
                f"Best model final shift {as_float(best, 'fig10_shift_from_modern_permil'):+.4f} per mil.",
            ),
        ]
    )

    liang_local = ratios.get("Liang local transport / exchange", math.nan)
    model_ratio = as_float(best, "residence_exchange_ratio")
    ratio_resid = abs(model_ratio - liang_local)
    result, threshold_text = threshold_status("Finite-exposure ratio vs Liang local", ratio_resid)
    metrics.append(
        metric(
            "literature",
            "Finite-exposure ratio vs Liang local",
            ratio_resid,
            liang_local,
            "absolute ratio difference",
            threshold_text,
            result,
            f"Best model ratio {model_ratio:.2f}; Liang local ratio {liang_local:.2f}.",
        )
    )
    metrics.append(
        metric(
            "literature",
            "Peak R7 transfer factor",
            as_float(best, "fig9_peak_R7_factor"),
            "<=1.333",
            "dimensionless",
            "pass <=1.15; warning <=1.333",
            status(as_float(best, "fig9_peak_R7_factor"), 1.15, 1.333),
            "Stays below the simple 4/3 statistical-branching ceiling.",
        )
    )

    return metrics, steady_rows, best, select_best_timeseries(best), ratios


def plot_scorecard(
    metrics: list[dict[str, Any]],
    steady_rows: list[dict[str, Any]],
    best: dict[str, Any],
    timeseries: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    fig7 = [row for row in steady_rows if row["figure"] == "fig7_digitized"]
    sc = axes[0, 0].scatter(
        [as_float(row, "gpp_percent") for row in fig7],
        [as_float(row, "residual_permil") for row in fig7],
        c=np.log10([as_float(row, "pco2_ppm") for row in fig7]),
        cmap="viridis",
        s=18,
    )
    axes[0, 0].axhline(0.0, color="#222222", lw=0.8)
    axes[0, 0].set_xlabel("GPP (% modern)")
    axes[0, 0].set_ylabel("Fig. 7 residual (permil)")
    fig.colorbar(sc, ax=axes[0, 0], label="log10 pCO2")

    colors = {50.0: "#b35806", 100.0: "#2166ac"}
    for gpp in (50.0, 100.0):
        rows = [
            row
            for row in steady_rows
            if row["figure"] == "fig8_digitized" and math.isclose(as_float(row, "gpp_percent"), gpp)
        ]
        rows = sorted(rows, key=lambda row: as_float(row, "pco2_ppm"))
        axes[0, 1].plot([as_float(row, "pco2_ppm") for row in rows], [as_float(row, "young_d17o_permil") for row in rows], "o", color=colors[gpp], ms=4, label=f"Young {gpp:.0f}%")
        axes[0, 1].plot([as_float(row, "pco2_ppm") for row in rows], [as_float(row, "model_d17o_permil") for row in rows], "-", color=colors[gpp], lw=2, label=f"model {gpp:.0f}%")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("pCO2 (ppm)")
    axes[0, 1].set_ylabel(r"O$_2$ $\Delta'^{17}$O (permil)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(
        [as_float(row, "time_yr") for row in timeseries],
        [as_float(row, "O2_trop_D17O_permil") for row in timeseries],
        color="#542788",
        lw=2.2,
        label="current architecture",
    )
    axes[1, 0].axhline(YOUNG_FIG9_VISUAL_MIN, color="#8c2d04", ls="--", lw=1.0, label="Young visual min")
    axes[1, 0].axhline(YOUNG_FIG9_FINAL_TEXT, color="#555555", ls=":", lw=1.0, label="Young final text")
    axes[1, 0].set_xlabel("time after rp/2 (yr)")
    axes[1, 0].set_ylabel(r"O$_2$ $\Delta'^{17}$O (permil)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    status_score = {"pass": 1.0, "warning": 0.5, "fail": 0.0}
    labels = [row["metric"].replace(" residual", "") for row in metrics]
    values = [status_score[row["status"]] for row in metrics]
    bar_colors = [{"pass": "#1b7837", "warning": "#d8b365", "fail": "#b2182b"}[row["status"]] for row in metrics]
    y = np.arange(len(metrics))
    axes[1, 1].barh(y, values, color=bar_colors)
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels(labels, fontsize=7)
    axes[1, 1].set_xlim(0.0, 1.05)
    axes[1, 1].set_xlabel("diagnostic status")
    axes[1, 1].set_xticks([0.0, 0.5, 1.0])
    axes[1, 1].set_xticklabels(["fail", "warning", "pass"])
    axes[1, 1].invert_yaxis()

    for ax in axes.flat:
        ax.grid(True, color="#dddddd", lw=0.6)
    fig.suptitle("Current young-like architecture scorecard", fontsize=13)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".svg"))


def write_markdown(metrics: list[dict[str, Any]], best: dict[str, Any], ratios: dict[str, float], path: Path) -> None:
    counts = {name: sum(1 for row in metrics if row["status"] == name) for name in ("pass", "warning", "fail")}
    lines = [
        "# Current Young-Like Architecture Scorecard",
        "",
        f"Architecture name: `{CURRENT_ARCHITECTURE_NAME}`",
        "",
        "The same name is now available as a `model_scenarios.py` preset with the steady source law integrated into `calibrated_model.build_reactions()`.",
        "",
        "Candidate architecture:",
        "",
        f"- label: {CURRENT_ARCHITECTURE_LABEL};",
        f"- steady Fig. 7/Fig. 8: three-term processed-column source/export law (`{STEADY_SOURCE_LAW.processed_signature}`, half `{STEADY_SOURCE_LAW.gate_half_ppm:g} ppm`, hill `{STEADY_SOURCE_LAW.gate_hill:g}`);",
        f"- transient Fig. 9/Fig. 10: transient-only finite-exposure R7 `{TRANSIENT_LAW.branch_group}` transfer (residence/exchange `{TRANSIENT_LAW.residence_exchange_ratio:g}`);",
        "- finite-exposure ratio constrained against Liang-style local transport/exchange time.",
        "",
        f"Overall diagnostic count: `{counts['pass']}` pass, `{counts['warning']}` warning, `{counts['fail']}` fail.",
        "",
        "## Metrics",
        "",
        "| category | metric | value | target | status | note |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in metrics:
        value = row["value"]
        value_text = f"{value:.5g}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {row['category']} | {row['metric']} | {value_text} | {row['target']} | "
            f"{row['status']} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Best Transient Row",
            "",
            "| field | value |",
            "|---|---:|",
            f"| branch group | `{best['branch_group']}` |",
            f"| exchange time | {as_float(best, 'exchange_time_modern_yr'):.3g} yr |",
            f"| residence time | {as_float(best, 'residence_time_yr'):.3g} yr |",
            f"| residence/exchange ratio | {as_float(best, 'residence_exchange_ratio'):.3f} |",
            f"| Fig. 9 minimum | {as_float(best, 'fig9_min_D17O_permil'):.4f} per mil |",
            f"| Fig. 9 final | {as_float(best, 'fig9_final_D17O_permil'):.4f} per mil |",
            f"| Fig. 9 peak pCO2 | {as_float(best, 'fig9_peak_pCO2_ppm'):.1f} ppm |",
            f"| Fig. 10 shift | {as_float(best, 'fig10_shift_from_modern_permil'):+.4f} per mil |",
            f"| peak R7 factor | {as_float(best, 'fig9_peak_R7_factor'):.3f} |",
            "",
            "## Literature Exposure Context",
            "",
            "| constraint | ratio |",
            "|---|---:|",
        ]
    )
    for label, ratio in sorted(ratios.items()):
        lines.append(f"| {label} | {ratio:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current architecture is now a coherent candidate rather than a pile of unrelated patches: the steady and transient mechanisms are separated, and the transient parameter lands near an independent Liang exposure ratio.",
            "- The main remaining steady warning is the 50% GPP Fig. 8 low/mid-pCO2 residual. This is small enough to continue, but large enough that it should remain visible.",
            "- This scorecard does not certify a final model. It defines the current best young-like branch and the exact residuals it must improve or preserve.",
            "",
            f"Figure: `outputs/{path.stem}.png`",
            f"Rows: `outputs/{OUT_ROWS.name}`",
            f"Summary: `outputs/{OUT_SUMMARY.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = list(missing_prerequisite_outputs())
    if missing:
        raise FileNotFoundError("Missing prerequisite audit output(s): " + ", ".join(str(path) for path in missing))

    metrics, steady_rows, best, timeseries, ratios = build_metrics()
    write_csv(metrics, OUT_ROWS)
    write_csv(
        [
            {
                "pass": sum(1 for row in metrics if row["status"] == "pass"),
                "warning": sum(1 for row in metrics if row["status"] == "warning"),
                "fail": sum(1 for row in metrics if row["status"] == "fail"),
                "architecture_name": CURRENT_ARCHITECTURE_NAME,
                "architecture_label": CURRENT_ARCHITECTURE_LABEL,
                "best_branch_group": best["branch_group"],
                "best_residence_exchange_ratio": as_float(best, "residence_exchange_ratio"),
                "best_fig9_min_D17O_permil": as_float(best, "fig9_min_D17O_permil"),
                "best_fig9_final_D17O_permil": as_float(best, "fig9_final_D17O_permil"),
                "best_fig10_shift_permil": as_float(best, "fig10_shift_from_modern_permil"),
            }
        ],
        OUT_SUMMARY,
    )
    plot_scorecard(metrics, steady_rows, best, timeseries, OUT_FIG)
    write_markdown(metrics, best, ratios, OUT_MD)
    print("Current architecture scorecard")
    print("------------------------------")
    for state in ("pass", "warning", "fail"):
        print(f"{state}: {sum(1 for row in metrics if row['status'] == state)}")
    print(
        "best transient: "
        f"{best['branch_group']} ratio={as_float(best, 'residence_exchange_ratio'):.2f} "
        f"Fig9 min={as_float(best, 'fig9_min_D17O_permil'):+.4f} "
        f"Fig9 final={as_float(best, 'fig9_final_D17O_permil'):+.4f}"
    )
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FIG.with_suffix('.png')}")


if __name__ == "__main__":
    main()
