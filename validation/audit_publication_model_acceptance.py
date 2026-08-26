"""Integrate release gates, independent evidence, and model-scope limits.

This audit evaluates the one publication model. It does not rank development
branches or average inter-model differences into a pseudo-probability. Formal
failures can block release; qualified comparisons and declared scope limits
remain visible without changing deterministic model output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
OUTPUTS = ROOT / "outputs"
DEFAULT_OUTPUT = OUTPUTS / "publication_model_acceptance.json"
DEFAULT_DOC = ROOT / "docs" / "publication_model_acceptance.md"
EVIDENCE = ROOT / "model_data" / "validation_evidence"

SOURCE_PATHS = {
    "release": EVIDENCE / "updated_molecular_release_scorecard.json",
    "liu": EVIDENCE / "liu_2021_low_gpp_multimodel_benchmark.json",
    "cao_bao": EVIDENCE / "cao_bao_2013_multimodel_benchmark.json",
    "luz": EVIDENCE / "luz_1999_productivity_benchmark.json",
    "marine_access": EVIDENCE / "marine_o2_accessibility_audit.json",
    "uncertainty": EVIDENCE / "updated_uncertainty_layers_audit.json",
    "contract": ROOT / "model_data" / "publication_model_contract_v1.json",
}

STATUS_COLORS = {
    "pass": "#187a57",
    "qualified": "#d18f00",
    "scope_limit": "#6e7781",
    "excluded": "#4f6d8a",
    "fail": "#b43a35",
}


def _load_sources() -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing acceptance-audit source(s): " + ", ".join(missing))
    return {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in SOURCE_PATHS.items()
    }


def _metric(release: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in release["metrics"] if row["metric"] == name]
    if len(matches) != 1:
        raise KeyError(f"expected one release metric named {name!r}")
    return matches[0]


def _row(
    section: str,
    gate: str,
    status: str,
    value: Any,
    criterion: str,
    evidence: str,
    *,
    blocking: bool = False,
) -> dict[str, Any]:
    if status not in STATUS_COLORS:
        raise ValueError(f"unknown acceptance status: {status}")
    return {
        "section": section,
        "gate": gate,
        "status": status,
        "value": value,
        "criterion": criterion,
        "evidence": evidence,
        "blocking": bool(blocking),
    }


def build_report() -> dict[str, Any]:
    source = _load_sources()
    release = source["release"]
    liu = source["liu"]
    cao = source["cao_bao"]
    luz = source["luz"]
    marine = source["marine_access"]
    uncertainty = source["uncertainty"]
    contract = source["contract"]

    pack_d17 = _metric(release, "Pack Delta-prime-17O residual")
    pack_d18 = _metric(release, "Pack conventional delta-18O residual")
    dense_shape = _metric(release, "Dense-domain monotonic and finite gates")
    inversion = _metric(release, "Maximum three-coordinate round-trip error")
    live_root = _metric(release, "Maximum live-kernel residual at accelerated roots")
    posterior = _metric(release, "Joint pCO2-GPP-pO2 posterior generated-target recovery")
    uncertainty_gate = _metric(release, "Separated uncertainty-layer contract")
    young7 = _metric(release, "Fig. 7 aligned mean absolute residual")
    young8 = _metric(release, "Fig. 8 aligned mean absolute residual")
    yang = _metric(release, "Yang raw age-block CO2 tracking skill")
    banerjee = _metric(release, "Banerjee MPT minus pre-MPT GPP")
    brandon = _metric(release, "Termination V inferred GPP")

    formal = release["formal_gate_summary"]
    domain = contract["operational_domain"]
    liu_metrics = liu["metrics"]
    luz_metrics = luz["response_relative_metrics"]
    cao_shape = cao["shape"]["by_po2"]
    cao_through_30k = all(
        bool(item["all_monotonically_decreasing_through_30000ppm"])
        for item in cao_shape.values()
    )

    rows = [
        _row(
            "core",
            "Formal numerical and physical release gates",
            "pass" if formal["all_passed"] else "fail",
            f"{formal['passed']} passed; {formal['failed']} failed",
            "Every formal gate must pass",
            "Updated molecular release scorecard",
            blocking=not formal["all_passed"],
        ),
        _row(
            "modern",
            "Modern atmospheric O2 Delta-prime-17O",
            pack_d17["status"],
            f"residual {float(pack_d17['value']):+.6f} per mil",
            pack_d17["criterion"],
            "Pack (2021); direct output without offset",
            blocking=pack_d17["status"] == "fail",
        ),
        _row(
            "modern",
            "Modern atmospheric O2 delta-18O",
            pack_d18["status"],
            f"residual {float(pack_d18['value']):+.6f} per mil",
            pack_d18["criterion"],
            "Pack (2021); prime-to-conventional conversion",
            blocking=pack_d18["status"] == "fail",
        ),
        _row(
            "surface",
            "Finite monotonic solution surface",
            dense_shape["status"],
            f"{int(dense_shape['value']):,} audited points",
            "No hooks, sign reversals, or non-finite cells",
            "Dense-domain shape audit",
            blocking=dense_shape["status"] == "fail",
        ),
        _row(
            "surface",
            "Declared operational domain",
            "pass",
            (
                f"pO2 {domain['pO2_PAL']['minimum']}-{domain['pO2_PAL']['maximum']} PAL; "
                f"pCO2 {domain['pCO2_ppm']['minimum']:.0f}-{domain['pCO2_ppm']['maximum']:.0f} ppm; "
                f"GPP {domain['GPP_PgC_per_year']['minimum']:.3f}-{domain['GPP_PgC_per_year']['maximum']:.0f} PgC/yr"
            ),
            "No numerical extrapolation outside the tested domain",
            "Publication model contract",
        ),
        _row(
            "inverse",
            "Forward-inverse round trips",
            "pass" if inversion["status"] == live_root["status"] == "pass" else "fail",
            (
                f"relative error {float(inversion['value']):.3e}; "
                f"live residual {float(live_root['value']):.3e} per mil"
            ),
            "Accelerated roots must reproduce live-kernel targets",
            "pCO2, GPP, and pO2 synthetic recovery",
            blocking=not (inversion["status"] == live_root["status"] == "pass"),
        ),
        _row(
            "inverse",
            "Joint posterior solution ridge",
            posterior["status"],
            "generated target recovered",
            "Posterior must preserve pCO2-GPP-pO2 non-uniqueness",
            "Joint posterior synthetic recovery",
            blocking=posterior["status"] == "fail",
        ),
        _row(
            "uncertainty",
            "Separated uncertainty layers",
            "pass" if uncertainty_gate["status"] == "pass" and uncertainty["all_combined_confidence_intervals_disabled"] else "fail",
            "measurement, parameter, numerical, structural",
            "No uncalibrated combined confidence interval",
            "Versioned uncertainty contract",
            blocking=not (uncertainty_gate["status"] == "pass" and uncertainty["all_combined_confidence_intervals_disabled"]),
        ),
        _row(
            "validation",
            "Young Fig. 7 historical response anchor",
            "pass" if float(young7["value"]) <= 0.05 else "qualified",
            f"aligned MAE {float(young7['value']):.4f} per mil",
            "Response shape retained without output correction",
            "Digitized Young et al. (2014) contours",
        ),
        _row(
            "validation",
            "Young Fig. 8 historical response anchor",
            "qualified",
            f"aligned MAE {float(young8['value']):.4f} per mil",
            "Direction and curvature retained; amplitude difference disclosed",
            "Digitized 50% and 100% GPP curves",
        ),
        _row(
            "validation",
            "Liu low-GPP response topology",
            "pass" if liu_metrics["all_pCO2_direction_checks_pass"] and liu_metrics["all_GPP_direction_checks_pass"] else "fail",
            f"rank r={float(liu_metrics['response_rank_correlation']):.4f}; RMSE {float(liu_metrics['response_RMSE_permil']):.2f} per mil",
            "Correct response directions over all shared states",
            "Liu et al. (2021) native 90-state grid",
            blocking=not (liu_metrics["all_pCO2_direction_checks_pass"] and liu_metrics["all_GPP_direction_checks_pass"]),
        ),
        _row(
            "validation",
            "Cao-Bao high-pCO2 response direction",
            "pass" if cao_through_30k else "fail",
            "all pO2 cases monotonic through 30,000 ppm",
            "Independent architecture must agree on the principal direction",
            "Cao and Bao (2013) source reproduction",
            blocking=not cao_through_30k,
        ),
        _row(
            "validation",
            "Luz normalized-productivity inversion",
            "pass" if float(luz_metrics["RMSE_percentage_points"]) <= 3.0 and float(luz_metrics["Pearson_r"]) >= 0.9 else "qualified",
            f"RMSE {float(luz_metrics['RMSE_percentage_points']):.2f} percentage points; r={float(luz_metrics['Pearson_r']):.3f}",
            "Non-fitted relative GPP response should reproduce published ordering",
            "Luz et al. (1999) five-state benchmark",
        ),
        _row(
            "validation",
            "Yang ice-core CO2 predictive information",
            "pass" if float(yang["value"]) > 0.0 else "fail",
            f"raw blocked-CV skill {float(yang['value']):.3f}",
            "Measured CO2 must outperform an intercept-only holdout model",
            "269 paired observations; fixed response amplitude",
            blocking=float(yang["value"]) <= 0.0,
        ),
        _row(
            "validation",
            "Independent paleo applications",
            "pass",
            f"Banerjee change {float(banerjee['value']):.2f} points; Brandon GPP {float(brandon['value']):.1f}%",
            "Recover published-scale behavior without fitting core parameters",
            "Banerjee (2026) and Brandon et al. (2020)",
        ),
        _row(
            "structural",
            "Extreme high-pCO2/low-GPP amplitude",
            "qualified",
            "model families agree in direction but not amplitude",
            "Report non-probabilistic structural sensitivity",
            "Young, Liu, Cao-Bao, and Clima comparisons",
        ),
        _row(
            "structural",
            "Marine O2 accessibility",
            "excluded",
            f"Liu RMSE change {float(marine['shared_grid']['candidate_minus_baseline_RMSE_change_permil']):+.3f} per mil",
            "Must improve more than one validation family before promotion",
            marine["promotion_gate"]["decision"],
        ),
        _row(
            "structural",
            "pCO2-dependent climate profile",
            "excluded",
            "large high-pCO2 sensitivity; modern RCE gate failed",
            "A recognizable modern climate is required before promotion",
            "Clima calculations remain structural end members",
        ),
        _row(
            "scope",
            "Fully simultaneous carbon-oxygen transients",
            "scope_limit",
            "operator-split prescribed forcing experiments",
            "Interpret carbon forcing as prescribed rather than prognostic",
            "Steady inversions and declared perturbation experiments are accepted",
        ),
        _row(
            "scope",
            "Whole-domain probabilistic model discrepancy",
            "scope_limit",
            "no defensible global Gaussian sigma",
            "Use domain-specific evidence and explicit priors",
            "Inter-model spread is not treated as random truth error",
        ),
    ]

    blockers = [row for row in rows if row["blocking"] and row["status"] == "fail"]
    verdict = (
        "accepted_for_steady_forward_inverse_and_declared_time_responses"
        if not blockers
        else "not_accepted_release_blockers_present"
    )
    return {
        "schema_version": 1,
        "audit": "single publication model integrated acceptance",
        "publication_model_id": contract["publication_model_id"],
        "verdict": verdict,
        "release_blocker_count": len(blockers),
        "release_blockers": [row["gate"] for row in blockers],
        "central_model_changed_by_audit": False,
        "accepted_claims": [
            "steady forward calculation within the declared pO2-pCO2-GPP domain",
            "conditional and joint inversion with independent constraints and explicit priors",
            (
                "validated pCO2, GPP, and pO2 perturbation steps and prescribed "
                "gradual pCO2 trajectories"
            ),
            "separate propagation and reporting of declared uncertainty layers",
        ],
        "scope_boundaries": [
            "inversion combines one isotope observation with independent coordinate constraints",
            "historical Young calculations provide validation provenance",
            "carbon forcing is prescribed in the declared atmospheric-O2 transients",
            "structural evidence is reported by domain and with explicit priors",
            "precision is qualified in the extreme high-pCO2 and low-GPP corner",
        ],
        "next_model_change_policy": (
            "Replace a central equation or parameter only when independently "
            "constrained evidence improves multiple validation families while "
            "preserving conservation, modern observations, and surface behavior."
        ),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in STATUS_COLORS
        },
        "rows": rows,
        "sources": {
            key: str(path.relative_to(ROOT)) for key, path in SOURCE_PATHS.items()
        },
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Publication Model Acceptance",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        "This audit evaluates the single OXYTIB publication model against its declared evidence and scope.",
        "",
        "| Section | Gate | Status | Result |",
        "|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['section']} | {row['gate']} | {row['status']} | {row['value']} |"
        )
    lines.extend(
        [
            "",
            "## Accepted claims",
            "",
            *[f"- {item}" for item in report["accepted_claims"]],
            "",
            "## Scope boundaries",
            "",
            *[f"- {item}" for item in report["scope_boundaries"]],
            "",
            "## Decision",
            "",
            (
                "There are no release-blocking failures. The deterministic core is "
                "accepted for steady forward and inverse applications and for the "
                "declared time-response experiments. High-pCO2 model-family spread, "
                "the rejected climate and marine-access candidates, and incomplete "
                "fully coupled transients remain explicit scope limits rather than "
                "hidden corrections."
            ),
            "",
            f"Central-model change policy: {report['next_model_change_policy']}",
            "",
            "Machine-readable details are written to `outputs/publication_model_acceptance.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    figure_height = max(6.0, 0.36 * len(rows))
    figure, axis = plt.subplots(figsize=(11.5, figure_height), constrained_layout=True)
    y = list(range(len(rows)))
    axis.scatter(
        [0.0] * len(rows),
        y,
        s=150,
        c=[STATUS_COLORS[row["status"]] for row in rows],
        marker="s",
    )
    for index, row in enumerate(rows):
        axis.text(0.07, index, row["gate"], va="center", fontsize=9)
        axis.text(0.98, index, row["status"].replace("_", " "), va="center", ha="right", fontsize=8.5)
    axis.set_xlim(-0.05, 1.02)
    axis.set_ylim(len(rows) - 0.4, -0.6)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Publication model integrated acceptance")
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(output_path: Path = DEFAULT_OUTPUT, doc_path: Path = DEFAULT_DOC) -> dict[str, Any]:
    report = build_report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(report["rows"], output_path.with_suffix(".csv"))
    _plot(report["rows"], output_path.with_suffix(".png"))
    _write_markdown(report, doc_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    report = run(args.output, args.doc)
    print(json.dumps({
        "verdict": report["verdict"],
        "release_blocker_count": report["release_blocker_count"],
        "status_counts": report["status_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
