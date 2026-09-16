"""Audit remaining Young Table 2 rows not covered by R4/R6/R7 audit."""

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
from dataclasses import dataclass
from pathlib import Path

from young_reactions import REACTION_RECORDS


HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
CSV_OUT = _PROJECT_OUTPUTS / "young_remaining_table2_row_audit.csv"
MD_OUT = HERE / "young_remaining_table2_row_audit.md"


@dataclass(frozen=True)
class ExpectedRow:
    key: str
    printed_reaction: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    rate_rule: str
    footnote_or_text: str
    visual_source: str
    status_note: str = ""


EXPECTED_ROWS = [
    ExpectedRow("R1a", "O2 + hv -> O + O", ("O2_strat",), ("O_strat", "O_strat"), "J_R1a", "footnote a; Eq. 24 photolysis integral", "page_08.png"),
    ExpectedRow("R1b", "O17O + hv -> O + 17O", ("O17O_strat",), ("O_strat", "O17_strat"), "J_R1a", "same as R1a", "page_08.png"),
    ExpectedRow("R1c", "O18O + hv -> O + 18O", ("O18O_strat",), ("O_strat", "O18_strat"), "J_R1a", "same as R1a", "page_08.png"),
    ExpectedRow("R2a", "O + O2 + M -> O3", ("O_strat", "O2_strat"), ("O3_strat",), "k_R2a", "footnote b folds 8.3e17 cm-3 into rate", "page_08.png", "Executable code handles R2a although inventory executable flag is false because of the folded third-body convention."),
    ExpectedRow("R2b", "18O + O2 + M -> 18OOO", ("O18_strat", "O2_strat"), ("OO18O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(18O+O2)) * aMIF", "Eq. 25 plus aMIF", "page_08.png"),
    ExpectedRow("R2c", "17O + O2 + M -> 17OOO", ("O17_strat", "O2_strat"), ("OO17O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(17O+O2)) * aMIF", "Eq. 25 plus aMIF", "page_08.png"),
    ExpectedRow("R2d", "O + O18O + M -> 18OOO", ("O_strat", "O18O_strat"), ("OO18O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(O+O18O)) * aMIF", "Eq. 25 plus aMIF", "page_08.png"),
    ExpectedRow("R2e", "O + O17O + M -> 17OOO", ("O_strat", "O17O_strat"), ("OO17O_strat",), "k_R2a * sqrt(mu(O+O2)/mu(O+O17O)) * aMIF", "Eq. 25 plus aMIF", "page_08.png", "OCR labels the rate line as kR2d; treated as a table/OCR label issue unless visual inspection proves otherwise."),
    ExpectedRow("R3a", "O3 + hv -> O2 + O", ("O3_strat",), ("O2_strat", "O_strat"), "J_R3a", "footnote a; Eq. 24 photolysis integral", "page_08.png"),
    ExpectedRow("R3b", "OO18O + hv -> O2 + 18O", ("OO18O_strat",), ("O2_strat", "O18_strat"), "1/3 * J_R3a", "footnote c", "page_08.png"),
    ExpectedRow("R3c", "OO18O + hv -> O18O + O", ("OO18O_strat",), ("O18O_strat", "O_strat"), "2/3 * J_R3a", "footnote c", "page_08.png"),
    ExpectedRow("R3d", "OO17O + hv -> O2 + 17O", ("OO17O_strat",), ("O2_strat", "O17_strat"), "1/3 * J_R3a", "footnote c", "page_08.png"),
    ExpectedRow("R3e", "OO17O + hv -> O17O + O", ("OO17O_strat",), ("O17O_strat", "O_strat"), "2/3 * J_R3a", "footnote c", "page_08.png"),
    ExpectedRow("R3f", "O3 + hv -> O2 + O(1D)", ("O3_strat",), ("O2_strat", "O1D_strat"), "J_R3f", "footnote a; Eq. 24 photolysis integral", "page_08.png"),
    ExpectedRow("R3g", "OO18O + hv -> O2 + 18O(1D)", ("OO18O_strat",), ("O2_strat", "O18_1D_strat"), "1/3 * J_R3f", "footnote c convention", "page_08.png"),
    ExpectedRow("R3h", "OO18O + hv -> O18O + O(1D)", ("OO18O_strat",), ("O18O_strat", "O1D_strat"), "2/3 * J_R3f", "footnote c convention", "page_08.png"),
    ExpectedRow("R3i", "OO17O + hv -> O2 + 17O(1D)", ("OO17O_strat",), ("O2_strat", "O17_1D_strat"), "1/3 * J_R3f", "footnote c convention", "page_08.png"),
    ExpectedRow("R3j", "OO17O + hv -> O17O + O(1D)", ("OO17O_strat",), ("O17O_strat", "O1D_strat"), "2/3 * J_R3f", "footnote c convention", "page_08.png"),
    ExpectedRow("R5a", "M + O(1D) -> M + O", ("O1D_strat",), ("O_strat",), "k_R5a * nM", "footnote d", "page_08.png/page_09_footnotes_crop_d.png", "Scalar row matches inventory; [M] convention remains unresolved."),
    ExpectedRow("R5b", "M + 18O(1D) -> M + 18O", ("O18_1D_strat",), ("O18_strat",), "k_R5a * nM", "same as R5a", "page_08.png", "Scalar row matches inventory; [M] convention remains unresolved."),
    ExpectedRow("R5c", "M + 17O(1D) -> M + 17O", ("O17_1D_strat",), ("O17_strat",), "k_R5a * nM", "same as R5a", "page_08.png", "Scalar row matches inventory; [M] convention remains unresolved."),
    ExpectedRow("R8a", "CO2 + H2(18)O -> CO18O + H2O", ("CO2_trop",), ("CO18O_trop",), "k_R8b * alpha_CO2_H2O_18 * (1.00525 * R18_SMOW)", "footnote e", "page_09.png"),
    ExpectedRow("R8b", "CO18O + H2O -> CO2 + H2(18)O", ("CO18O_trop",), ("CO2_trop",), "k_R8b", "footnote f; Welp et al. 2011", "page_09.png"),
    ExpectedRow("R8c", "CO2 + H2(17)O -> CO17O + H2O", ("CO2_trop",), ("CO17O_trop",), "k_R8b * alpha_CO2_H2O_18**0.528 * (1.00525**0.528 * R17_SMOW)", "footnote e; equilibrium beta 0.528", "page_09.png"),
    ExpectedRow("R8d", "CO17O + H2O -> CO2 + H2(17)O", ("CO17O_trop",), ("CO2_trop",), "k_R8b", "footnote f; Welp et al. 2011", "page_09.png"),
    ExpectedRow("k_ST_O2", "stratosphere -> troposphere O2 mixing", ("O2_strat",), ("O2_trop",), "k_ST", "footnote h; Appenzeller and Holton 1996", "page_09.png", "Inventory representative row; isotopologues analogous in executable code."),
    ExpectedRow("k_TS_O2", "troposphere -> stratosphere O2 mixing", ("O2_trop",), ("O2_strat",), "k_TS", "continuity from text", "page_09.png", "Inventory representative row; isotopologues analogous in executable code."),
    ExpectedRow("resp_O2", "respiration consumes O2 and produces CO2", ("O2_trop",), ("CO2_trop",), "k_respiration", "Section 3.4; Bender et al. turnover", "page_09.png", "Inventory representative row; isotope-specific executable terms use printed alpha/beta."),
    ExpectedRow("photo_O2", "photosynthesis produces O2", tuple(), ("O2_trop",), "r_p", "Section 3.4; rp balanced against respiration", "page_09.png", "Inventory representative row; isotope-specific executable terms use source-water factors."),
    ExpectedRow("CO2_volcanic", "volcanic CO2 source", tuple(), ("CO2_trop",), "f_volcanic_CO2", "footnote j; Solomon et al. 2007", "page_09.png", "Isotopologues scaled by SMOW ratios per footnote j."),
    ExpectedRow("CO2_ocean_effusion", "ocean CO2 effusion source", tuple(), ("CO2_trop",), "f_ocean_CO2", "footnote j; Solomon et al. 2007", "page_09.png", "Isotopologues scaled by SMOW ratios per footnote j."),
    ExpectedRow("CO2_weathering", "CO2 weathering sink", ("CO2_trop",), tuple(), "k_CO2_weathering", "footnote i; Kump et al. 2000", "page_09.png", "Isotopologues scaled by SMOW ratios per footnote i."),
    ExpectedRow("CO2_ocean_infusion", "ocean CO2 infusion sink", ("CO2_trop",), tuple(), "k_ocean_CO2_infusion", "footnote j; Solomon et al. 2007", "page_09.png"),
    ExpectedRow("O2_weathering", "O2 weathering uptake by geosphere", ("O2_trop",), ("O_geo",), "k_O2_weathering", "footnote g; Lasaga and Ohmoto 2002", "page_09.png", "Printed constant OK; exact geosphere stoichiometry represented by executable code, not fully printed as ODE."),
    ExpectedRow("organic_burial", "effective O2 production by organic burial", ("O_geo",), ("O2_trop",), "k_organic_burial / nO2", "footnote g plus Section 3.6", "page_09.png", "Printed constant OK; full normalization of 1/nO2 feedback is not printed."),
]


def audit_rows() -> list[dict[str, str]]:
    records = {record.key: record for record in REACTION_RECORDS}
    rows: list[dict[str, str]] = []
    for expected in EXPECTED_ROWS:
        record = records.get(expected.key)
        if record is None:
            status = "MISSING"
            actual_reactants = actual_products = actual_rate = ""
        else:
            actual_reactants = " + ".join(record.reactants)
            actual_products = " + ".join(record.products)
            actual_rate = record.rate_rule
            matches = (
                record.reactants == expected.reactants
                and record.products == expected.products
                and record.rate_rule == expected.rate_rule
            )
            status = "MATCH" if matches else "MISMATCH"
        rows.append(
            {
                "row": expected.key,
                "printed_reaction": expected.printed_reaction,
                "expected_reactants": " + ".join(expected.reactants),
                "actual_reactants": actual_reactants,
                "expected_products": " + ".join(expected.products),
                "actual_products": actual_products,
                "expected_rate_rule": expected.rate_rule,
                "actual_rate_rule": actual_rate,
                "footnote_or_text": expected.footnote_or_text,
                "visual_source": expected.visual_source,
                "status": status,
                "status_note": expected.status_note,
            }
        )
    return rows


def main() -> None:
    rows = audit_rows()
    mismatches = sum(row["status"] != "MATCH" for row in rows)

    CSV_OUT.parent.mkdir(exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Young Remaining Table 2 Row Audit",
        "",
        "Rows covered here: R1/R2/R3/R5/R8 and the non-stratospheric inventory rows.",
        "R4/R6/R7 are covered separately in `young_r4_r6_r7_row_audit.md`.",
        "",
        f"Rows checked: {len(rows)}",
        f"Mismatches: {mismatches}",
        "",
        "| row | printed reaction | expected rate rule | footnote/text | code status | note |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {row} | `{printed_reaction}` | `{expected_rate_rule}` | {footnote_or_text} | {status} | {status_note} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Audit Notes",
            "",
            "- R1/R2/R3/R8 row transcription matches the current source inventory.",
            "- R5 row transcription matches, but the `[M]` convention from footnote d remains unresolved.",
            "- The non-stratospheric rows are inventory representatives; Young prints some equations in text rather than all isotopologue ODEs.",
            "- Organic burial remains a source-level gap: Young prints `korg` and states a `1/nO2` feedback, but not the full normalized ODE term.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Rows checked: {len(rows)}")
    print(f"Mismatches: {mismatches}")


if __name__ == "__main__":
    main()
