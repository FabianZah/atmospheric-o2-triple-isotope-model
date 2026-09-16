"""Audit high-risk Young Table 2 reaction rows against the code inventory.

The purpose is transcription QA. It records whether every R4/R6/R7 row has a
matching `ReactionRecord` with the printed stoichiometry and rate rule. It does
not decide whether Young's hidden ODE implementation used extra conventions.
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
from dataclasses import dataclass
from pathlib import Path

from young_reactions import REACTION_RECORDS


HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (p for p in (HERE, *HERE.parents) if (p / ".project-root").exists()),
    HERE,
)
_PROJECT_OUTPUTS = _PROJECT_ROOT / "outputs"
CSV_OUT = _PROJECT_OUTPUTS / "young_r4_r6_r7_row_audit.csv"
MD_OUT = HERE / "young_r4_r6_r7_row_audit.md"


@dataclass(frozen=True)
class ExpectedRow:
    key: str
    printed_reaction: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    rate_rule: str
    footnote_or_text: str
    visual_source: str
    note: str = ""


EXPECTED_ROWS = [
    ExpectedRow("R4a", "O3 + O -> O2 + O2", ("O3_strat", "O_strat"), ("O2_strat", "O2_strat"), "k_R4a", "Table 2; footnote a source convention", "page_08.png"),
    ExpectedRow("R4b", "OO18O + O -> O2 + O18O", ("OO18O_strat", "O_strat"), ("O2_strat", "O18O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(OO18O+O))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4c", "OO17O + O -> O2 + O17O", ("OO17O_strat", "O_strat"), ("O2_strat", "O17O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(OO17O+O))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4d", "O3 + 18O -> O2 + O18O", ("O3_strat", "O18_strat"), ("O2_strat", "O18O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(O3+18O))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4e", "O3 + 17O -> O2 + O17O", ("O3_strat", "O17_strat"), ("O2_strat", "O17O_strat"), "k_R4a * sqrt(mu(O3+O)/mu(O3+17O))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4f", "O3 + O(1D) -> O2 + O2", ("O3_strat", "O1D_strat"), ("O2_strat", "O2_strat"), "k_R4f", "Table 2; footnote a source convention", "page_08.png"),
    ExpectedRow("R4g", "OO18O + O(1D) -> O2 + O18O", ("OO18O_strat", "O1D_strat"), ("O2_strat", "O18O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(OO18O+O1D))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4h", "OO17O + O(1D) -> O2 + O17O", ("OO17O_strat", "O1D_strat"), ("O2_strat", "O17O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(OO17O+O1D))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4i", "O3 + 18O(1D) -> O2 + O18O", ("O3_strat", "O18_1D_strat"), ("O2_strat", "O18O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(O3+18O1D))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4j", "O3 + 17O(1D) -> O2 + O17O", ("O3_strat", "O17_1D_strat"), ("O2_strat", "O17O_strat"), "k_R4f * sqrt(mu(O3+O1D)/mu(O3+17O1D))", "Eq. 25", "page_08.png"),
    ExpectedRow("R4k", "O3 + O(1D) -> O2 + O + O", ("O3_strat", "O1D_strat"), ("O2_strat", "O_strat", "O_strat"), "k_R4f", "Table 2", "page_08.png"),
    ExpectedRow("R4l", "OO18O + O(1D) -> O2 + O + 18O", ("OO18O_strat", "O1D_strat"), ("O2_strat", "O_strat", "O18_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO18O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4m", "OO18O + O(1D) -> O18O + O + O", ("OO18O_strat", "O1D_strat"), ("O18O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO18O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4n", "OO17O + O(1D) -> O2 + O + 17O", ("OO17O_strat", "O1D_strat"), ("O2_strat", "O_strat", "O17_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO17O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4o", "OO17O + O(1D) -> O17O + O + O", ("OO17O_strat", "O1D_strat"), ("O17O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(OO17O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4p", "O3 + 18O(1D) -> O2 + O + 18O", ("O3_strat", "O18_1D_strat"), ("O2_strat", "O_strat", "O18_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+18O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4q", "O3 + 18O(1D) -> O18O + O + O", ("O3_strat", "O18_1D_strat"), ("O18O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+18O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4r", "O3 + 17O(1D) -> O2 + O + 17O", ("O3_strat", "O17_1D_strat"), ("O2_strat", "O_strat", "O17_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+17O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R4s", "O3 + 17O(1D) -> O17O + O + O", ("O3_strat", "O17_1D_strat"), ("O17O_strat", "O_strat", "O_strat"), "1/2 * k_R4f * sqrt(mu(O3+O1D)/mu(O3+17O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png"),
    ExpectedRow("R6a", "O2 + 18O -> O18O + O", ("O2_strat", "O18_strat"), ("O18O_strat", "O_strat"), "k_R6 * sqrt(mu(O2+O)/mu(O2+18O))", "Eq. 25; kR6 = 2.0e-16", "page_08.png/page_09.png"),
    ExpectedRow("R6b", "O18O + O -> O2 + 18O", ("O18O_strat", "O_strat"), ("O2_strat", "O18_strat"), "1/2 * k_R6 * sqrt(mu(O2+O)/mu(O18O+O))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R6c", "O2 + 17O -> O17O + O", ("O2_strat", "O17_strat"), ("O17O_strat", "O_strat"), "k_R6 * sqrt(mu(O2+O)/mu(O2+17O))", "Eq. 25", "page_09.png"),
    ExpectedRow("R6d", "O17O + O -> O2 + 17O", ("O17O_strat", "O_strat"), ("O2_strat", "O17_strat"), "1/2 * k_R6 * sqrt(mu(O2+O)/mu(O17O+O))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7a", "CO2 + O(1D) -> CO2 + O", ("CO2_strat", "O1D_strat"), ("CO2_strat", "O_strat"), "k_R7a", "Table 2; kR7a = 4.46e-11", "page_08.png/page_09.png"),
    ExpectedRow("R7b", "CO2 + 18O(1D) -> CO2 + 18O", ("CO2_strat", "O18_1D_strat"), ("CO2_strat", "O18_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+18O1D))", "Eq. 25 plus printed 1/2 branch", "page_08.png/page_09.png"),
    ExpectedRow("R7c", "CO2 + 18O(1D) -> CO18O + O", ("CO2_strat", "O18_1D_strat"), ("CO18O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+18O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7d", "CO18O + O(1D) -> CO2 + 18O", ("CO18O_strat", "O1D_strat"), ("CO2_strat", "O18_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO18O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7e", "CO18O + O(1D) -> CO18O + O", ("CO18O_strat", "O1D_strat"), ("CO18O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO18O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7f", "CO2 + 17O(1D) -> CO2 + 17O", ("CO2_strat", "O17_1D_strat"), ("CO2_strat", "O17_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+17O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7g", "CO2 + 17O(1D) -> CO17O + O", ("CO2_strat", "O17_1D_strat"), ("CO17O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO2+17O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7h", "CO17O + O(1D) -> CO2 + 17O", ("CO17O_strat", "O1D_strat"), ("CO2_strat", "O17_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO17O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
    ExpectedRow("R7i", "CO17O + O(1D) -> CO17O + O", ("CO17O_strat", "O1D_strat"), ("CO17O_strat", "O_strat"), "1/2 * k_R7a * sqrt(mu(CO2+O1D)/mu(CO17O+O1D))", "Eq. 25 plus printed 1/2 branch", "page_09.png"),
]


def main() -> None:
    records = {record.key: record for record in REACTION_RECORDS}
    rows: list[dict[str, str]] = []
    mismatches = 0
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
        if status != "MATCH":
            mismatches += 1
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
                "note": expected.note,
            }
        )

    CSV_OUT.parent.mkdir(exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Young R4/R6/R7 Row Audit",
        "",
        "This is a source-transcription audit against Young Table 2 and the current `ReactionRecord` inventory.",
        "It does not validate whether the hidden Young DLSODE implementation had additional conventions.",
        "",
        f"Rows checked: {len(rows)}",
        f"Mismatches: {mismatches}",
        "",
        "| row | printed reaction | expected rate rule | footnote/text | code status | visual source |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {row} | `{printed_reaction}` | `{expected_rate_rule}` | {footnote_or_text} | {status} | `{visual_source}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Audit Notes",
            "",
            "- The R4/R6/R7 inventory matches the printed stoichiometry and rate-rule strings currently extracted from Young Table 2.",
            "- This does not resolve whether the reduced-mass species labels encode exactly the same mass convention Young used internally.",
            "- R7 remains behavior-flagged because the printed row transcription can be correct while the overall O(1D)-CO2 transfer behavior still differs from Young's numerical output.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Rows checked: {len(rows)}")
    print(f"Mismatches: {mismatches}")


if __name__ == "__main__":
    main()
