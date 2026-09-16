"""Pair published ice-core oxygen-isotope data with measured atmospheric CO2.

The script performs no model fitting. It linearly interpolates the corrected
Antarctic CO2 composite to each Yang et al. (2022) gas age and records the
distance to the bracketing CO2 observations so sparse intervals remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
DEFAULT_YANG = ROOT / "model_data" / "literature" / "yang_2022_pangaea_941483.csv"
DEFAULT_CO2 = ROOT / "model_data" / "literature" / "bereiter_2015_noaa_composite.txt"
DEFAULT_OUTPUT = ROOT / "outputs" / "yang_2022_co2_age_matched.csv"
DEFAULT_REPORT = ROOT / "outputs" / "ice_core_holdout_coverage.json"


@dataclass(frozen=True)
class Co2Observation:
    age_year_bp: float
    co2_ppm: float
    sigma_ppm: float


def load_yang(path: Path = DEFAULT_YANG) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "depth_ice_m",
        "gas_age_ka_bp",
        "cap_delta17_modern_air_relative_ppm",
        "cap_delta17_1sigma_ppm",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Unexpected Yang data columns in {path}")
    return [{name: float(value) for name, value in row.items()} for row in rows]


def load_bereiter_co2(path: Path = DEFAULT_CO2) -> list[Co2Observation]:
    data_lines = [
        line for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line and not line.startswith("#")
    ]
    reader = csv.DictReader(data_lines, delimiter="\t")
    rows = [
        Co2Observation(
            age_year_bp=float(row["age_gas_calBP"]),
            co2_ppm=float(row["co2_ppm"]),
            sigma_ppm=float(row["co2_1s_ppm"]),
        )
        for row in reader
    ]
    if len(rows) < 1000 or any(
        left.age_year_bp >= right.age_year_bp for left, right in zip(rows, rows[1:])
    ):
        raise ValueError(f"CO2 chronology is missing, too short, or unsorted in {path}")
    return rows


def interpolate_co2(age_year_bp: float, rows: list[Co2Observation]) -> dict[str, float]:
    ages = [row.age_year_bp for row in rows]
    index = bisect_right(ages, age_year_bp)
    if index == 0 or index == len(rows):
        raise ValueError(f"Age {age_year_bp:g} yr BP is outside the CO2 composite")
    lower = rows[index - 1]
    upper = rows[index]
    weight = (age_year_bp - lower.age_year_bp) / (upper.age_year_bp - lower.age_year_bp)
    return {
        "co2_ppm": lower.co2_ppm + weight * (upper.co2_ppm - lower.co2_ppm),
        "co2_1sigma_ppm": lower.sigma_ppm + weight * (upper.sigma_ppm - lower.sigma_ppm),
        "co2_lower_age_year_bp": lower.age_year_bp,
        "co2_upper_age_year_bp": upper.age_year_bp,
        "co2_bracket_width_year": upper.age_year_bp - lower.age_year_bp,
        "co2_nearest_age_distance_year": min(
            age_year_bp - lower.age_year_bp, upper.age_year_bp - age_year_bp
        ),
    }


def build_pairs(
    yang_rows: list[dict[str, float]],
    co2_rows: list[Co2Observation],
) -> list[dict[str, float]]:
    pairs = []
    for row in yang_rows:
        age_year_bp = 1000.0 * row["gas_age_ka_bp"]
        pairs.append(
            {
                **row,
                "gas_age_year_bp": age_year_bp,
                **interpolate_co2(age_year_bp, co2_rows),
            }
        )
    return pairs


def contiguous_age_windows(ages_ka: list[float], gap_ka: float = 10.0) -> list[list[float]]:
    ordered = sorted(ages_ka)
    windows: list[list[float]] = [[ordered[0], ordered[0]]]
    for age in ordered[1:]:
        if age - windows[-1][1] > gap_ka:
            windows.append([age, age])
        else:
            windows[-1][1] = age
    return windows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    yang_path: Path = DEFAULT_YANG,
    co2_path: Path = DEFAULT_CO2,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    yang_rows = load_yang(yang_path)
    co2_rows = load_bereiter_co2(co2_path)
    pairs = build_pairs(yang_rows, co2_rows)
    write_csv(pairs, output_path)

    co2_values = [row["co2_ppm"] for row in pairs]
    bracket_widths = [row["co2_bracket_width_year"] for row in pairs]
    report: dict[str, object] = {
        "role": "independent validation holdout; no model fitting",
        "isotope_source": {
            "citation": "Yang et al. (2022), Science",
            "doi": "10.1126/science.abj8826",
            "data_doi": "10.1594/PANGAEA.941483",
            "rows": len(yang_rows),
            "coordinate": (
                "17Delta in ppm relative to contemporaneously measured modern "
                "atmospheric O2; logarithmic reference slope lambda=0.516"
            ),
            "laboratory_normalization": (
                "at least two modern-air flasks measured each analytical day; "
                "Yang et al. (2022) supplementary methods"
            ),
        },
        "co2_source": {
            "citation": "Bereiter et al. (2015), corrected Antarctic composite",
            "dataset_doi": "10.25921/n8y4-bp27",
            "rows": len(co2_rows),
            "age_definition": "calendar years BP; present is 1950 CE",
        },
        "paired_coverage": {
            "rows": len(pairs),
            "age_ka_bp": [min(row["gas_age_ka_bp"] for row in pairs), max(row["gas_age_ka_bp"] for row in pairs)],
            "windows_ka_bp_gap_threshold_10ka": contiguous_age_windows(
                [row["gas_age_ka_bp"] for row in pairs]
            ),
            "interpolated_co2_ppm": [min(co2_values), max(co2_values)],
            "rows_within_validated_50_to_60000ppm_domain": sum(
                50.0 <= value <= 60000.0 for value in co2_values
            ),
            "maximum_co2_bracket_width_year": max(bracket_widths),
            "rows_with_co2_bracket_over_2000yr": sum(width > 2000.0 for width in bracket_widths),
        },
        "low_co2_extension": {
            "validated_physical_domain_ppm": [50.0, 60000.0],
            "native_training_nodes_ppm": [50.0, 100.0, 150.0, 200.0, 250.0],
            "native_crossed_holdouts_ppm": [75.0, 125.0, 175.0, 225.0, 275.0],
            "status": "promoted after native photochemistry and crossed holdouts passed",
        },
        "files": {
            "paired_csv": str(output_path.relative_to(ROOT)),
            "report_json": str(report_path.relative_to(ROOT)),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yang", type=Path, default=DEFAULT_YANG)
    parser.add_argument("--co2", type=Path, default=DEFAULT_CO2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(run(args.yang, args.co2, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
