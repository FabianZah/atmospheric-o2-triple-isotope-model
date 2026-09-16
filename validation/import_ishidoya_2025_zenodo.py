"""Verify and reduce the Ishidoya et al. (2025) Zenodo archive.

The source archive is not redistributed. This importer verifies the locally
downloaded files, preserves a checksum manifest, and writes a small derived
annual data table suitable for model validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / ".project-root").exists()
)
OUTPUT_PATH = ROOT / "model_data" / "literature" / "ishidoya_2025_annual_delta18.csv"
MANIFEST_PATH = (
    ROOT / "model_data" / "literature" / "ishidoya_2025_source_manifest.json"
)

DOI = "10.5281/zenodo.14221768"
ARTICLE_DOI = "10.5194/acp-25-1965-2025"
EXPECTED_SHA256 = {
    "raw_data_2013.csv": "898fe7d7637ab1122e47f8db5134c8ac6a9c43c4eeb48a0578bdd9c46ef70df7",
    "raw_data_2014.csv": "bdeb75dc3fa3a9470466134eb2762585c1f5000c87ee9576f318b5e75b8bf945",
    "raw_data_2015.csv": "5614417636c21d153164f8f246ae317854ad0fdbfe1fb0428331b6b6e21977ea",
    "raw_data_2016.csv": "956b9abe35bd3804edb60384b73e9b70f7dd0185ee7a556e8e0f61846abf3068",
    "raw_data_2017.csv": "9563e1da8759c2d2d341c2a188b6b858b393f85afa9b3009278b44f6f62a044e",
    "raw_data_2018.csv": "e056e1b594bd988d2914f80275b4420c0eeb7f0e012ec39f296875e086c19ff1",
    "raw_data_2019.csv": "35239f9954b4451f76c2a0fb6ccc6d0850eaa5e40cf692bd5b90828afc7e0c2b",
    "raw_data_2020.csv": "4d5b0de56127c728d322dc3ddbf87c8955134bab57dbb2c4589e0c3d4f49b747",
    "raw_data_2021.csv": "dc08de0d0e575cc85504e55adf201d15cbf7c3abc81ee87e5294a43516d7837f",
    "raw_data_2022.csv": "1bcaa31816a893f8caadbba6fee5c47cf23ab87439e4bbac672cfe4fec250a65",
    "data_for_Fig4.csv": "4266de3b9c95018b0645900fcc65cfebd49e855d60f5ab8778ce016c26f9c2e9",
    "data_for_Fig6.csv": "147efef1e889460cad01b0ec159626b2e83a30e137f735d5ccc1c83d789436cc",
    "data_for_Fig7.csv": "dff21d70d008281e9d62caa1ab19f914e8628d5bbd04031e96869313c33b56fd",
}

# Months omitted from the prepared Figure 7 delta18 series because the paper
# identifies four periods of unreliable mass-spectrometer operation.
EXCLUDED_MONTHS = {(2014, 6), (2014, 7), (2016, 9), (2020, 1)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(source_dir: Path) -> list[dict[str, object]]:
    records = []
    for name, expected in EXPECTED_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
        records.append({"name": name, "bytes": path.stat().st_size, "sha256": actual})
    return records


def _prepared_monthly_values(path: Path) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if line_number < 8:
                continue
            try:
                fractional_year = float(row[0])
                delta18_ppm = float(row[1])
            except (IndexError, ValueError):
                continue
            values[int(fractional_year)].append(delta18_ppm)
    return dict(values)


def _raw_annual_values(source_dir: Path) -> dict[int, dict[str, float | int]]:
    output: dict[int, dict[str, float | int]] = {}
    for year in range(2013, 2023):
        monthly_sum = defaultdict(float)
        monthly_count = defaultdict(int)
        annual_sum = 0.0
        annual_count = 0
        with (source_dir / f"raw_data_{year}.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            reader = csv.reader(stream)
            for _ in range(6):
                next(reader, None)
            for row in reader:
                try:
                    month = int(row[0].split("/")[1])
                    delta18_ppm = float(row[3])
                except (IndexError, ValueError):
                    continue
                if not math.isfinite(delta18_ppm) or (year, month) in EXCLUDED_MONTHS:
                    continue
                monthly_sum[month] += delta18_ppm
                monthly_count[month] += 1
                annual_sum += delta18_ppm
                annual_count += 1
        monthly_means = [
            monthly_sum[month] / monthly_count[month] for month in sorted(monthly_count)
        ]
        output[year] = {
            "raw_equal_month_mean_ppm": sum(monthly_means) / len(monthly_means),
            "raw_point_weighted_mean_ppm": annual_sum / annual_count,
            "included_months": len(monthly_means),
            "valid_raw_points": annual_count,
        }
    return output


def run(source_dir: Path, output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    source_dir = source_dir.expanduser().resolve()
    verified_files = _verify_archive(source_dir)
    prepared = _prepared_monthly_values(source_dir / "data_for_Fig7.csv")
    raw = _raw_annual_values(source_dir)
    years = list(range(2013, 2023))
    prepared_means = {year: sum(prepared[year]) / len(prepared[year]) for year in years}
    reference = {
        "prepared": prepared_means[2013],
        "raw_equal_month": float(raw[2013]["raw_equal_month_mean_ppm"]),
        "raw_point_weighted": float(raw[2013]["raw_point_weighted_mean_ppm"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "year_ce",
            "prepared_fig7_month_count",
            "prepared_fig7_annual_mean_ppm",
            "prepared_fig7_change_from_2013_ppm",
            "raw_included_month_count",
            "raw_valid_point_count",
            "raw_equal_month_annual_mean_ppm",
            "raw_equal_month_change_from_2013_ppm",
            "raw_point_weighted_annual_mean_ppm",
            "raw_point_weighted_change_from_2013_ppm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for year in years:
            equal_month = float(raw[year]["raw_equal_month_mean_ppm"])
            point_weighted = float(raw[year]["raw_point_weighted_mean_ppm"])
            writer.writerow(
                {
                    "year_ce": year,
                    "prepared_fig7_month_count": len(prepared[year]),
                    "prepared_fig7_annual_mean_ppm": f"{prepared_means[year]:.9f}",
                    "prepared_fig7_change_from_2013_ppm": (
                        f"{prepared_means[year] - reference['prepared']:.9f}"
                    ),
                    "raw_included_month_count": raw[year]["included_months"],
                    "raw_valid_point_count": raw[year]["valid_raw_points"],
                    "raw_equal_month_annual_mean_ppm": f"{equal_month:.9f}",
                    "raw_equal_month_change_from_2013_ppm": (
                        f"{equal_month - reference['raw_equal_month']:.9f}"
                    ),
                    "raw_point_weighted_annual_mean_ppm": f"{point_weighted:.9f}",
                    "raw_point_weighted_change_from_2013_ppm": (
                        f"{point_weighted - reference['raw_point_weighted']:.9f}"
                    ),
                }
            )

    endpoint_rates = {
        "prepared_fig7_ppm_per_year": (prepared_means[2022] - prepared_means[2013]) / 9,
        "raw_equal_month_ppm_per_year": (
            float(raw[2022]["raw_equal_month_mean_ppm"])
            - float(raw[2013]["raw_equal_month_mean_ppm"])
        )
        / 9,
        "raw_point_weighted_ppm_per_year": (
            float(raw[2022]["raw_point_weighted_mean_ppm"])
            - float(raw[2013]["raw_point_weighted_mean_ppm"])
        )
        / 9,
    }
    manifest: dict[str, object] = {
        "dataset": "Ishidoya et al. (2025) TKB atmospheric O2 delta18 archive",
        "archive_doi": DOI,
        "article_doi": ARTICLE_DOI,
        "license": "CC BY 4.0",
        "verified_files": verified_files,
        "excluded_unreliable_months": [f"{year:04d}-{month:02d}" for year, month in sorted(EXCLUDED_MONTHS)],
        "primary_validation_series": (
            "Equal-weight annual mean of the archived data_for_Fig7.csv monthly "
            "delta_atm(18O) values; missing values remain missing."
        ),
        "annualization_sensitivity": (
            "Raw equal-month and raw-point-weighted annual means are retained because "
            "the article does not state which weighting generated Figure 8."
        ),
        "source_data_note": (
            "The prepared Figure 7 file contains two nearly identical 2018.2034 rows "
            "and no distinct April fractional year. The archived values are preserved; "
            "the raw-data annualizations independently include April 2018."
        ),
        "endpoint_rates_ppm_per_year": endpoint_rates,
        "published_endpoint_rate_ppm_per_year": 0.22,
        "derived_table": str(output_path.relative_to(ROOT)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="Directory containing the 13 files from Zenodo record 14221768",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir), indent=2))


if __name__ == "__main__":
    main()
