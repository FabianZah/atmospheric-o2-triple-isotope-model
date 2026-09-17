"""Regenerate selected OXYTIB comparisons from bundled scientific inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
BENCHMARKS = {
    "fig8": ("audit_updated_fig8_response_shape.py",),
    "cao-bao": ("audit_cao_bao_2013_benchmark.py",),
    "luz": ("audit_luz_1999_productivity.py",),
    "banerjee": ("audit_banerjee_2026_updated_model.py",),
    "yang": ("prepare_ice_core_holdout.py", "audit_yang_2022_co2_tracking.py"),
    "brandon": ("audit_brandon_2020_termination_v.py",),
    "uncertainty": ("audit_uncertainty_layers.py",),
    "liu": ("merge_liu_2021_gpp_grid.py",),
    "climate": ("reproduce_climate.py",),
    "numerical": ("reproduce_numerical_audits.py",),
    "scorecard": ("prepare_ice_core_holdout.py", "audit_yang_2022_co2_tracking.py",
                  "audit_yang_2022_transient_co2_tracking.py",
                  "audit_yang_lowco2_predictive_error.py",
                  "audit_banerjee_2026_updated_model.py",
                  "audit_brandon_2020_termination_v.py", "audit_uncertainty_layers.py",
                  "reproduce_numerical_audits.py", "audit_updated_molecular_release_scorecard.py"),
}


def input_inventory() -> list[dict[str, object]]:
    roots = (ROOT / "model_data", ROOT / "validation/reference_data", ROOT / "code/data")
    paths = [p for folder in roots for p in sorted(folder.rglob("*")) if p.is_file()]
    paths.extend(p for folder in (ROOT / "code", ROOT / "validation")
                 for p in sorted(folder.glob("*.py")))
    paths.extend(ROOT / name for name in (
        "outputs/young_fig7_digitized_contours.csv",
        "outputs/young_fig8_digitized_curves.csv",
        "run_model.py", "pyproject.toml",
    ))
    paths.extend(sorted((ROOT / "code").glob("requirements*.txt")))
    return [
        {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(set(paths))
    ]


def run(names: list[str], *, timeout_seconds: int = 3600) -> dict[str, object]:
    if not names or any(name not in BENCHMARKS for name in names):
        raise ValueError("choose at least one registered benchmark")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = ROOT / "outputs/reproduction" / run_id
    output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "h5py", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {"schema_version": 1, "run_id": run_id, "python": sys.version,
              "packages": packages, "benchmarks": names, "timeout_seconds": timeout_seconds,
              "input_files": input_inventory(), "runs": [], "status": "running"}
    report_path = output / "manifest.json"
    completed_scripts: set[str] = set()
    try:
        for name in names:
            for script in BENCHMARKS[name]:
                if script in completed_scripts:
                    continue
                command = [sys.executable, str(ROOT / "validation" / script)]
                log = output / (Path(script).stem + ".log")
                started = time.monotonic()
                print(f"Running {name}: {script}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    try:
                        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                                stderr=subprocess.STDOUT,
                                                timeout=timeout_seconds, check=False)
                        returncode = result.returncode
                    except subprocess.TimeoutExpired:
                        returncode = "timeout"
                report["runs"].append({"script": "validation/" + script,
                                       "returncode": returncode,
                                       "seconds": time.monotonic() - started,
                                       "log": log.relative_to(ROOT).as_posix()})
                if returncode != 0:
                    raise RuntimeError(f"{script} failed ({returncode}); see {log}")
                completed_scripts.add(script)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "incomplete"
        raise
    finally:
        content = json.dumps(report, indent=2) + "\n"
        report_path.write_text(content, encoding="utf-8")
        latest = output.parent / (run_id + ".tmp")
        latest.write_text(content, encoding="utf-8")
        latest.replace(output.parent / "manifest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", choices=tuple(BENCHMARKS))
    parser.add_argument("--list", action="store_true", help="List available comparisons.")
    parser.add_argument("--timeout-seconds", type=int, default=3600,
                        help="Maximum runtime per calculation (default: 3600 seconds).")
    args = parser.parse_args()
    if args.list:
        print("\n".join(BENCHMARKS))
        return
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    report = run(args.benchmarks or [name for name in BENCHMARKS
                                    if name not in {"climate", "numerical", "scorecard"}],
                 timeout_seconds=args.timeout_seconds)
    print(f"Completed {len(report['runs'])} calculations; see outputs/reproduction/manifest.json")


if __name__ == "__main__":
    main()
