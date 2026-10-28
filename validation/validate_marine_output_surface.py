"""Validate marine cache in every interpolation cell before local activation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from build_marine_output_surface import calculate, write
from biosphere_configurations import output_surface_path
from updated_output_surface import UpdatedMolecularOutputSurface, UpdatedOutputSurfaceInput
from audit_updated_output_surface import _interpolated_fields


def cell_points(axes, phase):
    fractions = np.array([.2113248654, .3660254038, .6339745962, .7886751346])
    selected = []
    for dimension, key in enumerate(("po2_pal", "pco2_ppm", "gpp_pgC_per_year")):
        axis = np.log(np.asarray(axes[key]))
        f = fractions[(np.arange(len(axis)-1)+phase+dimension) % len(fractions)]
        selected.append(np.exp(axis[:-1]+f*np.diff(axis)))
    return [tuple(map(float, (p, c, g))) for p in selected[0] for c in selected[1] for g in selected[2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / "outputs/marine_surface_build/marine_molecular_output_surface_v1.json")
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    bundle = json.loads(args.candidate.read_text())
    for relative, expected in bundle["construction"]["source_hashes"].items():
        data = (ROOT / relative).read_bytes()
        if bundle["construction"].get("source_hash_convention") == "SHA-256 of LF-normalized text":
            data = data.replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Cache source changed: {relative}")
    surface = UpdatedMolecularOutputSurface(bundle)
    full_points = cell_points(bundle["axes"], 2)
    d18_points = cell_points(bundle["delta18_surface"]["axes"], 1)
    # Freeze the test coordinates before evaluating live values.
    design = dict(candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        full_points=full_points, delta18_points=d18_points,
        gates=dict(D17=.005, delta18=.015), scope="noncentral point in every cache cell")
    write(args.candidate.parent / "holdout_design.json", design)
    maximum_d17 = maximum_d18 = maximum_residual = 0.
    worst = {}
    for kind, points in (("full", full_points), ("central", d18_points)):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [(kind, i, p) for i, p in enumerate(points)]
            for count, (index, value, residual) in enumerate(pool.map(calculate, tasks, chunksize=16), 1):
                request = UpdatedOutputSurfaceInput(*points[index], biosphere_configuration="marine_endmember")
                cached = _interpolated_fields(surface, request)
                if kind == "full":
                    error = max(abs(cached[key]-item) for key, item in value.items() if key != "central_delta18_prime_permil")
                    if error > maximum_d17:
                        maximum_d17 = error
                        worst["D17"] = dict(point=points[index], error=error)
                    d18_error = abs(cached["central_delta18_prime_permil"]-value["central_delta18_prime_permil"])
                else:
                    d18_error = abs(cached["central_delta18_prime_permil"]-value)
                if d18_error > maximum_d18:
                    maximum_d18 = d18_error
                    worst["delta18"] = dict(point=points[index], error=d18_error)
                maximum_residual = max(maximum_residual, residual)
                if count % 2000 == 0:
                    print(f"Validated {kind} {count}/{len(points)}", flush=True)
    passed = maximum_d17 <= .005 and maximum_d18 <= .015 and maximum_residual <= 1e-8
    report = dict(passed=passed, full_holdout_count=len(full_points), delta18_holdout_count=len(d18_points),
        cap_delta17_maximum_absolute_residual_permil=maximum_d17,
        central_delta18_maximum_absolute_residual_permil=maximum_d18,
        maximum_root_residual_permil=maximum_residual, worst=worst,
        elapsed_seconds=time.monotonic()-start, design_sha256=hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest(),
        delta18_acceleration_validated=passed,
        status="every_cell_holdouts_passed" if passed else "failed_holdouts",
        scope="Numerical accelerator versus configured live budget; upstream column checks separate")
    write(args.candidate.parent / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)
    if args.activate:
        bundle["validation"] = report
        write(output_surface_path("marine_endmember"), bundle)


if __name__ == "__main__":
    main()
