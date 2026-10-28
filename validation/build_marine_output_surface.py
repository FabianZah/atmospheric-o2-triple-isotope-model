"""Build the fixed marine accelerator from live, configured isotope budgets.

The modern cache supplies grid coordinates only. Numerical values and source
endpoints are calculated anew. Frozen marine biology has no biological ensemble.
"""
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
from updated_molecular_forward_model import UpdatedForwardInput, run_updated_forward, run_updated_central_state
from audit_updated_output_surface import _live_fields
from biosphere_configurations import REGISTRY_PATH, output_surface_path


def calculate(task):
    kind, index, point = task
    request = UpdatedForwardInput(*point, biosphere_configuration="marine_endmember")
    if kind == "full":
        result = run_updated_forward(request)
        return index, _live_fields(result), result.maximum_fixed_point_residual_permil
    result = run_updated_central_state(request)
    return index, result.delta18_prime_permil, result.maximum_fixed_point_residual_permil


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def run_tasks(tasks, path, identity, workers):
    completed = {}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Incompatible checkpoint: {path}; use another checkpoint directory")
        completed = {int(k): v for k, v in saved["completed"].items()}
    pending = [task for task in tasks if task[1] not in completed]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, (index, value, residual) in enumerate(pool.map(calculate, pending, chunksize=16), 1):
            completed[index] = dict(value=value, residual=residual)
            if count % 500 == 0 or count == len(pending):
                write(path, dict(identity=identity, completed=completed))
                print(f"{path.stem}: {len(completed)}/{len(tasks)}", flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/marine_surface_build")
    args = parser.parse_args()
    started = time.monotonic()
    paths = [ROOT / "model_data/updated_r7_response_surface_v1.json", REGISTRY_PATH,
             ROOT / "code/updated_molecular_forward_model.py", ROOT / "code/biosphere_configurations.py",
             ROOT / "code/local_r7_response_operator.py", ROOT / "code/self_consistent_isotope_fixed_point.py"]
    identity = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}
    template = json.loads(output_surface_path().read_text())
    keys = ("po2_pal", "pco2_ppm", "gpp_pgC_per_year")
    axes = template["axes"]
    points = [tuple(map(float, (p, c, g))) for p in axes[keys[0]] for c in axes[keys[1]] for g in axes[keys[2]]]
    full = run_tasks([("full", i, p) for i, p in enumerate(points)], args.checkpoint_dir / "full.json", identity, args.workers)
    shape = tuple(len(axes[k]) for k in keys)
    fields = {k: np.array([full[i]["value"][k] for i in range(len(points))]).reshape(shape).tolist() for k in full[0]["value"]}
    da = template["delta18_surface"]["axes"]
    dpoints = [tuple(map(float, (p, c, g))) for p in da[keys[0]] for c in da[keys[1]] for g in da[keys[2]]]
    lookup = {p: full[i]["value"]["central_delta18_prime_permil"] for i, p in enumerate(points)}
    extra = run_tasks([("central", i, p) for i, p in enumerate(dpoints) if p not in lookup], args.checkpoint_dir / "delta18.json", identity, args.workers)
    values = [lookup[p] if p in lookup else extra[i]["value"] for i, p in enumerate(dpoints)]
    bundle = dict(schema_version=1, biosphere_configuration="marine_endmember",
        configuration_version="marine_blunier2012_light_active_v2",
        surface_data_id="oxytib_marine_output_20261019", upstream_model_data_id=template["upstream_model_data_id"],
        axes=axes, interpolation=template["interpolation"], fields=fields,
        delta18_surface=dict(axes=da, values=np.array(values).reshape(tuple(len(da[k]) for k in keys)).tolist()),
        construction=dict(source_hashes=identity, source_hash_convention="SHA-256 of LF-normalized text",
            node_count=len(points), delta18_node_count=len(dpoints),
            elapsed_seconds=time.monotonic()-started, biological_ensemble_included=False,
            maximum_fixed_point_residual_permil=max(x["residual"] for x in (*full.values(), *extra.values())),
            output_offset_applied=False, smoothing_applied=False),
        validation=dict(status="pending_independent_holdouts", delta18_acceleration_validated=False,
            cap_delta17_maximum_absolute_residual_permil=0.))
    # A pending cache stays in outputs until its holdouts pass.
    write(args.checkpoint_dir / "marine_molecular_output_surface_v1.json", bundle)
    print(f"Built pending marine cache in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
