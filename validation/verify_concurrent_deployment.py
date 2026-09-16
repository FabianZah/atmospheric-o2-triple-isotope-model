"""Compare sequential and simultaneous HTTP calculations on private staging.

Restart staging between baseline and concurrent phases to avoid transient-cache
hits. Never aim this resource-intensive check at a shared public deployment.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from http.cookies import SimpleCookie
import json
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen
from zipfile import ZipFile


def cases():
    initial = {"p_o2_pal": 1.0, "p_co2_ppm": 294.0, "gpp_pgC_per_year": 290.0}
    dense = {
        "solve_for": "pCO2", "target_air_cap_delta17_permil": -8.0,
        "measurement_sigma_permil": 0.015,
        "gpp_constraint": {"kind": "normal", "center": 58.0, "sigma": 29.0},
        "po2_constraint": {"kind": "normal", "center": 0.2, "sigma": 0.2},
    }
    sulfate = {
        "solve_for": "pCO2",
        "sulfate": {
            "measured_cap_delta17_permil": -0.2, "measured_delta18_permil": 15.0,
            "cap_delta17_sigma_permil": 0.03, "delta18_sigma_permil": 0.5,
            "isotope_error_correlation": 0.0,
            "incorporation": {"kind": "range", "lower": 0.2, "upper": 0.29},
            "background": {"kind": "normal", "center": 0.1, "sigma": 0.05},
            "fractionation_treatment": "none",
        },
        "gpp_constraint": {"kind": "normal", "center": 290.0, "sigma": 29.0},
        "po2_constraint": {"kind": "normal", "center": 1.0, "sigma": 0.2},
    }
    return {
        "photosynthesis": ("/api/v1/transients/photosynthesis-step", {
            "initial": initial, "photosynthesis_fraction": 0.5,
            "duration_years": 22000.0, "sample_count": 161,
        }),
        "gpp_step": ("/api/v1/transients/state-step", {
            "initial": initial, "final": dict(initial, gpp_pgC_per_year=145.0),
            "duration_years": 18000.0, "sample_count": 161,
        }),
        "dense_air": ("/api/v1/inference/coordinate", dense),
        "dense_air_second": ("/api/v1/inference/coordinate", dict(dense, target_air_cap_delta17_permil=-7.5)),
        "dense_air_repeat": ("/api/v1/inference/coordinate", dict(dense)),
        "sulfate": ("/api/v1/inference/coordinate", sulfate),
    }


def request(base, path, payload=None, cookie=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = Request(base.rstrip("/") + path, data=data, headers=headers)
    with urlopen(req, timeout=600) as response:
        return response.read()


def session_cookie(base):
    with urlopen(base.rstrip("/") + "/api/v1/model", timeout=30) as response:
        cookie = SimpleCookie(response.headers.get("Set-Cookie", ""))
        assert "oxytib_client" in cookie, "Staging must issue a client session"
        return "oxytib_client=" + cookie["oxytib_client"].value


def numerical_result(value):
    """Exclude wall-clock diagnostics; retain every scientific result and tolerance."""
    if isinstance(value, dict):
        return {key: numerical_result(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [numerical_result(item) for item in value]
    return value


def run(base, phase, reference):
    inventory = cases()
    # Independent cases model different visitors; the repeated dense pair tests
    # one visitor's two-slot burst. Export clients are independent downloaders.
    sessions = {name: session_cookie(base) for name in inventory}
    sessions["dense_air_repeat"] = sessions["dense_air"]
    report = {"phase": phase, "cases": {}, "pairs": []}

    def solve(name):
        path, payload = inventory[name]
        start = monotonic()
        result = json.loads(request(base, path, payload, sessions[name]))["result"]
        row = {"seconds": monotonic() - start,
               "result_sha256": sha256(json.dumps(numerical_result(result), sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        if phase == "concurrent":
            assert row["result_sha256"] == reference["cases"][name]["result_sha256"], name
        return row

    if phase == "baseline":
        identical = {}
        for name in inventory:
            key = json.dumps(inventory[name], sort_keys=True)
            if key in identical:
                report["cases"][name] = dict(report["cases"][identical[key]], identical_input_reference=identical[key])
            else:
                report["cases"][name] = solve(name)
                identical[key] = name
            print(json.dumps({name: report["cases"][name]}), flush=True)
    else:
        for pair in (("photosynthesis", "gpp_step"), ("dense_air", "dense_air_repeat"), ("sulfate", "dense_air_second")):
            start = monotonic()
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {name: pool.submit(solve, name) for name in pair}
                assert json.loads(request(base, "/api/v1/health"))["status"] == "ok"
                for name, future in futures.items():
                    report["cases"][name] = future.result()
            report["pairs"].append({"names": pair, "seconds": monotonic() - start})
            print(json.dumps(report["pairs"][-1]), flush=True)

        def export(name):
            cookie = session_cookie(base)
            start = monotonic()
            data = request(base, "/api/v1/export/coordinate.xlsx", {
                "inference": inventory[name][1], "context": {"isotope_source": "Direct air O2"},
            }, cookie)
            with ZipFile(BytesIO(data)) as archive:
                assert archive.testzip() is None
            return {"name": name, "seconds": monotonic() - start, "bytes": len(data)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            report["exports"] = list(pool.map(export, ("dense_air", "dense_air_second")))
        print(json.dumps({"exports": report["exports"]}), flush=True)
    assert json.loads(request(base, "/api/v1/health"))["status"] == "ok"
    report["status"] = "pass"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--phase", required=True, choices=("baseline", "concurrent"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "concurrent" and args.reference is None:
        parser.error("concurrent phase requires --reference")
    result = run(args.base_url, args.phase, json.loads(args.reference.read_text()) if args.reference else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
