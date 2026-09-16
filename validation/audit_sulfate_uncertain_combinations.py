"""Bounded stress audit for simultaneous sulfate and atmospheric uncertainties."""

from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from sulfate_uncertainty import (
    BackgroundConstraint,
    IncorporationConstraint,
    SulfateLikelihoodInput,
    SulfateIntegrationSettings,
    exact_sulfate_likelihood,
    sulfate_computation_budget,
)
from updated_constrained_pco2_posterior import (
    ConstrainedCoordinateInput,
    CoordinateConstraint,
    constrained_coordinate_posterior,
)


def cases():
    observation = SulfateLikelihoodInput(
        -0.2,
        15.0,
        0.03,
        0.5,
        0.0,
        IncorporationConstraint("range", lower=0.2, upper=0.29),
        BackgroundConstraint("normal", center=0.1, sigma=0.05),
        1.0,
        None,
        "Numerical validation scenario; primary preservation assumed.",
        1.0,
        "none",
    )
    normal = {
        "GPP": CoordinateConstraint("normal", center=290, sigma=29),
        "pO2": CoordinateConstraint("normal", center=1, sigma=0.2),
    }
    scenarios = [
        ("reported", observation, normal),
        (
            "wide_normals",
            replace(
                observation,
                cap_delta17_sigma_permil=0.1,
                delta18_sigma_permil=5,
                incorporation=IncorporationConstraint(
                    "normal", center=0.25, sigma=0.15
                ),
                background=BackgroundConstraint("normal", center=0, sigma=0.2),
            ),
            {
                "GPP": CoordinateConstraint("normal", center=290, sigma=145),
                "pO2": CoordinateConstraint("normal", center=1, sigma=0.6),
            },
        ),
        (
            "all_ranges",
            replace(
                observation,
                incorporation=IncorporationConstraint("range", lower=0.05, upper=0.5),
                background=BackgroundConstraint("range", lower=-0.2, upper=0.2),
            ),
            {
                "GPP": CoordinateConstraint("range", lower=58, upper=580),
                "pO2": CoordinateConstraint("range", lower=0.1, upper=2),
            },
        ),
        (
            "negative_low_oxygen",
            replace(
                observation,
                measured_cap_delta17_permil=-2.0,
                background=BackgroundConstraint("normal", center=0, sigma=0.1),
            ),
            {
                "GPP": CoordinateConstraint("normal", center=58, sigma=29),
                "pO2": CoordinateConstraint("normal", center=0.2, sigma=0.2),
            },
        ),
        (
            "negative_large_errors",
            replace(
                observation,
                measured_cap_delta17_permil=-2.0,
                cap_delta17_sigma_permil=0.15,
                delta18_sigma_permil=10,
                incorporation=IncorporationConstraint("normal", center=0.2, sigma=0.15),
                background=BackgroundConstraint("range", lower=-0.25, upper=0.25),
            ),
            normal,
        ),
        (
            "zero_fraction_endpoint",
            replace(
                observation,
                incorporation=IncorporationConstraint("range", lower=0.0, upper=0.4),
                background=BackgroundConstraint("range", lower=-0.1, upper=0.2),
            ),
            normal,
        ),
        (
            "narrow_isotopes",
            replace(observation, cap_delta17_sigma_permil=0.003),
            normal,
        ),
        (
            "wide_nonair",
            replace(
                observation,
                background=BackgroundConstraint("normal", center=0.0, sigma=0.5),
            ),
            normal,
        ),
        (
            "fixed_fraction",
            replace(
                observation, incorporation=IncorporationConstraint("fixed", center=0.25)
            ),
            normal,
        ),
        (
            "fixed_nonair",
            replace(observation, background=BackgroundConstraint("fixed", center=0.0)),
            normal,
        ),
    ]
    for name, obs, constraints in scenarios:
        yield name, ConstrainedCoordinateInput(
            solve_for="pCO2",
            target_air_cap_delta17_permil=None,
            measurement_sigma_permil=None,
            constraints=constraints,
            sulfate=obs,
        )
    for coordinate in ("GPP", "pO2"):
        for mode in ("normal", "range"):
            constraints = {
                "pCO2": (
                    CoordinateConstraint("normal", center=10000, sigma=10000)
                    if mode == "normal"
                    else CoordinateConstraint("range", lower=50, upper=60000)
                )
            }
            other = "GPP" if coordinate == "pO2" else "pO2"
            constraints[other] = (
                CoordinateConstraint("normal", center=145, sigma=100)
                if other == "GPP"
                else CoordinateConstraint("normal", center=0.5, sigma=0.4)
            )
            yield coordinate + "_" + mode, ConstrainedCoordinateInput(
                solve_for=coordinate,
                target_air_cap_delta17_permil=None,
                measurement_sigma_permil=None,
                constraints=constraints,
                sulfate=replace(observation, measured_cap_delta17_permil=-2.0),
            )


def run(selected=None, resume=False, max_seconds=180.0):
    import updated_output_surface_joint_posterior as joint

    accelerated = joint.exact_sulfate_likelihood
    implementation = hashlib.sha256()
    for path in sorted((ROOT / "code").glob("*.py")):
        implementation.update(path.name.encode())
        implementation.update(path.read_bytes())
    implementation.update(Path(__file__).read_bytes())
    fingerprint = implementation.hexdigest()
    target = ROOT / "outputs" / "sulfate_uncertain_combinations.json"
    old = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
    current = {
        row["case"]: row
        for row in old
        if row.get("implementation_sha256") == fingerprint
    }
    output = []
    for name, request in cases():
        if selected and name not in selected:
            continue
        if resume and name in current and current[name]["status"] == "passed":
            output.append(current[name])
            continue
        row = {
            "case": name,
            "request": asdict(request),
            "implementation_sha256": fingerprint,
            "test_time_limit_seconds": max_seconds,
        }
        checks = []
        samples = []

        def audited(a, d, r):
            result = accelerated(a, d, r)
            # Sample both the likelihood support and full domain, independently
            # of the interpolation's structured midpoint checks.
            order = np.argsort(result.log_likelihood.ravel())
            index = np.unique(
                np.r_[
                    order[np.linspace(0, len(order) - 1, 31).astype(int)],
                    np.random.default_rng(7123).choice(
                        a.size, min(31, a.size), replace=False
                    ),
                ]
            )
            samples.append(
                (
                    np.asarray(a).ravel()[index],
                    np.asarray(d).ravel()[index],
                    r,
                    result.log_likelihood.ravel()[index],
                    result.diagnostics.get("likelihood_interpolation"),
                )
            )
            return result

        joint.exact_sulfate_likelihood = audited
        start = perf_counter()
        try:
            with sulfate_computation_budget(max_seconds=max_seconds):
                result = constrained_coordinate_posterior(request)
            row.update(
                status="passed",
                median=result.posterior_median,
                interval=result.equal_tailed_credible_interval,
                diagnostics=result.sulfate_likelihood_diagnostics,
            )
        except Exception as exc:
            row.update(
                status="failed",
                error=type(exc).__name__ + ": " + str(exc),
                error_diagnostics=getattr(exc, "diagnostics", None),
            )
        finally:
            joint.exact_sulfate_likelihood = accelerated
        row["solve_seconds"] = perf_counter() - start
        print(name, "solve", row["status"], round(row["solve_seconds"], 2), flush=True)
        for a, d, r, log_likelihood, interpolation in (
            samples if row["status"] == "passed" else []
        ):
            check_start = perf_counter()
            try:
                with sulfate_computation_budget(max_seconds=60):
                    direct = exact_sulfate_likelihood(
                        a,
                        d,
                        r,
                        settings=SulfateIntegrationSettings(
                            relative_tolerance=5e-5,
                            absolute_tolerance=2.5e-11,
                            max_level=8,
                            adaptive_delta18=True,
                        ),
                    )
                scale = r.cap_delta17_sigma_permil * np.sqrt(
                    2 * np.pi * (1 - r.isotope_error_correlation**2)
                )
                exact, actual = (
                    np.exp(direct.log_likelihood) * scale,
                    np.exp(log_likelihood) * scale,
                )
                ratio = float(np.max(np.abs(exact - actual) / (1e-10 + 2e-4 * exact)))
                checks.append(
                    {
                        "states": a.size,
                        "maximum_error_over_tolerance": ratio,
                        "interpolation": interpolation,
                        "seconds": perf_counter() - check_start,
                    }
                )
                if ratio > 1:
                    row["status"] = "accuracy_failed"
            except Exception as exc:
                row.update(status="reference_failed", reference_error=str(exc))
        row.update(seconds=perf_counter() - start, independent_checks=checks)
        output.append(row)
        print(
            name,
            row["status"],
            round(row["seconds"], 2),
            row.get("error", row.get("median")),
            flush=True,
        )
        current[name] = row
        target.write_text(
            json.dumps(list(current.values()), indent=2), encoding="utf-8"
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse passed cases only for identical code and audit hashes",
    )
    args = parser.parse_args()
    run(args.cases, args.resume, args.max_seconds)
