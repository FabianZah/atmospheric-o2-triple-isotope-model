"""Reproduce the broad low-pO2 reviewer case through the public calculation."""
from pathlib import Path
import argparse
from dataclasses import asdict
import json
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from updated_constrained_pco2_posterior import (  # noqa: E402
    ConstrainedCoordinateInput, CoordinateConstraint, constrained_coordinate_posterior,
)


def run(case="low-gpp"):
    start = perf_counter()
    high_gpp = case == "high-gpp"
    request = ConstrainedCoordinateInput(
        solve_for="pCO2", target_air_cap_delta17_permil=-11. if high_gpp else -8., measurement_sigma_permil=.015,
        target_air_delta18_conventional_permil=23.9, delta18_measurement_sigma_permil=.3,
        constraints={"GPP": CoordinateConstraint("normal", center=522. if high_gpp else 58., sigma=29.),
                     "pO2": CoordinateConstraint("normal", center=.2, sigma=.2)},
        po2_grid_size=17,
    )
    result = constrained_coordinate_posterior(request)
    d = np.asarray(result.field_density).reshape(result.field_shape)
    m = np.asarray(result.field_probability_mass).reshape(result.field_shape)
    mask = np.asarray(result.field_hpd_mask).reshape(result.field_shape)
    report = {"inputs": asdict(request), "elapsed_seconds": perf_counter()-start, "median_ppm": result.posterior_median,
              "interval_ppm": result.equal_tailed_credible_interval,
              "components": int(label(mask, np.ones((3, 3)))[1]),
              "hpd_mass": result.field_hpd_probability_mass,
              "field_refinement": result.field_refinement_diagnostics,
              "constraint_medians": result.constraint_posterior_medians,
              "constraint_intervals": result.constraint_equal_tailed_credible_intervals}
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    stem = "high_gpp_field_refinement" if high_gpp else "low_po2_field_refinement"
    np.savez_compressed(out / (stem+".npz"), x=result.field_x_axis,
                        y=result.field_y_axis, density=d, mass=m, mask=mask)
    (out / (stem+".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    assert report["components"] == 1
    expected_median = 43822.69 if high_gpp else 5672.62
    expected_interval = [39698.46,49318.72] if high_gpp else [3369.84,9264.89]
    assert abs(result.posterior_median-expected_median) < 30.
    assert np.max(np.abs(np.asarray(result.equal_tailed_credible_interval)-expected_interval)) < 60.
    assert abs(m.sum()-1.) < 1e-10
    assert np.array_equal(mask, d >= result.field_hpd_density_threshold)
    assert abs(m[mask].sum()-.95) < .002


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("low-gpp", "high-gpp"), default="low-gpp")
    run(parser.parse_args().case)
