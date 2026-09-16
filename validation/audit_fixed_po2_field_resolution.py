"""Independent dense quadrature check for the precise fixed-pO2 reviewer case."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from updated_constrained_pco2_posterior import (  # noqa: E402
    ConstrainedCoordinateInput, CoordinateConstraint, constrained_coordinate_posterior,
)
from updated_output_surface import load_updated_output_surface  # noqa: E402
from updated_output_surface_joint_posterior import _quantile, _trapezoid_weights  # noqa: E402


def run():
    started = perf_counter()
    request = ConstrainedCoordinateInput(
        solve_for="pCO2", target_air_cap_delta17_permil=-11, measurement_sigma_permil=.015,
        target_air_delta18_conventional_permil=23.9, delta18_measurement_sigma_permil=.3,
        constraints={"GPP": CoordinateConstraint("normal", center=522, sigma=29),
                     "pO2": CoordinateConstraint("fixed", center=.2)}, po2_grid_size=17)
    result = constrained_coordinate_posterior(request)
    surface = load_updated_output_surface()
    def log_density(x, y):
        kwargs = dict(p_o2_pal=.2, p_co2_ppm=x, gpp_pgC_per_year=y)
        d17 = surface.evaluate_central_cap_delta17_grid(**kwargs)
        d18 = 1000*np.expm1(surface.evaluate_central_delta18_prime_grid(**kwargs)/1000)
        return -.5*((d17+11)/.015)**2-.5*((d18-23.9)/.3)**2-.5*((y-522)/29)**2

    density = np.array(result.field_density).reshape(result.field_shape)
    mass = np.array(result.field_probability_mass).reshape(result.field_shape)
    mask = np.array(result.field_hpd_mask).reshape(result.field_shape)
    peak = np.unravel_index(np.argmax(density), density.shape)
    peak_log = float(log_density(result.field_x_axis[peak[0]], result.field_y_axis[peak[1]]))
    cutoff = peak_log+np.log(result.field_hpd_density_threshold/density.max())
    # An independent, uniformly spaced full-domain rectangle, evaluated in strips.
    # It uses no axis concentration, nuisance integral, or image interpolation.
    x, y = np.linspace(50,60000,6001), np.linspace(406,638,3201)
    wx, wy = _trapezoid_weights(x), _trapezoid_weights(y)
    marginal = np.zeros(len(x))
    inside = 0.
    for first in range(0,len(x),16):
        stop = min(first+16,len(x))
        logs = log_density(x[first:stop,None],y[None,:])
        strip = np.exp(logs-peak_log)*wx[first:stop,None]*wy[None,:]
        marginal[first:stop] = strip.sum(axis=1)
        inside += strip[logs >= cutoff].sum()
    total = marginal.sum()
    marginal /= total
    reference = [_quantile(x,marginal,q) for q in (.025,.5,.975)]
    actual = [result.equal_tailed_credible_interval[0],result.posterior_median,result.equal_tailed_credible_interval[1]]
    report = {"inputs":asdict(request),"elapsed_seconds":perf_counter()-started,
              "reference_grid":[len(x),len(y)],"reference_quantiles_ppm":reference,
              "public_quantiles_ppm":actual,"quantile_errors_ppm":(np.array(actual)-reference).tolist(),
              "reference_probability_above_public_hpd_threshold":inside/total,
              "field_refinement":result.field_refinement_diagnostics}
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "fixed_po2_field_resolution.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(out / "fixed_po2_field_resolution.npz", x=result.field_x_axis,
                        y=result.field_y_axis,density=density,mass=mass,mask=mask)
    print(json.dumps(report,indent=2),flush=True)
    assert max(abs(np.array(actual)-reference)) < 20
    assert abs(inside/total-.95) < .002


if __name__ == "__main__":
    run()
