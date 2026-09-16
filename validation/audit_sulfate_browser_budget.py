"""Runtime and numerical audit of the multi-constraint sulfate browser case."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from web_api import ConstrainedCoordinateRequest  # noqa: E402
from sulfate_uncertainty import sulfate_computation_budget  # noqa: E402
from updated_constrained_pco2_posterior import constrained_coordinate_posterior  # noqa: E402
import updated_output_surface_joint_posterior as joint  # noqa: E402


def browser_request():
    return ConstrainedCoordinateRequest.model_validate({
        "solve_for":"pCO2", "pco2_grid_size":181,"gpp_grid_size":81,"po2_grid_size":17,
        "gpp_constraint":{"kind":"normal","center":174.,"sigma":29.},
        "po2_constraint":{"kind":"range","lower":.4,"upper":.8},
        "sulfate":{"measured_cap_delta17_permil":-.2,"measured_delta18_permil":15.,
            "cap_delta17_sigma_permil":.03,"delta18_sigma_permil":.5,
            "isotope_error_correlation":0.,
            "incorporation":{"kind":"range","lower":.18,"upper":.29},
            "background":{"kind":"fixed","center":0.},
            "alpha18_air_to_sulfate":1.,"alpha17_air_to_sulfate":1.,
            "theta_air_to_sulfate":None,"fractionation_treatment":"none"}})


def run():
    request = browser_request().solver_input()
    calls = []
    exact = joint.exact_sulfate_likelihood
    def timed(*args,**kwargs):
        started = perf_counter()
        result = exact(*args,**kwargs)
        calls.append({"seconds":perf_counter()-started,"states":np.asarray(args[0]).size,
            "roots":result.diagnostics["exact_forward_root_evaluations"],
            "maximum_level":result.diagnostics["maximum_level"]})
        print(json.dumps(calls[-1]),flush=True)
        return result
    started = perf_counter()
    joint.exact_sulfate_likelihood = timed
    try:
        with sulfate_computation_budget():
            result = constrained_coordinate_posterior(request)
    finally:
        joint.exact_sulfate_likelihood = exact
    mass = np.asarray(result.field_probability_mass).reshape(result.field_shape)
    density = np.asarray(result.field_density).reshape(result.field_shape)
    mask = np.asarray(result.field_hpd_mask).reshape(result.field_shape)
    report = {"inputs":asdict(request),"elapsed_seconds":perf_counter()-started,
              "median_ppm":result.posterior_median,"interval_ppm":result.equal_tailed_credible_interval,
              "likelihood_calls":calls,"total_roots":sum(c["roots"] for c in calls),
              "field_shape":result.field_shape,"hpd_mass":result.field_hpd_probability_mass,
              "components":int(label(mask,np.ones((3,3)))[1]),
              "sulfate_diagnostics":result.sulfate_likelihood_diagnostics}
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "sulfate_browser_budget.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(out / "sulfate_browser_budget.npz",x=result.field_x_axis,
                        y=result.field_y_axis,density=density,mass=mass,mask=mask)
    print(json.dumps(report,indent=2),flush=True)
    np.testing.assert_allclose(np.sum(mass),1.,atol=1e-10)
    assert np.array_equal(mask,density >= result.field_hpd_density_threshold)
    assert abs(mass[mask].sum()-.95) < .002


def serve(port):
    """Record local browser timings and exact requests; never used in deployment."""
    import uvicorn
    import web_api

    original = web_api.constrained_coordinate

    def recorded(request):
        started = perf_counter()
        report = {"inputs": asdict(request)}
        out = ROOT / "outputs" / "sulfate_browser_live_audit.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        try:
            result = original(request)
            report["result"] = result
            return result
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            report["elapsed_seconds"] = perf_counter() - started
            out.write_text(json.dumps(report, indent=2))
            print(f"Browser calculation: {report['elapsed_seconds']:.2f} seconds", flush=True)

    web_api.constrained_coordinate = recorded
    uvicorn.run(web_api.app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=int, metavar="PORT")
    args = parser.parse_args()
    if args.serve is not None:
        serve(args.serve)
    else:
        run()
