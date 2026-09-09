"""Compare the public high-CO2 field with independent dense pO2 integration."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from updated_constrained_pco2_posterior import (  # noqa: E402
    ConstrainedCoordinateInput, CoordinateConstraint, constrained_coordinate_posterior,
    _pair_hpd, _trapezoid_weights,
)
from updated_output_surface import load_updated_output_surface  # noqa: E402


def reference_field(x, y, *, target, sigma, po2_sigma, nodes):
    surface = load_updated_output_surface()
    z = np.linspace(max(.1, 1-4*po2_sigma), min(2.,1+4*po2_sigma), nodes)
    result = np.zeros((len(x),len(y)))
    for start in range(0,len(x),8):
        kwargs = dict(p_co2_ppm=x[start:start+8,None,None],
                      gpp_pgC_per_year=y[None,:,None],p_o2_pal=z[None,None,:])
        d17 = surface.evaluate_central_cap_delta17_grid(**kwargs)
        d18 = 1000*np.expm1(surface.evaluate_central_delta18_prime_grid(**kwargs)/1000)
        likelihood = np.exp(-.5*((d17-target)/sigma)**2-.5*((d18-23.9)/.3)**2
                            -.5*((z[None,None,:]-1)/po2_sigma)**2)
        result[start:start+8] = np.trapz(likelihood,z,axis=2)
    result *= np.exp(-.5*((y[None,:]-290)/29)**2)
    weights = np.multiply.outer(_trapezoid_weights(x),_trapezoid_weights(y))
    return result/np.sum(result*weights)


def run():
    started = perf_counter()
    result = constrained_coordinate_posterior(ConstrainedCoordinateInput(
        solve_for="pCO2",target_air_cap_delta17_permil=-10.,measurement_sigma_permil=.015,
        target_air_delta18_conventional_permil=23.9,delta18_measurement_sigma_permil=.3,
        constraints={"GPP":CoordinateConstraint("normal",center=290.,sigma=29.),
                     "pO2":CoordinateConstraint("normal",center=1.,sigma=.2)}))
    print(f"Public field completed in {perf_counter()-started:.1f}s",flush=True)
    x,y = np.array(result.field_x_axis),np.array(result.field_y_axis)
    current = np.array(result.field_density).reshape(result.field_shape)
    weights = np.multiply.outer(_trapezoid_weights(x),_trapezoid_weights(y))
    references = {}
    for nodes in (41,257,513):
        references[nodes] = reference_field(x,y,target=-10.,sigma=.015,po2_sigma=.2,nodes=nodes)
        print(f"Independent {nodes}-point reference complete",flush=True)
    reference = references[513]
    mask = np.array(result.field_hpd_mask).reshape(result.field_shape)
    reference_mass = reference*weights
    report = {"inputs": {"air_D17O_permil":-10.,"air_sigma_permil":.015,
                          "GPP_percent_modern":100.,"GPP_sigma_percent_modern":10.,
                          "pO2_PAL":1.,"pO2_sigma_PAL":.2},
              "public_median_ppm":result.posterior_median,
              "public_interval_ppm":result.equal_tailed_credible_interval,
              "numerical_integration":result.coordinate_integration_diagnostics,
              "normalized_field_L1_to_513":float(np.sum(abs(current-reference)*weights)),
              "old_41_node_L1_to_513":float(np.sum(abs(references[41]-reference)*weights)),
              "reference_257_to_513_L1":float(np.sum(abs(references[257]-reference)*weights)),
              "public_HPD_mass":result.field_hpd_probability_mass,
              "public_HPD_mass_under_reference":float(np.sum(reference_mass[mask])),
              "public_HPD_components":int(label(mask,np.ones((3,3)))[1]),
              "elapsed_seconds":perf_counter()-started}
    out = ROOT/"outputs"
    out.mkdir(exist_ok=True)
    (out/"probability_field_resolution.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/"probability_field_resolution.npz",x=x,y=y,current=current,
                        old=references[41],reference=reference,weights=weights,mask=mask)
    fig,panels = plt.subplots(1,3,figsize=(12,4.2),constrained_layout=True)
    for panel,(name,density) in zip(panels,[("Previous point sampling",references[41]),
                                           ("Cell-integrated likelihood",current),
                                           ("Independent dense reference",reference)],strict=True):
        _,local_mask,_ = _pair_hpd(density*weights,x,y,.95)
        cutoff = float(np.min(density[local_mask]))
        panel.pcolormesh(x,y/2.9,density.T,shading="auto",cmap="viridis")
        panel.contour(x,y/2.9,density.T,levels=[cutoff],colors="black",linewidths=.7)
        panel.set(title=name,xlabel=r"pCO$_2$ (ppm)",ylabel="GPP (% modern)",
                  xlim=(22000,48000),ylim=(70,140))
    fig.savefig(out/"probability_field_resolution.png",dpi=170)
    plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    assert report["normalized_field_L1_to_513"] < .002
    assert report["reference_257_to_513_L1"] < .002
    assert abs(report["public_HPD_mass_under_reference"]-.95) < .005
    return report


if __name__ == "__main__":
    run()
