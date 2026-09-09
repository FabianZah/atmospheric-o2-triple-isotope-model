"""Independent equation checks for the conditional literature-pathway audit."""
from dataclasses import replace
import json

import numpy as np
import pytest

import audit_sulfate_fractionation as a


@pytest.mark.parametrize("water18", [-20., -10., 0.])
def test_water_and_reference_slope(water18):
    r = a.water_ratios(water18)
    pair = a.isotope_pair(r)
    assert pair["delta18_permil"] == pytest.approx(water18, abs=1e-10)
    assert pair["cap_delta17_permil"] == pytest.approx(.033, abs=1e-10)
    native = a.isotope_pair(r, .5305)
    assert native["cap_delta17_permil"] == pytest.approx(.033-.0025*1000*np.log1p(water18/1000), abs=1e-10)


def test_equation16_printed_ratios_and_independent_atom_mixture():
    w, o = a.water_ratios(-10.), a.ratios(24.2, -.554, .5305)
    nonair = w * np.array([1.0129**.524*.9916**.511, 1.0129*.9916])
    transferred = o * np.array([.985**.511, .985])
    printed = .75*nonair + .25*transferred
    assert np.allclose(a.formation("dissolved_o2", w, o, exact_atoms=False)["ratios"], printed, atol=1e-18, rtol=0)
    b, t = np.r_[1., nonair], np.r_[1., transferred]
    mix = .75*b/b.sum() + .25*t/t.sum()
    exact = a.formation("dissolved_o2", w, o)
    assert np.allclose(exact["ratios"], mix[1:]/mix[0], atol=1e-18, rtol=0)
    assert exact["fraction"] == .25
    assert a.isotope_pair(exact["nonair_ratios"])["cap_delta17_permil"] == pytest.approx(
        .033 + 1000*(.524-.528)*np.log(1.0129) + 1000*(.511-.528)*np.log(.9916), abs=1e-10)


@pytest.mark.parametrize("pathway,alpha", [("surface", .9916), ("ferric", .9984)])
def test_water_only_equations_and_absence_of_air_sensitivity(pathway, alpha):
    w = a.water_ratios(-10.)
    r1 = a.formation(pathway, w, a.ratios(24., -.4), exact_atoms=False)
    r2 = a.formation(pathway, w, a.ratios(24., -20.), exact_atoms=False)
    expected = w*(.75*np.array([1.0129**.524*alpha**.511, 1.0129*alpha])
                  + .25*np.array([.9976**.511, .9976]))
    assert np.allclose(r1["ratios"], expected, rtol=0, atol=1e-18)
    assert np.array_equal(r1["ratios"], r2["ratios"])
    assert r1["fraction"] == 0


def test_equilibrium_uses_equation6_not_rounded_table_alpha():
    w = a.water_ratios(0.)
    alpha = np.exp((2.68e6/298.15**2-7.45)/1000)
    result = a.formation("equilibrium", w, a.ratios(24., -10.))
    assert np.allclose(result["ratios"], w*np.array([alpha**.524, alpha]), rtol=0, atol=1e-18)
    assert abs(alpha-1.023) > 1e-5
    assert result["fraction"] == 0


@pytest.mark.parametrize("water18", [-20., -10., 0.])
@pytest.mark.parametrize("anomaly", [-.432, -10., -20.])
@pytest.mark.parametrize("mode", ["identity", "paper_pair_ratio"])
def test_compact_transfer_recovers_full_pathway(water18, anomaly, mode):
    air = {"delta18_permil": 23.9, "cap_delta17_permil": anomaly}
    row = a.comparison(air, water18, mode)
    recovered = row["variants"]["literature"]
    assert recovered["air_bias_permil"] == pytest.approx(0., abs=1e-8)
    assert recovered["sulfate_residual_permil"] == pytest.approx(0., abs=1e-8)
    assert recovered["implied_nonair_delta18_permil"] == pytest.approx(row["nonair"]["delta18_permil"], abs=1e-8)
    assert recovered["atom_closure_residual"] < 1e-12
    assert row["variants"]["no_air_fractionation"]["air_bias_permil"] > 0


def test_gas_bridge_keeps_both_isotope_ratios():
    gas = a.dissolution_factor("paper_pair_ratio")
    p = a.LiteratureParameters()
    total = gas*np.array([.985**.511, .985])
    r = a.formation("dissolved_o2", a.water_ratios(0.), a.ratios(24.2, -.554, .5305))
    process = a.compact_process(r, gas, p)
    assert np.allclose(a.fractionation(process.alpha18_air_to_sulfate, process.theta_air_to_sulfate), total, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="cannot be represented"):
        a.compact_process(r, np.array([1.001, 1.]), replace(p, o2_kinetic=1.))


@pytest.mark.parametrize("po2,gpp,co2", [(1., 290., 294.), (1., 290., 30000.), (.5, 72.5, 60000.)])
def test_co2_roundtrip_and_grid_refinement(po2, gpp, co2):
    row = a.comparison(a.atmospheric_pair(po2, co2, gpp), 0.)
    for variant in row["variants"].values():
        process = a.SulfateProcessAssumptions(**variant["process"])
        low = a.solve_pco2(row["sulfate"], process, po2=po2, gpp=gpp, nodes=513)
        high = a.solve_pco2(row["sulfate"], process, po2=po2, gpp=gpp, nodes=1025)
        assert low["status"] == high["status"]
        assert high["roots_ppm"] == pytest.approx(low["roots_ppm"], abs=1e-4)
        if high["roots_ppm"]:
            assert high["maximum_sulfate_residual_permil"] < 1e-8
    correct = a.SulfateProcessAssumptions(**row["variants"]["literature"]["process"])
    assert a.solve_pco2(row["sulfate"], correct, po2=po2, gpp=gpp)["roots_ppm"] == pytest.approx([co2], abs=1e-4)


def test_missing_modern_root_is_not_clamped():
    row = a.comparison(a.atmospheric_pair(1., 294., 290.), 0.)
    process = a.SulfateProcessAssumptions(**row["variants"]["no_air_fractionation"]["process"])
    result = a.solve_pco2(row["sulfate"], process, po2=1., gpp=290.)
    assert result["status"] == "no_root_in_domain"
    assert result["roots_ppm"] == []
    assert result["maximum_sulfate_residual_permil"] is None


def test_bracket_search_retains_multiple_crossings(monkeypatch):
    def pair(po2, pco2, gpp):
        return {"delta18_permil": 23.9, "cap_delta17_permil": -.432 + (pco2-250)*(pco2-750)*1e-5}
    def curve(po2, gpp, nodes):
        x = np.linspace(50., 1000., nodes)
        return x, pair(po2, x, gpp)["cap_delta17_permil"], np.full_like(x, 23.9)
    monkeypatch.setattr(a, "atmospheric_pair", pair)
    monkeypatch.setattr(a, "atmospheric_curve", curve)
    row = a.comparison(pair(1., 250., 290.), 0.)
    process = a.SulfateProcessAssumptions(**row["variants"]["literature"]["process"])
    result = a.solve_pco2(row["sulfate"], process, po2=1., gpp=290.)
    assert result["status"] == "multiple_roots"
    assert result["roots_ppm"] == pytest.approx([250., 750.], abs=1e-5)


def test_complete_audit_serialization_and_numerical_contracts():
    report = a.build_report()
    json.dumps(report, allow_nan=False)
    assert report["summary"]["matched_transfer_cases"] == 80
    assert report["summary"]["matched_transfer_max_air_error_permil"] < 1e-8
    assert report["summary"]["matched_transfer_max_pco2_error_ppm"] < 1e-3
    assert report["max_printed_vs_exact_atom_anomaly_difference_permil"] < 1e-5
    assert all(r["converged"] for r in report["live_central_checks"])
    assert max(abs(r["surface_minus_live_air_permil"]["cap_delta17_permil"]) for r in report["live_central_checks"]) < .01


@pytest.mark.parametrize("log18,native_air_product", [
    (-21.6, -.1544), (-9.8, -.35618), (23.3, -.92219),
])
def test_peng_figure7_native_coordinates(log18, native_air_product):
    transfer = a.peng_2026_apparent_transfer(log18)
    assert -.5+transfer["anomaly_shift_5305_permil"] == pytest.approx(native_air_product, abs=1e-12)
    assert transfer["anomaly_shift_528_permil"] == pytest.approx(
        transfer["anomaly_shift_5305_permil"]+.0025*log18, abs=1e-12)
    air = a.ratios(23.9, -.432)
    product = air*np.array([transfer["alpha17"], transfer["alpha18"]])
    assert a.isotope_pair(product)["cap_delta17_permil"]+.432 == pytest.approx(
        transfer["anomaly_shift_528_permil"], abs=1e-10)


def test_peng_log_factors_are_exponentiated_not_conventional_epsilon():
    transfer = a.peng_2026_apparent_transfer(-21.6)
    assert transfer["alpha18"] == pytest.approx(np.exp(-.0216), abs=1e-15)
    assert abs(transfer["alpha18"]-(1-.0216)) > .0002
    assert transfer["theta"] == pytest.approx(.5145, abs=1e-14)
    assert transfer["anomaly_shift_528_permil"] == pytest.approx(.2916, abs=1e-12)


def test_peng_net_zero_delta18_retains_finite_anomaly():
    transfer = a.peng_2026_apparent_transfer(0.)
    assert transfer["alpha18"] == 1.
    assert transfer["alpha17"] == pytest.approx(np.exp(-.02376/1000), abs=1e-15)
    assert transfer["anomaly_shift_528_permil"] == pytest.approx(-.02376, abs=1e-12)
    assert transfer["theta"] is None
    assert transfer["alpha17"] != 1.


def test_peng_reported_equilibrium_remains_separate_from_rounded_figure_line():
    report = a.source_2026_report()
    equilibrium = next(r for r in report["peng"]["transfers"] if r["name"] == "reported_SO5_O2_equilibrium")
    assert equilibrium["anomaly_shift_528_permil"] == pytest.approx(.02835, abs=1e-12)
    assert equilibrium["anomaly_shift_5305_permil"] == pytest.approx(.03710, abs=1e-12)
    # The independent rounded equilibrium theta is not silently replaced by
    # the ratio implied by the schematic backward-process line.
    assert equilibrium["anomaly_shift_528_permil"]-a.peng_2026_apparent_transfer(-3.5)["anomaly_shift_528_permil"] == pytest.approx(.00101, abs=1e-12)
    assert report["peng"]["forward"]["log18_2se"] == .6
    assert report["peng"]["forward"]["n"] == 4
    assert report["peng"]["equilibrium"]["n"] == 3


@pytest.mark.parametrize("log18", [-21.6001, 23.3001, np.nan, np.inf])
def test_peng_construction_does_not_extrapolate_missing_process_law(log18):
    with pytest.raises(ValueError, match="illustrated"):
        a.peng_2026_apparent_transfer(log18)


@pytest.mark.parametrize("species,log18,log17,theta,gap", [
    ("bisulfite", 14.879356028844544, 7.712320643971157, .5202, -.02792036223377),
    ("sulfite", 9.542968304544692, 4.910736206607413, .5155, -.00866395438538),
])
def test_wei_retains_paired_equations_and_reported_mean_theta(species, log18, log17, theta, gap):
    paired = a.wei_2026_transfer(species, 25., "paired_printed_equations")
    exponent = a.wei_2026_transfer(species, 25., "reported_mean_theta")
    assert paired["log18_permil"] == pytest.approx(log18, abs=1e-12)
    assert paired["log17_permil"] == pytest.approx(log17, abs=1e-12)
    assert exponent["theta"] == pytest.approx(theta, abs=1e-14)
    assert paired["anomaly_shift_528_permil"]-exponent["anomaly_shift_528_permil"] == pytest.approx(gap, abs=1e-12)
    row = next(r for r in a.source_2026_report()["wei"]["comparisons_25C"] if r["species"] == species)
    assert row["rounding_could_cover_25C_gap"] is True
    assert row["independent_last_digit_rounding_bound_permil"] > abs(gap)


def test_wei_table_preserves_native_compositions_errors_and_printed_differences():
    report = a.source_2026_report()["wei"]
    rows = report["observations"]
    assert len(rows) == 16
    assert {r["temperature_c"] for r in rows} == {12., 25., 40., 55.}
    assert {r["pH"] for r in rows} == {4.60, 6.28, 7.30, 8.89}
    assert all(r["n"] == 2 for r in rows)
    assert report["water"]["n"] == 5
    assert report["water"]["delta18_sd"] == .013
    row = rows[1]
    assert row["anomaly_5305_permil"] == -.106
    assert row["anomaly_sd"] == .003
    assert row["printed_anomaly_difference_permil"] == -.153
    assert row["printed_minus_subtracted_anomaly_difference_permil"] == pytest.approx(.001, abs=1e-14)
    expected = 1000*np.log((1000+6.771)/(1000-9.010))
    assert row["log18_fractionation_permil"] == pytest.approx(expected, abs=1e-12)
    assert row["anomaly_difference_528_from_compositions_permil"] == pytest.approx(-.154+.0025*expected, abs=1e-12)


@pytest.mark.parametrize("temperature", [11.99, 55.01, np.nan, np.inf])
def test_wei_transfer_stays_inside_experimental_temperature_range(temperature):
    with pytest.raises(ValueError, match="measured"):
        a.wei_2026_transfer("sulfite", temperature, "reported_mean_theta")


def test_source_terms_do_not_change_2021_coefficients_or_sulfite_kinetic_factor():
    before = a.LiteratureParameters()
    a.source_2026_report()
    assert a.LiteratureParameters() == before
    assert before.theta_kinetic == .511
    assert before.sulfite_o2_kinetic == .9916
