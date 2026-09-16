"""Modern-reference validation helpers for scenario outputs."""

from __future__ import annotations

from typing import Any

from modern_reference_constraints import RECENT_REFERENCE_CONSTRAINTS, ReferenceConstraint


MOLAR_MASS_C_G_PER_MOL = 12.0107


def mol_c_per_year_to_pgc_per_year(mol_per_year: float) -> float:
    """Convert mol C yr-1 to PgC yr-1."""

    return mol_per_year * MOLAR_MASS_C_G_PER_MOL / 1.0e15


def permil_mol_per_year_to_permil_pgc_per_year(value: float) -> float:
    """Convert a permil*mol C yr-1 isoflux to permil*PgC yr-1."""

    return value * MOLAR_MASS_C_G_PER_MOL / 1.0e15


def _constraint(key: str) -> ReferenceConstraint:
    for item in RECENT_REFERENCE_CONSTRAINTS:
        if item.key == key:
            return item
    raise KeyError(key)


def _reference_value_text(reference: ReferenceConstraint) -> str:
    value = reference.value
    uncertainty = reference.uncertainty
    if isinstance(value, tuple):
        return f"{value[0]:g} to {value[1]:g}"
    if uncertainty is None:
        return f"{value:g}"
    if isinstance(uncertainty, tuple):
        return f"{value:g} (+{uncertainty[1]:g}/-{uncertainty[0]:g})"
    return f"{value:g} +/- {uncertainty:g}"


def _row(
    reference: ReferenceConstraint,
    model_quantity: str,
    model_value: float | None,
    model_units: str,
    comparable: bool,
    residual: float | None = None,
    residual_units: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "reference_key": reference.key,
        "source": reference.source,
        "reference_value": _reference_value_text(reference),
        "reference_units": reference.units,
        "delta_reference": reference.delta_reference,
        "model_quantity": model_quantity,
        "model_value": model_value,
        "model_units": model_units,
        "residual_model_minus_reference": residual,
        "residual_units": residual_units,
        "numeric_comparison": comparable,
        "use_for": reference.use_for,
        "note": note or reference.note,
    }


def modern_validation_rows(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a modern-reference validation table from scenario outputs.

    Only rows with `numeric_comparison=True` should be treated as residuals.
    Other rows are included because they are useful context in the UI and
    exported metadata, but they use different Delta conventions or represent
    transient/process constraints rather than the current single-run state.
    """

    o2_d17 = float(outputs["O2_trop_D17O_permil"])
    co2_trop_d17 = float(outputs["CO2_trop_D17O_permil"])
    co2_flux_pgc = permil_mol_per_year_to_permil_pgc_per_year(
        float(outputs["CO2_strat_D17O_flux_permil_mol_per_year"])
    )
    gpp_pgc = float(outputs["effective_gpp_pgC_per_year"])

    rows = []

    pack = _constraint("modern_o2_delta17o_pack_2021")
    rows.append(
        _row(
            pack,
            "O2_trop_D17O_permil",
            o2_d17,
            "permil",
            True,
            o2_d17 - float(pack.value),
            "permil",
            "Direct modern O2 anchor for the updated physical branch; keep convention metadata in manuscripts.",
        )
    )

    adnew_flux = _constraint("adnew_2025_strat_trop_co2_delta17o_net_isoflux")
    rows.append(
        _row(
            adnew_flux,
            "CO2_strat_D17O_flux converted from permil mol yr-1",
            co2_flux_pgc,
            "permil PgC yr-1",
            True,
            co2_flux_pgc - float(adnew_flux.value),
            "permil PgC yr-1",
            "Approximate box-model analogue of the net isotope flux; useful as a scale check, not yet a fitted target.",
        )
    )

    adnew_gpp = _constraint("adnew_2025_gpp_from_utls_co2_delta17o")
    rows.append(
        _row(
            adnew_gpp,
            "terrestrial GPP not separated by the global model",
            None,
            "",
            False,
            None,
            None,
            "Context only: Adnew's FA = 0.88 x GPP relation constrains terrestrial leaf assimilation, not total terrestrial-plus-marine GPP.",
        )
    )

    liang_gpp = _constraint("liang_2023_global_gpp_o2_co2_delta17o")
    rows.append(
        _row(
            liang_gpp,
            "effective_gpp_pgC_per_year",
            gpp_pgc,
            "PgC yr-1",
            True,
            gpp_pgc - float(liang_gpp.value),
            "PgC yr-1",
            "Useful for choosing the meaning of 100% modern GPP in updated model runs.",
        )
    )

    koren_global = _constraint("koren_2019_global_mean_co2_delta17o")
    rows.append(
        _row(
            koren_global,
            "CO2_trop_D17O_permil converted to per meg",
            co2_trop_d17 * 1000.0,
            "per meg",
            False,
            None,
            None,
            "Context only: Koren uses lambda_RL=0.5229 and a 3-D lower-atmosphere CO2 field.",
        )
    )

    koren_mauna_loa = _constraint("koren_2019_mauna_loa_co2_delta17o")
    rows.append(
        _row(
            koren_mauna_loa,
            "CO2_trop_D17O_permil converted to per meg",
            co2_trop_d17 * 1000.0,
            "per meg",
            False,
            None,
            None,
            "Context only: site-specific 3-D prediction, not directly comparable to the one-box troposphere.",
        )
    )

    for key in (
        "adnew_2025_co2_isotope_turnover",
        "liang_2023_co2_isotope_recycling_time",
        "steur_2024_lutjewad_measured_delta17o_range",
        "thiemens_2014_enso_co2_delta17o_excursion",
        "crockford_2018_photochemical_o2_threshold",
        "liu_2021_mid_proterozoic_low_ch4_pco2_floor",
    ):
        reference = _constraint(key)
        rows.append(
            _row(
                reference,
                "not a single-run model output",
                None,
                "",
                False,
                None,
                None,
                reference.note,
            )
        )

    return rows
