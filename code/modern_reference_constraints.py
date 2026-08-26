"""Recent literature constraints for the updated physical model branch.

These entries are not Young et al. (2014) validation targets and are not
equations for the 27-ODE reconstruction. They are reference anchors for later
model comparison, UI display, and publication metadata. The `delta_reference`
field is intentionally explicit because the CO2 literature uses several
different Delta17O conventions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceConstraint:
    key: str
    value: float | tuple[float, float]
    uncertainty: float | tuple[float, float] | None
    units: str
    source: str
    delta_reference: str
    use_for: str
    note: str = ""


RECENT_REFERENCE_CONSTRAINTS: tuple[ReferenceConstraint, ...] = (
    ReferenceConstraint(
        key="modern_o2_delta17o_pack_2021",
        value=-0.432,
        uncertainty=0.015,
        units="permil",
        source="Pack, 2021",
        delta_reference="Delta'17O, lambda convention to be kept with source paper",
        use_for="updated physical model modern O2 anchor",
        note=(
            "Modern atmospheric O2 mean and stated analytical uncertainty. The "
            "observation is an independent validation interval, not an output offset."
        ),
    ),
    ReferenceConstraint(
        key="adnew_2025_caribic_ut_co2_delta18o",
        value=41.78933333333333,
        uncertainty=0.096969835,
        units="permil VSMOW",
        source="Adnew et al., 2025, Supplement Table S2",
        delta_reference="conventional delta18O; n=33 selected by N2O >= 313.5 ppb",
        use_for="modern 10 km CO2 lower-boundary composition",
        note=(
            "Arithmetic mean and standard error calculated directly from the "
            "published upper-tropospheric rows; sample SD is 0.55705 per mil. "
            "Convert to delta18-prime before use in the isotope solver."
        ),
    ),
    ReferenceConstraint(
        key="adnew_2025_caribic_ut_co2_cap_delta17_prime",
        value=-0.2186969696969697,
        uncertainty=0.002951786,
        units="permil",
        source="Adnew et al., 2025, Supplement Table S2",
        delta_reference="Delta-prime-17O(CO2), lambda_ref=0.528; n=33",
        use_for="modern 10 km CO2 lower-boundary composition",
        note=(
            "Arithmetic mean and standard error calculated directly from the "
            "published upper-tropospheric rows; sample SD is 0.01696 per mil."
        ),
    ),
    ReferenceConstraint(
        key="adnew_2025_strat_trop_co2_delta17o_net_isoflux",
        value=51.3,
        uncertainty=1.6,
        units="permil PgC yr-1",
        source="Adnew et al., 2025",
        delta_reference="Delta'17O(CO2), lambda_ref=0.528",
        use_for="modern CO2 stratospheric-source validation",
        note="Derived from the N2O-Delta'17O(CO2) slope in UTLS samples; much tighter than older flux estimates.",
    ),
    ReferenceConstraint(
        key="adnew_2025_gpp_from_utls_co2_delta17o",
        value=211.0,
        uncertainty=26.0,
        units="PgC yr-1",
        source="Adnew et al., 2025",
        delta_reference="Delta'17O(CO2), lambda_ref=0.528",
        use_for="modern terrestrial GPP comparison",
        note=(
            "Terrestrial leaf-assimilation GPP from FA = 0.88 x GPP; not a "
            "terrestrial-plus-marine global O2-production normalization. Sensitive "
            "to cm/ca, soil invasion, and UT CO2 Delta'17O."
        ),
    ),
    ReferenceConstraint(
        key="adnew_2025_co2_isotope_turnover",
        value=0.98,
        uncertainty=0.10,
        units="yr",
        source="Adnew et al., 2025",
        delta_reference="Delta'17O(CO2), lambda_ref=0.528",
        use_for="modern CO2 isotope turnover comparison",
        note="Surface-flux isotope turnover, not the same parameter as Young's R8 CO2-H2O exchange rate.",
    ),
    ReferenceConstraint(
        key="liang_2023_global_gpp_o2_co2_delta17o",
        value=290.0,
        uncertainty=30.0,
        units="PgC yr-1",
        source="Liang et al., 2023",
        delta_reference="linear/log Delta17O budget, lambda=0.516 for CO2 budget",
        use_for="modern absolute GPP scaling option",
        note="Agrees with Dole-effect estimate of 292 +/- 20 PgC yr-1; do not silently equate with Young 100% GPP.",
    ),
    ReferenceConstraint(
        key="liang_2023_terrestrial_gpp",
        value=(170.0, 200.0),
        uncertainty=None,
        units="PgC yr-1",
        source="Liang et al., 2023",
        delta_reference="linear/log Delta17O budget, lambda=0.516 for CO2 budget",
        use_for="modern terrestrial GPP comparison",
        note="Published as an approximate range.",
    ),
    ReferenceConstraint(
        key="liang_2023_co2_isotope_recycling_time",
        value=1.5,
        uncertainty=0.2,
        units="yr",
        source="Liang et al., 2023",
        delta_reference="lambda=0.516 primary; lambda=0.528 gives similar central value with larger error",
        use_for="modern CO2 isotope turnover comparison",
        note="Atmospheric CO2 oxygen-isotope recycling time from surface flux balance.",
    ),
    ReferenceConstraint(
        key="koren_2019_global_mean_co2_delta17o",
        value=39.6,
        uncertainty=None,
        units="per meg",
        source="Koren et al., 2019",
        delta_reference="Delta17O(CO2), lambda_RL=0.5229",
        use_for="3-D model validation benchmark",
        note="Base TM5 prediction for the lowest 500 m of the atmosphere.",
    ),
    ReferenceConstraint(
        key="koren_2019_mauna_loa_co2_delta17o",
        value=36.2,
        uncertainty=None,
        units="per meg",
        source="Koren et al., 2019",
        delta_reference="Delta17O(CO2), lambda_RL=0.5229",
        use_for="background-site CO2 validation benchmark",
        note="Base TM5 prediction for Mauna Loa; South Pole prediction is 52.5 per meg.",
    ),
    ReferenceConstraint(
        key="steur_2024_lutjewad_measured_delta17o_range",
        value=(-0.27, -0.16),
        uncertainty=None,
        units="permil",
        source="Steur et al., 2024",
        delta_reference="Delta17O(CO2), logarithmic lambda_RL=0.528",
        use_for="modern CO2 variability validation",
        note="Moving-average observed range; original Koren-style model underpredicts variability.",
    ),
    ReferenceConstraint(
        key="thiemens_2014_enso_co2_delta17o_excursion",
        value=-0.06,
        uncertainty=None,
        units="permil change",
        source="Thiemens et al., 2014",
        delta_reference="Delta17O(CO2), paper convention",
        use_for="transient CO2 isotope response sanity check",
        note="Negative ENSO-period excursion; simple CO2 addition and increased STE alone were judged insufficient.",
    ),
    ReferenceConstraint(
        key="crockford_2018_photochemical_o2_threshold",
        value=0.001,
        uncertainty=None,
        units="PAL O2",
        source="Crockford et al., 2018",
        delta_reference="Delta17O = delta17O - 0.5305 * delta18O",
        use_for="low-O2 model-domain warning",
        note="Below roughly 0.1% PAL O2, the O-MIF signal is not expected to form in the same way.",
    ),
    ReferenceConstraint(
        key="liu_2021_mid_proterozoic_low_ch4_pco2_floor",
        value=3600.0,
        uncertainty=None,
        units="ppmv CO2",
        source="Liu et al., 2021",
        delta_reference="Delta'17O sedimentary O-MIF framework",
        use_for="paleo scenario context",
        note="Approximate lower pCO2 limit at 1.4 Ga under low methane; not an Ordovician constraint.",
    ),
)


def constraints_by_use(use_for: str) -> tuple[ReferenceConstraint, ...]:
    """Return reference constraints for a given intended use."""

    return tuple(item for item in RECENT_REFERENCE_CONSTRAINTS if item.use_for == use_for)
