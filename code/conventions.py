"""Named literature and diagnostic conventions for under-determined parameters.

Young et al. (2014) did not print (or their unavailable source code did not fully
determine) several conventions that the reconstruction needs. This module maps
each to an explicit literature or diagnostic choice. Diagnostic conventions are
labelled as such and must not be presented as values derived from the paper.

Each convention exposes the ModelConfig field overrides it implies, so a preset
or scenario can select a convention by name and the model records the citation in
its outputs. See docs/tier2_first_principles_design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class Convention:
    key: str
    description: str
    citation: str
    parameters: dict[str, Any]


# --- R7: O(1D) + CO2 throughput -------------------------------------------------
# Young's printed kR7a (4.46e-11 cm3/s) cannot reproduce Table 3 / Fig. 8; the
# reconstruction needs ~2.4-2.5x. That factor is consistent with the Yung et al.
# (1991) O(1D)+CO2 rate (see code/r7_rate_sources.py), so we cite it rather than
# tune it.
R7_O1D_CO2_RATE: dict[str, Convention] = {
    "young_printed": Convention(
        "young_printed",
        "Young Table 2 printed kR7a (4.46e-11 cm3/s); does not reproduce Table 3/Fig. 8.",
        "Young et al. (2014) Table 2",
        {"r7_throughput_factor": 1.0},
    ),
    "yung_1991": Convention(
        "yung_1991",
        "Yung et al. (1991) O(1D)+CO2 rate, 298 K text value k=1.1e-10 cm3/s (x2.466 vs Young printed).",
        "Yung, DeMore & Pinto (1991)",
        {"r7_throughput_factor": 2.466},
    ),
    "yung_1991_220k": Convention(
        "yung_1991_220k",
        "Yung et al. (1991) Arrhenius form at 220 K stratosphere (x2.863 vs Young printed); overdrives Fig. 8.",
        "Yung, DeMore & Pinto (1991)",
        {"r7_throughput_factor": 2.863},
    ),
}
R7_DEFAULT = "yung_1991"


# --- R5: O(1D) quenching by M ---------------------------------------------------
# Young Table 2 footnote (d) states k5a[M] = k[O2] with k = 2.93e-11,
# but its literal direction conflicts with the printed k5a. The effective
# 5.589807e18 mol normalization used by dynamic_air_calibrated was inferred by
# balancing the reconstructed Table 3 state; it is not printed by Young.
R5_M_QUENCH: dict[str, Convention] = {
    "footnote_d_dynamic_air": Convention(
        "footnote_d_dynamic_air",
        "Diagnostic Table 3-balanced effective [M], scaled with dynamic air; Young footnote d remains unresolved.",
        "Young et al. (2014) Table 2 and footnote d; diagnostic reconstruction normalization",
        {"r5_mode": "dynamic_air_calibrated"},
    ),
    "whole_stratosphere": Convention(
        "whole_stratosphere",
        "Printed kR5a with whole-stratosphere [M]=1.8e19; diagnostic, does not match Eq. 26.",
        "Young et al. (2014) Table 2",
        {"r5_mode": "whole_stratosphere"},
    ),
}
R5_DEFAULT = "footnote_d_dynamic_air"


# --- O(1D) Delta'17O (via aMIF) -------------------------------------------------
# Young Section 6: the O(1D) composition is "the ultimate driver" for O2 Delta'17O
# and the most uncertain input. Sets the saturation cap delta_eq = -(O(1D) D17O).
O1D_D17O: dict[str, Convention] = {
    "young": Convention(
        "young",
        "O(1D) Delta'17O = 27 permil (aMIF 1.065); reproduces modern O2 and CO2 isoflux.",
        "Young et al. (2014) Section 3.2/6",
        {"a_mif": 1.065},
    ),
    "barkan_luz": Convention(
        "barkan_luz",
        "Young sensitivity case with O(1D) Delta'17O = 45.9 permil (aMIF 1.109), adjusted to match the "
        "Barkan & Luz direct-air O2 target; it raises the modeled CO2 isoflux ~4x.",
        "Barkan & Luz (2011); Young et al. (2014) Section 6",
        {"a_mif": 1.109},
    ),
}
O1D_D17O_DEFAULT = "young"


# --- R8: CO2-H2O isotope exchange time ------------------------------------------
R8_EXCHANGE_TIME: dict[str, Convention] = {
    "young_1yr": Convention(
        "young_1yr",
        "1-year CO2-H2O exchange e-folding (fast end of Young's stated 1-2 yr range).",
        "Young et al. (2014) Section 3.5",
        {"r8_rate_factor": 1.0},
    ),
    "young_2yr": Convention(
        "young_2yr",
        "2-year CO2-H2O exchange e-folding (slow end of Young's stated 1-2 yr range).",
        "Young et al. (2014) Section 3.5",
        {"r8_rate_factor": 0.5},
    ),
}
R8_DEFAULT = "young_1yr"


# --- Slow O2 / organic-burial closure -------------------------------------------
BURIAL_CLOSURE: dict[str, Convention] = {
    "young_inverse_o2": Convention(
        "young_inverse_o2",
        "Young-style inverse-pO2 organic burial closure for the slow O2 budget.",
        "Young et al. (2014) Section 3.6",
        {"closure_mode": "modern"},
    ),
    "lasaga_ohmoto": Convention(
        "lasaga_ohmoto",
        "Lasaga & Ohmoto (2002) burial feedback; literature-extended low-O2 behavior.",
        "Lasaga & Ohmoto (2002)",
        {"closure_mode": "modern_lasaga_burial"},
    ),
}
BURIAL_DEFAULT = "young_inverse_o2"


_REGISTRY: dict[str, tuple[dict[str, Convention], str]] = {
    "r7": (R7_O1D_CO2_RATE, R7_DEFAULT),
    "r5": (R5_M_QUENCH, R5_DEFAULT),
    "o1d_d17o": (O1D_D17O, O1D_D17O_DEFAULT),
    "r8": (R8_EXCHANGE_TIME, R8_DEFAULT),
    "burial": (BURIAL_CLOSURE, BURIAL_DEFAULT),
}

DEFAULT_CONVENTIONS: dict[str, str] = {name: default for name, (_opts, default) in _REGISTRY.items()}


def get_convention(group: str, key: str) -> Convention:
    options, _default = _REGISTRY[group]
    if key not in options:
        raise KeyError(f"unknown {group} convention '{key}'; options: {sorted(options)}")
    return options[key]


def convention_parameters(selections: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve a set of convention selections to ModelConfig field overrides.

    Missing groups fall back to their documented defaults.
    """
    chosen = dict(DEFAULT_CONVENTIONS)
    if selections:
        chosen.update(selections)
    params: dict[str, Any] = {}
    for group, key in chosen.items():
        params.update(get_convention(group, key).parameters)
    return params


def conventions_metadata(selections: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Citation metadata for the active conventions, for scenario outputs."""
    chosen = dict(DEFAULT_CONVENTIONS)
    if selections:
        chosen.update(selections)
    rows = []
    for group, key in chosen.items():
        conv = get_convention(group, key)
        rows.append(
            {"group": group, "convention": conv.key, "description": conv.description, "citation": conv.citation}
        )
    return rows


def _parameter_values_match(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1.0e-10, abs_tol=1.0e-12)
    return actual == expected


def conventions_metadata_for_parameters(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Identify active conventions from the resolved model configuration.

    A preset may use a value that does not exactly match a named literature
    convention. Such a case is recorded as ``custom`` with the actual values;
    it must never be silently labelled as the registry default.
    """

    rows: list[dict[str, Any]] = []
    for group, (options, _default) in _REGISTRY.items():
        matches = [
            convention
            for convention in options.values()
            if all(
                field in parameters and _parameter_values_match(parameters[field], expected)
                for field, expected in convention.parameters.items()
            )
        ]
        if len(matches) == 1:
            convention = matches[0]
            rows.append(
                {
                    "group": group,
                    "convention": convention.key,
                    "description": convention.description,
                    "citation": convention.citation,
                    "parameters": dict(convention.parameters),
                    "matched_named_convention": True,
                }
            )
            continue

        fields = sorted({field for convention in options.values() for field in convention.parameters})
        rows.append(
            {
                "group": group,
                "convention": "custom",
                "description": "Preset or scenario values do not exactly match one named convention.",
                "citation": "Resolved scenario configuration",
                "parameters": {field: parameters.get(field) for field in fields},
                "matched_named_convention": False,
            }
        )
    return rows
