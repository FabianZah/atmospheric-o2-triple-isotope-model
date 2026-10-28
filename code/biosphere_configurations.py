"""Fixed biological configurations for the shared atmospheric isotope budget."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from math import log
from pathlib import Path

import numpy as np

from biological_o2_ensemble import central_biological_member, fixed_po2_partitioned_biological_budget
from global_o2_isotope_reservoir import BiologicalO2Compartment, PartitionedBiologicalO2Budget
from young_global_o2_budget import GLOBAL_MAJOR_O2_MOLES_1PAL, TROPOSPHERIC_FRACTION

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_data/biosphere_configurations_v1.json"
DEFAULT_BIOSPHERE = "modern_reference"


def output_surface_path(name=DEFAULT_BIOSPHERE):
    configuration(name)
    filename = ("updated_molecular_output_surface_v1.json" if name == DEFAULT_BIOSPHERE
                else "marine_molecular_output_surface_v1.json")
    return REGISTRY_PATH.parent / filename


def select_surface_path(path, name=DEFAULT_BIOSPHERE):
    """Resolve the standard cache for a configuration, preserving explicit paths."""
    configuration(name)
    path = Path(path).resolve()
    if path == output_surface_path(DEFAULT_BIOSPHERE).resolve():
        return output_surface_path(name)
    return path


@lru_cache(maxsize=1)
def _registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def configuration(name=DEFAULT_BIOSPHERE):
    try:
        return deepcopy(_registry()["configurations"][name])
    except KeyError as error:
        raise ValueError(f"Unknown biosphere configuration: {name}") from error


def marine_factors(config=None):
    """Mix uptake ratios; include dissolved/air equilibrium exactly once."""
    c = configuration("marine_endmember") if config is None else config
    w, v = c["surface_respiration_fraction"], c["interior_respiration_fraction"]
    if not np.isclose(w+v, 1., rtol=0., atol=1e-14) or min(w, v) < 0:
        raise ValueError("Marine respiration fractions must be nonnegative and sum to one")
    equilibrium18 = 1+(c["equilibrium_epsilon18_intercept_permil"]
        +c["equilibrium_epsilon18_temperature_slope_permil_per_c"]
        *c["reference_sea_surface_temperature_c"])/1000
    equilibrium17 = equilibrium18**c["equilibrium_log_exponent"]
    surface18, interior18 = c["surface_respiration_alpha18"], c["interior_respiration_alpha18"]
    alpha18 = equilibrium18*(w*surface18+v*interior18)
    alpha17 = equilibrium17*(w*surface18**c["surface_respiration_log_exponent"]
        +v*interior18**c["interior_respiration_log_exponent"])
    return dict(alpha18=alpha18, alpha17=alpha17, log_exponent=log(alpha17)/log(alpha18),
                equilibrium_alpha18=equilibrium18, equilibrium_alpha17=equilibrium17)


def configured_budget(name, request, photo, *, biological_member=None,
                      marine_accessible_fraction=1., marine_accessibility_source=None):
    c = configuration(name)
    if name == DEFAULT_BIOSPHERE:
        return fixed_po2_partitioned_biological_budget(
            central_biological_member() if biological_member is None else biological_member,
            po2_pal=request.p_o2_pal, gpp_pgC_per_year=request.gpp_pgC_per_year,
            photochemical=photo, marine_accessible_fraction=marine_accessible_fraction,
            marine_accessibility_source=marine_accessibility_source)
    if biological_member is not None or marine_accessible_fraction != 1.:
        raise ValueError("The fixed marine configuration requires its own biology and full accessibility")
    factors = marine_factors(c)
    production = request.gpp_pgC_per_year*1e15/12.0107
    major = GLOBAL_MAJOR_O2_MOLES_1PAL*request.p_o2_pal
    rate = (production+photo.o16o16)/(TROPOSPHERIC_FRACTION*major)
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("Nonpositive marine respiratory closure")
    compartment = BiologicalO2Compartment(
        key="marine", photosynthetic_o16o16_mol_per_year=production,
        respiration_rate_per_year=rate, tropospheric_fraction=TROPOSPHERIC_FRACTION,
        alpha_respiration_18=factors["alpha18"], beta_respiration_17=factors["log_exponent"],
        source_alpha_18=c["source_alpha18"], source_beta_17=.528,
        source=f"{c['version']}; {c['source_locations']}; gas equilibrium included once")
    return PartitionedBiologicalO2Budget((compartment,), source=c["description"])
