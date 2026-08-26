"""Table 3 nominal state vector for diagnostics."""

from __future__ import annotations

import numpy as np

from young_model_inventory import SPECIES, TABLE3_MOLE_TARGETS


SPECIES_ORDER = [s.key for s in SPECIES]


TABLE3_MOLES = {
    **TABLE3_MOLE_TARGETS,
    # Biosphere/geosphere reservoir sizes are not tabulated in Table 3.
    "O_bio": 0.0,
    "O18_bio": 0.0,
    "O17_bio": 0.0,
    "O_geo": 0.0,
    "O18_geo": 0.0,
    "O17_geo": 0.0,
}


def table3_state() -> np.ndarray:
    return np.array([TABLE3_MOLES[name] for name in SPECIES_ORDER], dtype=float)
