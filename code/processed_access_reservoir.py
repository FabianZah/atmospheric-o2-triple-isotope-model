"""Optional processed-access reservoir for transient Young-style experiments.

The accepted v2 reconstruction uses an instantaneous pO2 threshold proxy for
the low-GPP processed-column recovery term. This module makes the same proxy a
real scalar state variable so transient tests can ask whether the access term is
fast, slow, or effectively instantaneous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from table3_state import SPECIES_ORDER, TABLE3_MOLES


ACCESS_SPECIES = "processed_access_state"
EXTENDED_SPECIES_ORDER = [*SPECIES_ORDER, ACCESS_SPECIES]


def threshold_access_target(po2_pal: float, half_pal: float = 0.85, hill: float = 8.0) -> float:
    """Return the v2 pO2 threshold-access target, normalized to 1 at modern pO2."""

    half = max(half_pal, 1.0e-30)
    power = max(hill, 1.0e-30)

    def sigmoid(value: float) -> float:
        x = max(value, 1.0e-30) / half
        return x**power / (1.0 + x**power)

    return min(sigmoid(po2_pal) / max(sigmoid(1.0), 1.0e-30), 1.0)


def extend_state_with_access(y: np.ndarray, access: float = 1.0) -> np.ndarray:
    """Append the scalar processed-access state to a normal 27-species vector."""

    return np.append(y, float(access))


@dataclass(frozen=True)
class AccessStateRelaxation:
    """First-order relaxation of the processed-access state toward pO2 access."""

    tau_yr: float
    half_pal: float = 0.85
    hill: float = 8.0

    @property
    def key(self) -> str:
        return "processed_access_state_relaxation"

    def apply_state(self, dydt: np.ndarray, y: np.ndarray, index: Mapping[str, int]) -> None:
        po2_pal = float(y[index["O2_trop"]]) / TABLE3_MOLES["O2_trop"]
        target = threshold_access_target(po2_pal, self.half_pal, self.hill)
        current = float(y[index[ACCESS_SPECIES]])
        dydt[index[ACCESS_SPECIES]] += (target - current) / max(self.tau_yr, 1.0e-12)
