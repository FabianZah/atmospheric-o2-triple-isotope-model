"""Generic mass-action reaction machinery for the Young model rebuild."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Reaction:
    """A stoichiometric reaction with a mass-action rate law.

    Species keys must correspond to indices in a model state vector. First-order
    rates use one reactant. Second-order rates use two reactants. Constant fluxes
    can be represented with no reactants and a rate constant in mol/year.
    """

    key: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_constant: float
    units: str
    note: str = ""

    def rate(self, y: np.ndarray, index: Mapping[str, int]) -> float:
        r = self.rate_constant
        for species, power in self.reactants.items():
            r *= y[index[species]] ** power
        return float(r)

    def apply(self, dydt: np.ndarray, rate: float, index: Mapping[str, int]) -> None:
        for species, coeff in self.reactants.items():
            dydt[index[species]] -= coeff * rate
        for species, coeff in self.products.items():
            dydt[index[species]] += coeff * rate


def derivative(y: np.ndarray, reactions: list[Reaction], species_order: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(species_order)}
    dydt = np.zeros_like(y, dtype=float)
    for reaction in reactions:
        if hasattr(reaction, "apply_state"):
            reaction.apply_state(dydt, y, index)
        else:
            reaction.apply(dydt, reaction.rate(y, index), index)
    return dydt
