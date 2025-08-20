from typing import Optional

import cantera as ct


def perturbed_gas(mech: str, rxn_no: Optional[int], perturbation_fraction: Optional[float]) -> ct.Solution:
    gas = ct.Solution(mech)
    if None not in (rxn_no, perturbation_fraction):
        gas.set_multiplier(1 + perturbation_fraction, rxn_no)

    return gas
