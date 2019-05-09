from funcs import cell_size, database
from multiprocessing import Pool, Lock
from itertools import permutations
import numpy as np
import cantera as ct


def check_stored_base_cj_speed():
    pass


def check_stored_pert_cj_speed():
    pass


def check_stored_base_reactions():
    pass


def calc_cell_size():
    pass


def perform_study(
        mech,
        init_temp,
        init_press,
        equivalence,
        fuel,
        oxidizer,
):
    gas = ct.Solution(mech)
    gas.TP = init_temp, init_press
    gas.set_equivalence_ratio(
        equivalence,
        fuel,
        oxidizer
    )
    stored = check_stored_base_cj_speed()
    if not stored:
        # calculate cj speed
        # lock DB
        # store cj speed
        # release lock
        pass
    stored = check_stored_base_reactions()
    if not stored:
        # lock DB
        # store reactions
        # release lock
        pass
    # calculate base cell size
    # lock DB
    # store cell size data
    # release lock
    # perturb gas (set multiplier)
    stored = check_stored_pert_cj_speed()
    if not stored:
        # calculate cj speed
        # lock DB
        # store cj speed
        # release lock
        pass
    # calculate perturbed cell size
    # lock DB
    # store cell size and reaction data
    # release lock


if __name__ == '__main__':
    fuels = {'CH4', 'C3H8'}
    oxidizers = {'N2O'}
    equivs = {0.4, 0.7, 1.0}
    init_pressures = {101325.}
    init_temps = {300}
    dilution_numbers = np.array([1, 5.5, 10], dtype=float)
    dilution = {
        'None': [0],
        'CO2': dilution_numbers * 1e-2,
        'NO': dilution_numbers * 1e-4
    }
