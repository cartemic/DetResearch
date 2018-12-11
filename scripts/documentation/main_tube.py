# -*- coding: utf-8 -*-
"""
PURPOSE:
    This script performs the following tasks as part of the tube documentation:
       - things involving the main section of the det tube

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""
import beaverdet as bd
import pint
import numpy as np
import cantera as ct
from plotting import figure
from matplotlib import pyplot as plt

# initialize unit registry
ureg = pint.UnitRegistry()
quant = ureg.Quantity

# set desired tube quantities
material = '316L'
schedule = '80'
nominal_size = '6'
welded = False
safety_factor = 4
initial_temps = np.linspace(0.1, 200, 10)

# set mixture
mech = 'gri30.cti'
fuel = 'CH4'
oxidizer = 'N2O'
phi = 1
gas = ct.Solution(mech)
gas.set_equivalence_ratio(
    phi,
    fuel,
    oxidizer
)
species = gas.mole_fraction_dict()

pressures = []

for temp in initial_temps:
    my_tube = bd.tube.Tube(
        material,
        schedule,
        nominal_size,
        welded,
        safety_factor
    )

    my_tube.calculate_max_stress(
        quant(temp, 'degC')
    )

    my_tube.calculate_max_pressure()

    pressures.append(
        my_tube.calculate_initial_pressure(
            species,
            mech
        ).to('atm').magnitude
    )

    flange_class = my_tube.lookup_flange_class()

[fig, axes] = figure(
    'Initial Temperature ($^\circ$C)',
    'Initial Pressure (atm)',
    'Safe Initial Temperature and Pressure',
    '#333333'
)

axes.plot(
    initial_temps,
    pressures
)

fig.savefig('initial_press-temp.pdf ', format='pdf')

plt.show()
