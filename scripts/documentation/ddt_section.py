# -*- coding: utf-8 -*-
"""
PURPOSE:
    This script performs the following tasks as part of the tube documentation:
       - calculates diameter for a Shchelkin spiral to achieve a desired
       blockage ratio (too big for practical manufacture)
       - calculates the diameter required for OSU-style blockage apparatus
       - rounds OSU blockage diameter to the nearest 1/4 inch (for washer
        availability)
       - calculates the resulting blockage ratio
       - uses this blockage ratio to calculate run-up distances for methane- and
       propane-nitrous mixtures over a range of equivalence ratios
       - saves a plot of run-up distance vs. blockage ratio as .pdf

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
from matplotlib import pyplot as plt
from .plotting import figure

# initialize unit registry
ureg = pint.UnitRegistry()
quant = ureg.Quantity

# define initial state
init_temp = quant(20, 'degC')
init_press = quant(1, 'atm')

# define tube and mixture quantities
tube_diam = quant(5.76, 'in').to('cm')      # inner diameter
desired_br = 0.45                           # blockage ratio
mech = 'gri30.cti'                          # mechanism for Cantera calculations
gas = ct.Solution(mech)
equivs = np.linspace(
    0.4,
    1.0,
    20
)                                           # desired equivalence ratios
fuels = ['CH4', 'C3H8']                     # fuels to test with
fuel_names = {
    'CH4': 'Methane',
    'C3H8': 'Propane'
}
oxidizer = 'N2O'

# find spiral diameter resulting in desired blockage ratio
spiral_diam = bd.tube.DDT.calculate_spiral_diameter(
    tube_diam,
    desired_br
)
print(
    'Spiral diameter: {0:.2f} ({1:.2f})'.format(
        spiral_diam.to('cm'),
        spiral_diam.to('in')
    )
)

# calculate actual blockage ratio, since a 3/4 inch spiral will be hard to make
num_struts = 4                              # number of blockage struts
washer_diam = tube_diam * np.sqrt(desired_br / num_struts)
print(
    'Calculated diameter: {0:.2f} ({1:.2f})'.format(
        washer_diam.to('cm'),
        washer_diam.to('in')
    )
)

# round washer diameter to the nearest 1/4 inch
washer_diam = np.round(washer_diam.to('in') * 4) / 4.
print(
    'Washer diameter: {0:.2f} ({1:.2f})'.format(
        washer_diam.to('cm'),
        washer_diam.to('in')
    )
)

# calculate new blockage ratio
actual_br = num_struts * (washer_diam.to('m') / tube_diam.to('m')).magnitude**2
print(
    'Actual blockage ratio: {:.2f}%'.format(
        actual_br * 100
    )
)

# calculate run-up distance as a function of equivalence for each mixture
axis_color = '#333333'
fig, axes = figure(
    'Equivalence Ratio',
    'Run-up Distance (m)',
    'DDT Section Minimum Length',
    axis_color
)
for fuel in fuels:
    print(fuel)
    runup_lengths = []

    for phi in equivs:
        gas.set_equivalence_ratio(
            phi,
            fuel,
            oxidizer
        )
        species = gas.mole_fraction_dict()

        runup = bd.tube.DDT.calculate_run_up(
            actual_br,
            tube_diam,
            init_temp,
            init_press,
            species,
            mech,
            ureg
        )
        runup_lengths.append(runup)
        print(
            '    {0}: {1:.2f} ({2:.2f})'.format(
                phi,
                runup,
                runup.to('ft')
            )
        )

    axes.plot(
        equivs,
        [length.to('m').magnitude for length in runup_lengths],
        label=fuel_names[fuel]
    )

axes.legend(
    edgecolor='none'
)

fig.savefig('runup_calcs.pdf ', format='pdf')
plt.show()
