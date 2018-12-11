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
# from plotting import figure
import json
import warnings

# initialize unit registry
ureg = pint.UnitRegistry()
quant = ureg.Quantity

mech = 'gri30.cti'


def quantity_remover(my_thing):
    """
    removes pint quantities to make json output happy

    Parameters
    ----------
    my_thing

    Returns
    -------

    """

    if hasattr(my_thing, 'magnitude'):
        return 'QUANTITY', my_thing.magnitude, my_thing.units.format_babel()

    elif isinstance(my_thing, dict):
        newdict = dict()
        for key, item in my_thing.items():
                newdict[key] = quantity_remover(item)
        return newdict

    elif hasattr(my_thing, '__iter__') and not isinstance(my_thing, str):
        my_type = type(my_thing)
        return my_type([quantity_remover(item) for item in my_thing])

    else:
        return my_thing


def quantity_putter_backer(my_thing):
    """
    undoes the work of quantity_remover

    Parameters
    ----------
    my_thing

    Returns
    -------

    """

    if isinstance(my_thing, dict):
        newdict = dict()
        for key, item in my_thing.items():
                newdict[key] = quantity_putter_backer(item)
        return newdict

    elif hasattr(my_thing, '__iter__') and not isinstance(my_thing, str):
        my_type = type(my_thing)

        if len(my_thing) == 3 and my_thing[0] == 'QUANTITY':
            return quant(my_thing[1], my_thing[2])

        else:
            return my_type([quantity_putter_backer(item) for item in my_thing])

    else:
        return my_thing


def run_studies(
        fuels,
        oxidizer,
        initial_temps,
        equivalence_ratios,
        pipe_sizes,
        pipe_schedules,
        studies=('temp', 'equiv', 'size', 'sched')
):
    # make sure I didn't put in any wrong studies
    good_studies = {'temp', 'equiv', 'size', 'sched'}
    studies = set(studies)
    for bad_thing in studies.difference(good_studies):
        warnings.warn(
            '{0} is not a valid study, and will be skipped'.format(bad_thing)
        )
    if len(studies.difference(good_studies)) > 0:
        print('\nValid studies: {0}'.format(good_studies))
    studies = studies.intersection(good_studies)

    if 'temp' in studies:
        # define terms
        material = '316L'
        schedule = '80'
        nominal_size = '6'
        welded = False
        safety_factor = 4

        pressures = {fuel: [] for fuel in fuels}

        # [fig, axes] = figure(
        #     'Initial Temperature ($^\circ$C)',
        #     'Initial Pressure (atm)',
        #     'Safe Initial Temperature and Pressure',
        #     '#333333'
        # )

        for current_fuel in fuels:
            equivalence = 1
            gas = ct.Solution(mech)
            gas.set_equivalence_ratio(
                equivalence,
                current_fuel,
                oxidizer
            )
            species = gas.mole_fraction_dict()

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

                pressures[current_fuel].append(
                    my_tube.calculate_initial_pressure(
                        species,
                        mech
                    ).to('atm')
                )

        #     axes.plot(
        #         initial_temps,
        #         pressures,
        #         label=current_fuel
        #     )
        #
        # axes.legend(edgecolor='none')
        # fig.savefig('temperature.pdf ', format='pdf')
        pressures['sizes'] = pipe_sizes

        data_out = quantity_remover(pressures)
        with open('pipe_size.json', 'w') as file:
            json.dump(data_out, file)

    if 'equiv' in studies:
        # define terms
        material = '316L'
        schedule = '80'
        nominal_size = '6'
        welded = False
        safety_factor = 4

        pressures = {fuel: [] for fuel in fuels}

        # [fig, axes] = figure(
        #     'Equivalence Ratio ($\Phi$)',
        #     'Initial Pressure (atm)',
        #     'Safe Initial Pressure',
        #     '#333333'
        # )

        for current_fuel in fuels:
            temp = quant(20, 'degC')

            for phi in equivalence_ratios:
                # build gas object
                gas = ct.Solution(mech)
                gas.set_equivalence_ratio(
                    phi,
                    current_fuel,
                    oxidizer
                )
                species = gas.mole_fraction_dict()

                # build tube
                my_tube = bd.tube.Tube(
                    material,
                    schedule,
                    nominal_size,
                    welded,
                    safety_factor
                )

                my_tube.calculate_max_stress(
                    temp
                )

                my_tube.calculate_max_pressure()

                pressures[current_fuel].append(
                    my_tube.calculate_initial_pressure(
                        species,
                        mech
                    ).to('atm')
                )

        #     axes.plot(
        #         equivalence_ratios,
        #         pressures,
        #         label=current_fuel
        #     )
        #
        # axes.legend(edgecolor='none')
        # fig.savefig('equivalence.pdf ', format='pdf')
        pressures['sizes'] = pipe_sizes

        data_out = quantity_remover(pressures)
        with open('pipe_size.json', 'w') as file:
            json.dump(data_out, file)

    if 'size' in studies:
        # define terms
        equivalence = 1
        material = '316L'
        schedule = '80'
        welded = False
        safety_factor = 4

        pressures = {fuel: [] for fuel in fuels}

        # [fig, axes] = figure(
        #     'Nominal Pipe Size (cm)',
        #     'Initial Pressure (atm)',
        #     'Safe Initial Pressure',
        #     '#333333'
        # )

        for current_fuel in fuels:
            temp = quant(20, 'degC')

            for size in pipe_sizes:
                # build gas object
                gas = ct.Solution(mech)
                gas.set_equivalence_ratio(
                    equivalence,
                    current_fuel,
                    oxidizer
                )
                species = gas.mole_fraction_dict()

                # build tube
                my_tube = bd.tube.Tube(
                    material,
                    schedule,
                    size,
                    welded,
                    safety_factor
                )

                my_tube.calculate_max_stress(
                    temp
                )

                my_tube.calculate_max_pressure()

                pressures[current_fuel].append(
                    my_tube.calculate_initial_pressure(
                        species,
                        mech
                    ).to('atm')
                )

            # sizes = [quant(float(size), 'in').to('cm') for size in pipe_sizes]
            # axes.plot(
            #     sizes,
            #     pressures,
            #     label=current_fuel
            # )

        # axes.legend(edgecolor='none')
        # fig.savefig('size.pdf ', format='pdf')
        pressures['sizes'] = pipe_sizes

        data_out = quantity_remover(pressures)
        with open('pipe_size.json', 'w') as file:
            json.dump(data_out, file)

    if 'sched' in studies:
        # define terms
        equivalence = 1
        material = '316L'
        nominal_size = '6'
        welded = False
        safety_factor = 4

        pressures = {fuel: [] for fuel in fuels}

        # [fig, axes] = figure(
        #     'Pipe Schedule',
        #     'Initial Pressure (atm)',
        #     'Safe Initial Pressure',
        #     '#333333'
        # )

        for current_fuel in fuels:
            temp = quant(20, 'degC')

            for schedule in pipe_schedules:
                # build gas object
                gas = ct.Solution(mech)
                gas.set_equivalence_ratio(
                    equivalence,
                    current_fuel,
                    oxidizer
                )
                species = gas.mole_fraction_dict()

                # build tube
                my_tube = bd.tube.Tube(
                    material,
                    schedule,
                    nominal_size,
                    welded,
                    safety_factor
                )

                my_tube.calculate_max_stress(
                    temp
                )

                my_tube.calculate_max_pressure()

                pressures[current_fuel].append(
                    my_tube.calculate_initial_pressure(
                        species,
                        mech
                    ).to('atm')
                )

        #     axes.plot(
        #         pipe_schedules,
        #         pressures,
        #         label=current_fuel
        #     )
        #
        # axes.legend(edgecolor='none')
        # fig.savefig('schedule.pdf ', format='pdf')
        pressures['sizes'] = pipe_sizes

        data_out = quantity_remover(pressures)
        with open('pipe_size.json', 'w') as file:
            json.dump(data_out, file)


if __name__ == '__main__':
    # set varied properties
    fuel_list = ['CH4', 'C3H8']
    temps = np.linspace(0.1, 200, 20)
    equivs = np.linspace(0.4, 1.0, 20)
    size_list = ['1', '2', '3', '4', '5', '6']
    sched_list = ['5', '10', '40', '80', '160', 'XXH']

    # set constant properties
    ox = 'N2O'

    # pick studies to run
    # desired_studies = ['size']
    run_studies(
        fuel_list,
        ox,
        temps,
        equivs,
        size_list,
        sched_list  #,
        # desired_studies
    )
