# -*- coding: utf-8 -*-
"""
PURPOSE:
    Tools for ZND detonation simulation and chemical sensitivity analysis using
    cantera

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import cantera as ct


def get_species_reactions(gas_object, inert_species):
    """
    Finds all reactions in a cantera gas object containing the desired species

    Parameters
    ----------
    gas_object : ct.Solution
        Cantera gas object containing reaction equations

    inert_species : str or iterable
        String or iterable of the specie or species to find the reactions for

    Returns
    -------
    reaction_keys : list
        List of all equations containing the input species
    """
    if isinstance(inert_species, str):
        # we are gonna treat this like a list
        inert_species = [inert_species.upper()]
    else:
        # make sure everything is upper case to make cantera happy
        inert_species = [s.upper() for s in inert_species]

    reaction_keys = []
    equations = gas_object.reaction_equations()

    for species in inert_species:
        check_length = len(species) + 1
        reaction_keys += [
            eqn for eqn in equations if ' %s ' % species in eqn
                or eqn[:check_length] == '%s ' % species
                or eqn[-check_length:] == ' %s' % species
        ]

    return reaction_keys
