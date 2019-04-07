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
import os


def _find_specie_in_str(specie, equation_str):
    """
    does what it says on the label

    Parameters
    ----------
    specie : str
        Desired chemical specie
    equation_str : str
        Equation string to search

    Returns
    -------
    found_specie : bool
        True if species is found, False if not

    """
    check_length = len(specie) + 1
    found_specie = ' %s ' % specie in equation_str \
        or equation_str[:check_length] == '%s ' % specie \
        or equation_str[-check_length:] == ' %s' % specie

    return found_specie


def _check_input_filetype(fname, extension):
    if fname[-len(extension):] != extension:
        raise NameError('{0:s} is not a {1:s} file'.format(fname, extension))


def _check_output_filetype(fname, extension):
    if fname[-len(extension):] != extension:
        fname += extension
    return fname


def get_species_reactions(gas_object, species):
    """
    Finds all reactions in a cantera gas object containing the desired species

    Parameters
    ----------
    gas_object : ct.Solution
        Cantera gas object containing reaction equations

    species : str or iterable
        String or iterable of the specie(s) to find the reactions for

    Returns
    -------
    reaction_keys : list
        List of all equations containing the input species
    """
    if isinstance(species, str):
        # we are gonna treat this like a list
        species = [species.upper()]
    else:
        # make sure everything is upper case to make cantera happy
        species = [s.upper() for s in species]

    reaction_keys = []
    equations = gas_object.reaction_equations()

    for s in species:
        reaction_keys += [
            eqn for eqn in equations if _find_specie_in_str(s, eqn)
        ]

    return reaction_keys


def make_inert_cti(mech, species, out_name):
    """
    Parses a .cti mechanism file to find the location of desired chemical
    species, turns them inert, and saves the inert mechanism with the desired
    file name.
    Parameters
    ----------
    mech : str
        name of mechanism to search, e.g. `gri30.cti`
    species : str or iterable
        specie(s) to search for
    out_name : str
        name of output file, e.g. `gri30_inert_co.cti

    Returns
    -------

    """
    file_type = '.cti'
    _check_input_filetype(mech, file_type)
    out_name = _check_output_filetype(out_name, file_type)

    if isinstance(species, str):
        # we are gonna treat this like a list
        species = [species.upper()]
    else:
        # make sure everything is upper case to make cantera happy
        species = [s.upper() for s in species]

    mech_loc = os.path.join(
        os.path.split(ct.__file__)[0],
        'data',
        mech
    )
    if not os.path.exists(mech_loc):
        raise FileNotFoundError(
            '%s not found in cantera data directory' % mech
        )

    with open(mech_loc, 'r') as f:
        data_in = f.read()

    data_header_length = 97
    start_loc = data_in.find('#  Reaction')
    start_loc = data_in[start_loc + data_header_length:].find('# Reaction') + \
        start_loc + data_header_length + 1
    header = data_in[:start_loc]
    to_scan = data_in[start_loc:]
    scanned = to_scan.split('#')[1:]

    for loc, rxn in enumerate(scanned):
        rxn = '#' + rxn
        if any([_find_specie_in_str(s, rxn) for s in species]):
            if 'falloff' in rxn:
                # todo: figure out falloff reactions
                rxn_data = []
                pass
            else:
                # this search method will work for three body and regular
                # reactions. Three body needs its own sub-case to handle the
                # efficiency term.
                rxn_data_loc = [
                    rxn.find('[')+1,
                    rxn.find(']')
                ]
                rxn_start = rxn[:rxn_data_loc[0]]
                rxn_end = rxn[rxn_data_loc[1]:]
                rxn_data = [
                    item for item
                    in rxn[rxn_data_loc[0]: rxn_data_loc[1]].split(',')
                ]
                # reaction data is in the form [A, b, E]
                # todo: zero all three?
                rxn_data[0] = ' 0'
                if 'three_body' in rxn:
                    # todo: figure out three body reactions
                    pass

            if loc == 1:
                print(rxn)
                print()
                new_rxn = '{0:s}{1:s},{2:s},{3:s}{4:s}'.format(
                    rxn_start,
                    *rxn_data,
                    rxn_end
                )
                print(new_rxn)


def make_inert_xml(mech, species, out_name):
    """
    Parses an .xml mechanism file to find the location of desired chemical
    species so they can be turned off
    Parameters
    ----------
    mech : str
        name of mechanism to search, e.g. `gri30.xml`
    species : str or iterable
        specie(s) to search for
    out_name : str
        name of output file, e.g. `gri30_inert_co.xml

    Returns
    -------

    """
    # todo: need soup!
    file_type = '.xml'
    _check_input_filetype(mech, file_type)
    out_name = _check_output_filetype(out_name, file_type)
    pass


if __name__ == '__main__':
    find_in_cti('air.cti', 'N2')
