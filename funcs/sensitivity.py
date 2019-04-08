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
        or equation_str[-check_length:] == ' %s' % specie \

    return found_specie


def _check_input_filetype(fname, extension):
    if fname[-len(extension):] != extension:
        raise NameError('{0:s} is not a {1:s} file'.format(fname, extension))


def _check_output_filetype(fname, extension):
    if fname[-len(extension):] != extension:
        fname += extension
    return fname


def _enforce_species_list(species):
    if isinstance(species, str):
        species = [species.upper()]
    elif hasattr(species, '__iter__'):
        species = [s.upper() for s in species]
    else:
        raise TypeError('Bad species type: %s' % type(species))

    return species


def _get_file_locations(mech, out_file):
    mech_dir = os.path.join(
        os.path.split(ct.__file__)[0],
        'data'
    )
    in_loc = os.path.join(
        mech_dir,
        mech
    )
    out_loc = os.path.join(
        mech_dir,
        out_file
    )

    if not os.path.exists(in_loc):
        raise FileNotFoundError(
            '%s not found in cantera data directory' % mech
        )

    return[mech_dir, in_loc, out_loc]


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


def make_inert_cti(mech, species, out_file):
    """
    Parses a .cti mechanism file to find the location of desired chemical
    species, turns them inert, and saves the inert mechanism with the desired
    file name.
    Parameters
    ----------
    mech : str
        name of mechanism to search, e.g. `gri30.cti`
    species : str or iterable[str]
        specie(s) to search for
    out_file : str
        name of output file, e.g. `gri30_inert_co.cti

    Returns
    -------

    """
    file_type = '.cti'
    _check_input_filetype(mech, file_type)
    out_file = _check_output_filetype(out_file, file_type)
    species = _enforce_species_list(species)
    mech_dir, in_loc, out_loc = _get_file_locations(mech, out_file)

    with open(in_loc, 'r') as f:
        data_in = f.read()

    data_header_length = 97
    start_loc = data_in.find('#  Reaction')
    start_loc = data_in[start_loc + data_header_length:].find('# Reaction') + \
        start_loc + data_header_length + 1
    header = data_in[:start_loc]
    to_scan = data_in[start_loc:]
    scanned = to_scan.split('#')[1:]
    new_rxns = ''

    for loc, rxn in enumerate(scanned):
        # if any of the desired species are in the current reaction, delete
        # it from the mechanism
        rxn = '#' + rxn
        eqn_start = rxn.find('\"')+1
        eqn_end = rxn[eqn_start:].find('\"') + eqn_start
        eqn = rxn[eqn_start:eqn_end]
        if not any([_find_specie_in_str(s, eqn) for s in species]):
            new_rxns += rxn

    new_mech = header + '\n\n' + new_rxns
    with open(out_loc, 'w') as f:
        f.writelines(new_mech)
        f.flush()


def make_inert_xml(mech, species, out_file):
    """
    Parses an .xml mechanism file to find the location of desired chemical
    species so they can be turned off
    Parameters
    ----------
    mech : str
        name of mechanism to search, e.g. `gri30.xml`
    species : str or iterable
        specie(s) to search for
    out_file : str
        name of output file, e.g. `gri30_inert_co.xml

    Returns
    -------

    """
    file_type = '.xml'
    _check_input_filetype(mech, file_type)
    out_file = _check_output_filetype(out_file, file_type)
    species = _enforce_species_list(species)
    mech_dir, in_loc, out_loc = _get_file_locations(mech, out_file)

    # todo: need soup!
    pass


if __name__ == '__main__':
    # build a test mechanism with inert oxygen
    base_mech = 'gri30.cti'
    test_mech = 'test_mech.cti'
    inert_species = ['O2']
    init_temp = 1000
    init_press = ct.one_atm
    mechs = [base_mech, test_mech]
    _, _, cti_out = _get_file_locations(base_mech, test_mech)

    make_inert_cti(base_mech, inert_species, test_mech)
    gases = [ct.Solution(m) for m in mechs]
    for idx, g in enumerate(gases):
        g.set_equivalence_ratio(1, 'H2', 'O2')
        init_mf = g.mole_fraction_dict()['O2']
        g.TP = init_temp, init_press
        r = ct.Reactor(g)
        n = ct.ReactorNet([r])
        n.advance_to_steady_state()
        print(mechs[idx])
        print('-'*len(mechs[idx]))
        print('# reactions:     {:1.0f}'.format(len(g.reaction_equations())))
        print('final temp:      {:1.0f} K'.format(r.thermo.TP[0]))
        print('final pressure:  {:1.0f} atm'.format(r.thermo.TP[1]/ct.one_atm))
        print('Y_02 init/final: {:0.3f} / {:0.3f}'.format(
            init_mf,
            r.thermo.mole_fraction_dict()['O2']
        ))
        print()

    # clean up
    os.remove(cti_out)
