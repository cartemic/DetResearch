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


def _enforce_species_list(species):
    if isinstance(species, str):
        species = [species.upper()]
    elif hasattr(species, '__iter__'):
        species = [s.upper() for s in species]
    else:
        raise TypeError('Bad species type: %s' % type(species))

    return species


# noinspection PyArgumentList
def solution_with_inerts(
        mech,
        inert_species
):
    inert_species = _enforce_species_list(inert_species)
    species = ct.Species.listFromFile(mech)
    reactions = []
    for rxn in ct.Reaction.listFromFile(mech):
        if not any([
            s in list(rxn.reactants) + list(rxn.products)
            for s in inert_species
        ]):
            reactions.append(rxn)

    return ct.Solution(
        thermo='IdealGas',
        species=species,
        reactions=reactions
    )

if __name__ == '__main__':
    # # build a test mechanism with inert oxygen
    # base_mech = 'gri30'
    # test_mech = 'test_mech'
    # file_types = ['.cti', '.xml']
    # funcs = [make_inert_cti, make_inert_xml]
    # inert_species = ['O2']
    # init_temp = 1000
    # init_press = ct.one_atm
    # for ftype, func in zip(file_types, funcs):
    #     mechs = [base_mech + ftype, test_mech + ftype]
    #     _, _, file_out = _get_file_locations(mechs[0], mechs[1])
    #
    #     func(mechs[0], inert_species, mechs[1])
    #     gases = [ct.Solution(m) for m in mechs]
    #     for idx, g in enumerate(gases):
    #         g.set_equivalence_ratio(1, 'H2', 'O2')
    #         init_mf = g.mole_fraction_dict()['O2']
    #         g.TP = init_temp, init_press
    #         r = ct.Reactor(g)
    #         n = ct.ReactorNet([r])
    #         n.advance_to_steady_state()
    #         print(mechs[idx])
    #         print('-'*len(mechs[idx]))
    #         print('# reactions:     {:1.0f}'.format(
    #             len(g.reaction_equations()))
    #         )
    #         print('final temp:      {:1.0f} K'.format(
    #             r.thermo.TP[0])
    #         )
    #         print('final pressure:  {:1.0f} atm'.format(
    #             r.thermo.TP[1]/ct.one_atm)
    #         )
    #         print('Y_02 init/final: {:0.3f} / {:0.3f}'.format(
    #             init_mf,
    #             r.thermo.mole_fraction_dict()['O2']
    #         ))
    #         print()
    #
    #     # clean up
    #     os.remove(file_out)

    mechanism = 'gri30.cti'

    rxns = ct.Reaction.listFromFile(mechanism)
    print(len([rxn for rxn in rxns if 'O' not in list(rxn.reactants) + list(rxn.products) and 'O2' not in list(rxn.reactants) + list(rxn.products)]))
    print(len(ct.Species.listFromFile(mechanism)))

    test_gas = solution_with_inerts('gri30.cti', ['o', 'o2'])
    print(test_gas.n_reactions, test_gas.n_species)
