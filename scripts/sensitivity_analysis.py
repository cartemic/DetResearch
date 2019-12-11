from multiprocessing import Pool, Lock
from funcs.sensitivity import init, perform_study
from tqdm import tqdm
import funcs.database as db
import cantera as ct
# from itertools import permutations
# import numpy as np
# import cantera as ct


# fuels = {'CH4', 'C3H8'}
# oxidizers = {'N2O'}
# equivs = {0.4, 0.7, 1.0}
# init_pressures = {101325.}
# init_temps = {300}
# dilution_numbers = np.array([1, 5.5, 10], dtype=float)
# dilution = {
#     'None': [0],
#     'CO2': dilution_numbers * 1e-2,
#     'NO': dilution_numbers * 1e-4
# }

def study_with_progress_bar(
        progress_bar,
        chanism,
        itial_temp,
        itial_press,
        uivalence,
        el,
        idizer,
        luent,
        luent_mol_frac,
        ert,
        rturbation_fraction,
        iii
):
    perform_study(
        chanism,
        itial_temp,
        itial_press,
        uivalence,
        el,
        idizer,
        luent,
        luent_mol_frac,
        ert,
        rturbation_fraction,
        iii
    )
    progress_bar.update(1)


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    # inert = 'AR'
    inert = 'None'
    # mechanism = 'Mevel2017.cti'
    mechanism = 'gri30.cti'
    inert_species = [inert]
    initial_temp = 300
    initial_press = 101325
    equivalence = 1
    fuel = 'CH4'
    oxidizer = 'N2O'
    # diluent = 'AR'
    diluent = 'None'
    diluent_mol_frac = 0
    perturbation_fraction = 1e-4

    t = db.Table(
        'sensitivity.sqlite',
        'data'
    )
    exist_check = t.fetch_test_rows(
        mechanism=mechanism,
        initial_temp=initial_temp,
        initial_press=initial_press,
        fuel=fuel,
        oxidizer=oxidizer,
        equivalence=equivalence,
        diluent=diluent,
        diluent_mol_frac=diluent_mol_frac,
        inert=inert
    )['rxn_table_id']
    # if len(exist_check) > 0:
    #     t.delete_test(exist_check[0])

    reactions = []
    # noinspection PyCallByClass,PyArgumentList
    for rxn in ct.Reaction.listFromFile(mechanism):
        if not any([
            s in list(rxn.reactants) + list(rxn.products)
            for s in inert_species
        ]):
            reactions.append(rxn)

    pbar = tqdm(total=len(reactions))

    lock = Lock()
    # p = Pool(initializer=init, initargs=(lock,))
    # p.starmap(
    #     study_with_progress_bar,
    #     [
    #         [
    #             pbar,
    #             mechanism,
    #             initial_temp,
    #             initial_press,
    #             equivalence,
    #             fuel,
    #             oxidizer,
    #             diluent,
    #             diluent_mol_frac,
    #             inert,
    #             perturbation_fraction,
    #             i,
    #         ] for i in range(len(reactions))
    #     ]
    # )
    for i in range(len(reactions)):
        perform_study(
            mechanism,
            initial_temp,
            initial_press,
            equivalence,
            fuel,
            oxidizer,
            diluent,
            diluent_mol_frac,
            inert,
            perturbation_fraction,
            i,
            lock
        )
        pbar.update(1)
