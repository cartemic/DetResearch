from multiprocessing import Pool, Lock
from funcs.sensitivity import init, perform_study
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

inert = 'AR'
# inert = 'None'
mechanism = 'Mevel2017.cti'
inert_species = [inert]
initial_temp = 300
initial_press = 101325.2
equivalence = 1
fuel = 'H2'
oxidizer = 'O2'
diluent = 'AR'
diluent_mol_frac = 0
perturbation_fraction = 1e-2

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
if len(exist_check) > 0:
    t.delete_test(exist_check[0])

reactions = []
# noinspection PyCallByClass,PyArgumentList
for rxn in ct.Reaction.listFromFile(mechanism):
    if not any([
        s in list(rxn.reactants) + list(rxn.products)
        for s in inert_species
    ]):
        reactions.append(rxn)

lock = Lock()
p = Pool(initializer=init, initargs=(lock,))
num_tests = 5
p.starmap(
    perform_study,
    [
        [
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
        ] for i in range(len(reactions))
    ]
)
