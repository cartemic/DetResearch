from funcs import cell_size
from funcs import database as db
from multiprocessing import Pool, Lock
from itertools import permutations
import numpy as np
import cantera as ct
import sdtoolbox as sd


def check_stored_base_cj_speed(
        table,
        mechanism,
        initial_temp,
        initial_press,
        fuel,
        oxidizer,
        equivalence,
        diluent,
        diluent_mol_frac,
        inert,
):
    current_data = table.fetch_test_rows(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
            inert=inert
    )
    return len(current_data['cj_speed']) == 1


def perform_study(
        mech,
        init_temp,
        init_press,
        equivalence,
        fuel,
        oxidizer,
        diluent,
        diluent_mol_frac,
        inert,
        # db_lock,
        perturbation_fraction,
        perturbed_reaction_no,
):
    gas = ct.Solution(mech)
    gas.TP = init_temp, init_press
    gas.set_equivalence_ratio(
        equivalence,
        fuel,
        oxidizer
    )
    db_name = 'sensitivity.sqlite'
    table_name = 'data'
    current_table = db.Table(
        database=db_name,
        table_name=table_name

    )

    # check for stored cj speed
    stored = check_stored_base_cj_speed(
        table=current_table,
        mechanism=mech,
        initial_temp=init_temp,
        initial_press=init_press,
        fuel=fuel,
        oxidizer=oxidizer,
        equivalence=equivalence,
        diluent=diluent,
        diluent_mol_frac=diluent_mol_frac,
        inert=inert
    )
    if not stored:
        # calculate and store cj speed
        base_cj_speed = sd.postshock.CJspeed(
            P1=init_press,
            T1=init_temp,
            q=gas.mole_fraction_dict(),
            mech=mech
        )
        with db_lock:
            rxn_table_id = current_table.store_test_row(
                mechanism=mech,
                initial_temp=init_temp,
                initial_press=init_press,
                fuel=fuel,
                oxidizer=oxidizer,
                equivalence=equivalence,
                diluent=diluent,
                diluent_mol_frac=diluent_mol_frac,
                inert=inert,
                cj_speed=base_cj_speed,
                ind_len_west=0,
                ind_len_gav=0,
                ind_len_ng=0,
                cell_size_west=0,
                cell_size_gav=0,
                cell_size_ng=0,
                overwrite_existing=False
            )
    else:
        current_data = current_table.fetch_test_rows(
            mechanism=mech,
            initial_temp=init_temp,
            initial_press=init_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
            inert=inert
        )
        [rxn_table_id] = current_data['rxn_table_id']
        [base_cj_speed] = current_data['cj_speed']
        del current_data

    # check for stored base reaction data
    stored = current_table.check_for_stored_base_data(rxn_table_id)
    if not stored:
        with db_lock:
            current_table.store_base_rxn_table(
                rxn_table_id=rxn_table_id,
                gas=gas
            )
    del stored

    # calculate base cell size
    CellSize = cell_size.CellSize()
    base_cell_calcs = CellSize(
        base_mechanism=mech,
        cj_speed=base_cj_speed,
        initial_temp=init_temp,
        initial_press=init_press,
        fuel=fuel,
        oxidizer=oxidizer,
        equivalence=equivalence,
        diluent=diluent,
        diluent_mol_frac=diluent_mol_frac,
        inert=inert,
        database=db_name,
        table_name=table_name
    )
    base_ind_len = CellSize.induction_length

    with db_lock:
        current_table.store_test_row(
            mechanism=mech,
            initial_temp=init_temp,
            initial_press=init_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
            inert=inert,
            cj_speed=base_cj_speed,
            ind_len_west=base_ind_len['Westbrook'],
            ind_len_gav=base_ind_len['Gavrikov'],
            ind_len_ng=base_ind_len['Ng'],
            cell_size_west=base_cell_calcs['Westbrook'],
            cell_size_gav=base_cell_calcs['Gavrikov'],
            cell_size_ng=base_cell_calcs['Ng'],
            overwrite_existing=True
        )

    # perturb gas (set multiplier)
    if perturbed_reaction_no != -1:
        # alter ct.Solution within sdtoolbox so that it returns perturbed
        # reaction results
        def my_solution(mechanism):
            original_gas = cell_size.OriginalSolution(mechanism)
            original_gas.set_multiplier(
                1 + perturbation_fraction,
                perturbed_reaction_no
            )
            return original_gas

        sd.postshock.ct.Solution = my_solution

    # check for existing perturbed cj speed
    current_data = current_table.fetch_pert_table(
        rxn_table_id=rxn_table_id,
        rxn_no=perturbed_reaction_no,
    )
    if len(current_data['cj_speed']) == 0:
        # calculate cj speed
        pert_cj_speed = sd.postshock.CJspeed(
            P1=init_press,
            T1=init_temp,
            q=gas.mole_fraction_dict(),
            mech=mech
        )
    else:
        [pert_cj_speed] = current_data['cj_speed']
    del current_data

    # calculate perturbed cell size
    pert_cell_calcs = CellSize(
        base_mechanism=mech,
        cj_speed=pert_cj_speed,
        initial_temp=init_temp,
        initial_press=init_press,
        fuel=fuel,
        oxidizer=oxidizer,
        equivalence=equivalence,
        diluent=diluent,
        diluent_mol_frac=diluent_mol_frac,
        inert=inert,
        database=db_name,
        table_name=table_name,
        perturbed_reaction=perturbed_reaction_no
    )
    pert_ind_len = CellSize.induction_length

    # calculate sensitivities
    sens_cj_speed = (pert_cj_speed - base_cj_speed) / \
                    (base_cj_speed * perturbation_fraction)
    sens_ind_len_west = (pert_ind_len['Westbrook'] -
                         base_ind_len['Westbrook']) / \
                        (base_ind_len['Westbrook'] * perturbation_fraction)
    sens_cell_size_west = (pert_cell_calcs['Westbrook'] -
                           base_cell_calcs['Westbrook']) / \
                          (base_cell_calcs['Westbrook'] * perturbation_fraction)
    sens_ind_len_gav = (pert_ind_len['Gavrikov'] -
                        base_ind_len['Gavrikov']) / \
                       (base_ind_len['Gavrikov'] * perturbation_fraction)
    sens_cell_size_gav = (pert_cell_calcs['Gavrikov'] -
                          base_cell_calcs['Gavrikov']) / \
                         (base_cell_calcs['Gavrikov'] * perturbation_fraction)
    sens_ind_len_ng = (pert_ind_len['Ng'] -
                       base_ind_len['Ng']) / \
                      (base_ind_len['Ng'] * perturbation_fraction)
    sens_cell_size_ng = (pert_cell_calcs['Ng'] -
                         base_cell_calcs['Ng']) / \
                        (base_cell_calcs['Ng'] * perturbation_fraction)

    with db_lock:
        current_table.store_perturbed_row(
            rxn_table_id=rxn_table_id,
            rxn_no=perturbed_reaction_no,
            rxn=gas.reaction_equation(perturbed_reaction_no),
            k_i=gas.forward_rate_constants[perturbed_reaction_no],
            cj_speed=pert_cj_speed,
            ind_len_west=pert_ind_len['Westbrook'],
            ind_len_gav=pert_ind_len['Gavrikov'],
            ind_len_ng=pert_ind_len['Ng'],
            cell_size_west=pert_cell_calcs['Westbrook'],
            cell_size_gav=pert_cell_calcs['Gavrikov'],
            cell_size_ng=pert_cell_calcs['Ng'],
            sens_cj_speed=sens_cj_speed,
            sens_ind_len_west=sens_ind_len_west,
            sens_ind_len_gav=sens_ind_len_gav,
            sens_ind_len_ng=sens_ind_len_ng,
            sens_cell_size_west=sens_cell_size_west,
            sens_cell_size_gav=sens_cell_size_gav,
            sens_cell_size_ng=sens_cell_size_ng,
            overwrite_existing=True
        )


def init(l):
    global db_lock
    db_lock = l


if __name__ == '__main__':
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
    import cantera as ct
    g = ct.Solution('gri30.cti')

    lock = Lock()
    p = Pool(initializer=init, initargs=(lock,))
    num_tests = 5
    p.starmap(
        perform_study,
        [
            [
                'gri30.cti',
                300,
                101325,
                1,
                'H2',
                'O2',
                'None',
                0,
                'None',
                1e-2,
                i,
            ] for i in range(g.n_reactions)
        ]
    )

    # perform_study(
    #     mech='gri30.cti',
    #     init_temp=300,
    #     init_press=101325,
    #     equivalence=1,
    #     fuel='H2',
    #     oxidizer='O2',
    #     diluent='None',
    #     diluent_mol_frac=0,
    #     inert='None',
    #     db_lock=lock,
    #     perturbation_fraction=1e-2,
    #     perturbed_reaction_no=1,
    # )
