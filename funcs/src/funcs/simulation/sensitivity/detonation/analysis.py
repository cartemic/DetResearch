import dataclasses

import sdtoolbox as sd
from funcs.simulation import cell_size
from funcs.simulation.sensitivity.gas import build as build_gas

from . import database as db


def initialize_study(db_path: str, test_conditions: db.TestConditions, max_step_znd: float) -> db.TestConditions:
    """
    Creates a new row and performs CJ speed calculations, both as needed, and returns updated TestConditions
    """
    database = db.DataBase(path=db_path)
    if test_conditions.test_id is None:
        test_conditions = database.new_test(test_conditions=test_conditions)
    else:
        # in case we don't want to recalculate CJ speed
        test_conditions = database.test_conditions_table.fetch_row(test_conditions.test_id)

    if test_conditions.needs_cj_calc():
        gas = build_gas(
            test_conditions.mechanism,
            test_conditions.initial_temp,
            test_conditions.initial_press,
            test_conditions.equivalence,
            test_conditions.fuel,
            test_conditions.oxidizer,
            test_conditions.diluent,
            test_conditions.diluent_mol_frac,
        )
        test_conditions.cj_speed = sd.postshock.CJspeed(
            P1=test_conditions.initial_press,
            T1=test_conditions.initial_temp,
            q=gas.mole_fraction_dict(),
            mech=test_conditions.mechanism,
        )
        database.test_conditions_table.update_row(test_conditions=test_conditions)

    # This is the base cell size, which is used for all sensitivity calculations and is therefore required before any
    # further cell size calculations occur.
    if test_conditions.needs_cell_size_calc():
        base_cell_calcs = cell_size.calculate(
            mechanism=test_conditions.mechanism,
            cj_speed=test_conditions.cj_speed,
            initial_temp=test_conditions.initial_temp,
            initial_press=test_conditions.initial_press,
            fuel=test_conditions.fuel,
            oxidizer=test_conditions.oxidizer,
            equivalence=test_conditions.equivalence,
            diluent=test_conditions.diluent,
            diluent_mol_frac=test_conditions.diluent_mol_frac,
            max_step_znd=max_step_znd,
        )
        test_conditions.ind_len_gav = base_cell_calcs.induction_length.gavrikov
        test_conditions.ind_len_ng = base_cell_calcs.induction_length.ng
        test_conditions.ind_len_west = base_cell_calcs.induction_length.westbrook
        test_conditions.cell_size_gav = base_cell_calcs.cell_size.gavrikov
        test_conditions.cell_size_ng = base_cell_calcs.cell_size.ng
        test_conditions.cell_size_west = base_cell_calcs.cell_size.westbrook
        database.test_conditions_table.update_row(test_conditions=test_conditions)

    return test_conditions


def perform_study(
    test_conditions: db.TestConditions,
    perturbation_fraction: float,
    perturbed_reaction_no: int,
    db_path: str,
    max_step_znd: float,
    overwrite_existing: bool,
):
    database = db.DataBase(path=db_path)

    # I could definitely be doing a much better job of cache invalidation, but more than anything I want this to run,
    # and this isn't exactly production code so... meh.
    if test_conditions.test_id is None:
        raise ValueError("TestConditions must be initiated prior to study initiation")
    if not database.test_conditions_table.test_exists(test_id=test_conditions.test_id):
        raise db.DatabaseError(f"Test conditions were given for a nonexistent row: {test_conditions}")

    if overwrite_existing:
        # calculate, then overwrite or insert
        perturbed_results = calculate_perturbed_cell_size_and_sensitivity(
            base_test_conditions=test_conditions,
            perturbed_reaction_no=perturbed_reaction_no,
            max_step_znd=max_step_znd,
            perturbation_fraction=perturbation_fraction,
        )
        if database.perturbed_results_table.row_exists(test_conditions.test_id, perturbed_reaction_no):
            database.perturbed_results_table.update_row(perturbed_results)
        else:
            database.perturbed_results_table.insert_new_row(perturbed_results)
    elif not database.perturbed_results_table.row_exists(test_conditions.test_id, perturbed_reaction_no):
        perturbed_results = calculate_perturbed_cell_size_and_sensitivity(
            base_test_conditions=test_conditions,
            perturbed_reaction_no=perturbed_reaction_no,
            max_step_znd=max_step_znd,
            perturbation_fraction=perturbation_fraction,
        )
        database.perturbed_results_table.insert_new_row(perturbed_results)


@dataclasses.dataclass(frozen=True)
class SensitivityResults:
    induction_length: cell_size.ModelResults
    cell_size: cell_size.ModelResults


def calculate_sensitivities(
    base_test_conditions: db.TestConditions,
    perturbed_results: cell_size.CellSizeResults,
    perturbation_fraction: float,
):
    def calculate_sensitivity(base: float, perturbed: float):
        return (perturbed - base) / (base * perturbation_fraction)

    return SensitivityResults(
        induction_length=cell_size.ModelResults(
            gavrikov=calculate_sensitivity(
                base=base_test_conditions.cell_size_gav,
                perturbed=perturbed_results.cell_size.gavrikov,
            ),
            ng=calculate_sensitivity(
                base=base_test_conditions.cell_size_ng,
                perturbed=perturbed_results.cell_size.ng,
            ),
            westbrook=calculate_sensitivity(
                base=base_test_conditions.cell_size_west,
                perturbed=perturbed_results.cell_size.westbrook,
            ),
        ),
        cell_size=cell_size.ModelResults(
            gavrikov=calculate_sensitivity(
                base=base_test_conditions.ind_len_gav,
                perturbed=perturbed_results.induction_length.gavrikov,
            ),
            ng=calculate_sensitivity(
                base=base_test_conditions.ind_len_ng,
                perturbed=perturbed_results.induction_length.ng,
            ),
            westbrook=calculate_sensitivity(
                base=base_test_conditions.ind_len_west,
                perturbed=perturbed_results.induction_length.westbrook,
            ),
        ),
    )


def calculate_perturbed_cell_size_and_sensitivity(
    base_test_conditions: db.TestConditions,
    perturbed_reaction_no: int,
    max_step_znd: float,
    perturbation_fraction: float,
) -> db.PerturbedResults:
    if perturbed_reaction_no is None:
        raise ValueError("Cannot calculate perturbed cell sizes with perturbed_reaction = None")

    pert_cell_calcs = cell_size.calculate(
        mechanism=base_test_conditions.mechanism,
        cj_speed=base_test_conditions.cj_speed,
        initial_temp=base_test_conditions.initial_temp,
        initial_press=base_test_conditions.initial_press,
        fuel=base_test_conditions.fuel,
        oxidizer=base_test_conditions.oxidizer,
        equivalence=base_test_conditions.equivalence,
        diluent=base_test_conditions.diluent,
        diluent_mol_frac=base_test_conditions.diluent_mol_frac,
        perturbed_reaction=perturbed_reaction_no,
        max_step_znd=max_step_znd,
    )
    sensitivity = calculate_sensitivities(
        base_test_conditions=base_test_conditions,
        perturbed_results=pert_cell_calcs,
        perturbation_fraction=perturbation_fraction,
    )

    return db.PerturbedResults(
        test_id=base_test_conditions.test_id,
        rxn_no=perturbed_reaction_no,
        perturbation_fraction=perturbation_fraction,
        rxn=pert_cell_calcs.reaction_equation,  # won't be None if perturbed_reaction_no is not None
        k_i=pert_cell_calcs.k_i,  # won't be None if perturbed_reaction_no is not None
        ind_len_west=pert_cell_calcs.induction_length.westbrook,
        ind_len_gav=pert_cell_calcs.induction_length.gavrikov,
        ind_len_ng=pert_cell_calcs.induction_length.ng,
        cell_size_west=pert_cell_calcs.cell_size.westbrook,
        cell_size_gav=pert_cell_calcs.cell_size.gavrikov,
        cell_size_ng=pert_cell_calcs.cell_size.ng,
        sens_ind_len_west=sensitivity.induction_length.westbrook,
        sens_ind_len_gav=sensitivity.induction_length.gavrikov,
        sens_ind_len_ng=sensitivity.induction_length.ng,
        sens_cell_size_west=sensitivity.cell_size.westbrook,
        sens_cell_size_gav=sensitivity.cell_size.gavrikov,
        sens_cell_size_ng=sensitivity.cell_size.ng,
    )


# ruff: noqa: PLW0603
def init(lock):
    # noinspection PyGlobalUndefined
    global db_lock
    db_lock = lock
