import os
from tempfile import NamedTemporaryFile

import cantera as ct
import deepdiff

import funcs.simulation.sensitivity.detonation.database as db
from funcs.dir import MECH_DIR

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
# noinspection PyUnresolvedReferences
ct.add_directory(MECH_DIR)


def test_database():
    with NamedTemporaryFile() as db_file:
        db.DataBase(path=db_file.name)


def test_base_reaction_table():
    with NamedTemporaryFile() as db_file:
        test_db = db.DataBase(path=db_file.name)
        table = test_db.base_rxn_table

        mechanism = "gri30.yaml"
        assert not table.has_mechanism(mechanism=mechanism)

        table.store_all_reactions(gas=ct.Solution(mechanism), mechanism=mechanism)
        table.cur.execute(
            f"select count(*) n_rxns from {table.name} where mechanism = :mech",
            {"mech": mechanism},
        )
        result = table.cur.fetchone()
        assert result["n_rxns"] == 325
        assert table.has_mechanism(mechanism=mechanism)


def test_test_conditions_table():
    with NamedTemporaryFile() as db_file:
        test_db = db.DataBase(path=db_file.name)
        table = test_db.test_conditions_table

        test_conditions = db.TestConditions(
            mechanism="gri30.yaml",
            initial_temp=300.0,
            initial_press=101325.0,
            fuel="H2",
            oxidizer="O2",
            equivalence=1.0,
            diluent="CO2",
            diluent_mol_frac=0.2,
        )
        assert not table.test_exists(test_id=1)

        test_conditions = table.insert_new_row(test_conditions=test_conditions)
        assert table.test_exists(test_id=test_conditions.test_id)

        test_conditions.cj_speed = 1234.5
        test_conditions.ind_len_west = 1e-1
        test_conditions.ind_len_gav = 1e-2
        test_conditions.ind_len_ng = 1e-3
        test_conditions.cell_size_west = 2e-1
        test_conditions.cell_size_gav = 2e-2
        test_conditions.cell_size_ng = 2e-3
        table.update_row(test_conditions=test_conditions)
        row = table.fetch_row(test_id=test_conditions.test_id)
        assert deepdiff.DeepDiff(test_conditions, row) == {}
        rows = table.fetch_rows(test_ids=[test_conditions.test_id])
        assert deepdiff.DeepDiff(rows, [test_conditions]) == {}


def test_perturbed_results_table():
    with NamedTemporaryFile() as db_file:
        test_db = db.DataBase(path=db_file.name)
        table = test_db.perturbed_results_table

        perturbed_results = db.PerturbedResults(
            test_id=1,
            rxn_no=0,
            perturbation_fraction=1e-6,
            rxn="C2 => O + O",
            k_i=19.7,
            ind_len_west=1e-1,
            ind_len_gav=1e-2,
            ind_len_ng=1e-3,
            cell_size_west=2e-1,
            cell_size_gav=2e-2,
            cell_size_ng=2e-3,
            sens_cell_size_west=1.2,
            sens_cell_size_gav=3.4,
            sens_cell_size_ng=5.6,
            sens_ind_len_west=2.1,
            sens_ind_len_gav=4.3,
            sens_ind_len_ng=6.5,
        )
        assert not table.row_exists(test_id=perturbed_results.test_id, rxn_no=perturbed_results.rxn_no)

        table.insert_new_row(perturbed_results=perturbed_results)
        assert table.row_exists(test_id=perturbed_results.test_id, rxn_no=perturbed_results.rxn_no)

        # This is done on purpose
        # ruff: noqa: PLW2901
        for item in (
            perturbed_results.perturbation_fraction,
            perturbed_results.k_i,
            perturbed_results.ind_len_west,
            perturbed_results.ind_len_gav,
            perturbed_results.ind_len_ng,
            perturbed_results.cell_size_west,
            perturbed_results.cell_size_gav,
            perturbed_results.cell_size_ng,
            perturbed_results.sens_cell_size_west,
            perturbed_results.sens_cell_size_gav,
            perturbed_results.sens_cell_size_ng,
            perturbed_results.sens_ind_len_west,
            perturbed_results.sens_ind_len_gav,
            perturbed_results.sens_ind_len_ng,
        ):
            item *= 2

        table.update_row(perturbed_results=perturbed_results)
        row = table.fetch_row(test_id=perturbed_results.test_id, rxn_no=perturbed_results.rxn_no)
        assert deepdiff.DeepDiff(perturbed_results, row) == {}
        rows = table.fetch_rows(test_id=perturbed_results.test_id)
        assert deepdiff.DeepDiff(rows, [perturbed_results]) == {}
