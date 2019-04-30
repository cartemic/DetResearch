import funcs.database as db
import string
import random
import os
import sqlite3
import pytest
import inspect
from numpy import isclose, allclose
import uuid
import cantera as ct


def remove_stragglers():  # pragma: no cover
    stragglers = set(file for file in os.listdir('.') if '.sqlite' in file)
    for file in stragglers:
        os.remove(file)


def generate_db_name():  # pragma: no cover
    return str(uuid.uuid4()) + '.sqlite'


def bind(instance, func, as_name=None):  # pragma: no cover
    """
    https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method

    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


class TestDataBase:
    def test_list_all_tables(self):
        num_tables = 16
        num_chars = 16
        test_db = generate_db_name()
        table_names = [
            ''.join([random.choice(string.ascii_letters)
                     for _ in range(num_chars)])
            for _ in range(num_tables)
        ]
        with sqlite3.connect(test_db) as con:
            cur = con.cursor()
            for name in table_names:
                cur.execute(
                    """
                    CREATE TABLE {:s} (
                        test TEXT
                    );
                    """.format(name)
                )
        con.close()

        test_names = db.DataBase.list_all_tables(test_db)

        os.remove(test_db)

        assert all([
            test == good for test, good in zip(test_names, table_names)
        ])


# noinspection PyProtectedMember,PyUnresolvedReferences
class TestTable:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    class FakeTable:  # pragma: no cover
        """
        Fake table object for testing methods independently
        """
        def __init__(
                self,
                test_db,
                test_table_name,
                allow_create=True
        ):
            self.database = test_db
            self.table_name = test_table_name
            self.con = sqlite3.connect(self.database)
            self._build_query_str = db.Table._build_query_str
            if allow_create and self.table_name not in\
                    db.DataBase.list_all_tables(test_db):
                bind(self, db.Table._create_test_table)
                self._create_test_table()

        def __del__(self):
            self.con.commit()
            self.con.close()
            os.remove(self.database)

    def test__clean_bad_input(self):
        num_chars = 16
        dirty_string = ''.join([
            random.choice(string.ascii_uppercase + string.punctuation)
            for _ in range(num_chars)
        ])
        with pytest.raises(
            NameError,
            match='Table name must be entirely alphanumeric. '
                  'Underscores are allowed.'
        ):
            db.Table._clean(dirty_string)

    def test__clean_good_input(self):
        num_chars = 16
        good_stuff = set(string.ascii_lowercase)
        dirty_string = ''.join([
            random.choice(string.ascii_uppercase)
            for _ in range(num_chars)
        ])
        test_out = set(db.Table._clean(dirty_string))
        assert not test_out.intersection(good_stuff).difference(test_out)

    def test__build_query_str(self):
        # test empty, one input, and two inputs
        good_results = [
            'SELECT * FROM {:s};',
            'SELECT * FROM {:s} WHERE initial_temp = :initial_temp;',
            'SELECT * FROM {:s} WHERE initial_temp = :initial_temp '
            'AND fuel = :fuel;',
        ]
        sig = inspect.signature(db.Table._build_query_str)
        base_inputs = {item: None for item in sig.parameters.keys()}
        inputs = [
            base_inputs,
            {**base_inputs, 'initial_temp': 300},
            {**base_inputs, 'initial_temp': 300, 'fuel': 'CH4'}
        ]
        test_results = [db.Table._build_query_str(kw) for kw in inputs]
        # dict order is unreliable; use sorted() to compare strings
        assert all([
            ''.join(sorted(test)) == ''.join(sorted(good))
            for test, good in zip(test_results, good_results)
        ])

    def test_test_columns(self):
        num_cols = 16
        num_chars = 16
        test_db = generate_db_name()
        column_names = [
            ''.join([random.choice(string.ascii_letters)
                     for _ in range(num_chars)])
            for _ in range(num_cols)
        ]
        test_table_name = 'test_table'
        header = 'CREATE TABLE {:s} ('.format(test_table_name)
        footer = ' TEXT);'
        sep = ' TEXT, '

        with sqlite3.connect(test_db) as con:
            cur = con.cursor()
            cur.execute(
                header + sep.join(column_names) + footer
            )
        con.close()

        test_table = self.FakeTable(
            test_db,
            test_table_name,
            allow_create=False
            )
        bind(test_table, db.Table.test_columns)
        test_columns = test_table.test_columns()
        assert all([
            test == good for test, good in zip(test_columns, column_names)
        ])

    def test__create_test_table(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        FakeTable = self.FakeTable
        FakeTable.columns = db.Table.test_columns
        test_table = FakeTable(
            test_db,
            test_table_name,
            allow_create=True
            )
        actual_columns = set(test_table.columns())
        assert not set(db.Table._test_table_args).difference(actual_columns)

    def test__create_base_table(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        test_table.store_test_row(
            mechanism='gri30.cti',
            initial_temp=300,
            initial_press=101325,
            fuel='CH4',
            oxidizer='N2O',
            equivalence=1,
            diluent='N2',
            diluent_mol_frac=0.1,
            inert='None',
            cj_speed=1986.12354679687543,
            ind_len_west=1,
            ind_len_gav=2,
            ind_len_ng=3,
            cell_size_west=4,
            cell_size_gav=5,
            cell_size_ng=6
        )
        info = test_table.fetch_test_rows(
            mechanism='gri30.cti',
            initial_temp=300,
            initial_press=101325,
            fuel='CH4',
            oxidizer='N2O',
            equivalence=1,
            diluent='N2',
            diluent_mol_frac=0.1,
        )
        [rxn_id] = info['rxn_table_id']
        actual_columns = set(test_table.base_columns(rxn_id))
        assert not {'rxn_no', 'rxn', 'k_i'}.difference(actual_columns)

    def test__create_pert_table(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        test_table.store_test_row(
            mechanism='gri30.cti',
            initial_temp=300,
            initial_press=101325,
            fuel='CH4',
            oxidizer='N2O',
            equivalence=1,
            diluent='N2',
            diluent_mol_frac=0.1,
            inert='None',
            cj_speed=1986.12354679687543,
            ind_len_west=1,
            ind_len_gav=2,
            ind_len_ng=3,
            cell_size_west=4,
            cell_size_gav=5,
            cell_size_ng=6
        )
        info = test_table.fetch_test_rows(
            mechanism='gri30.cti',
            initial_temp=300,
            initial_press=101325,
            fuel='CH4',
            oxidizer='N2O',
            equivalence=1,
            diluent='N2',
            diluent_mol_frac=0.1,
        )
        [rxn_id] = info['rxn_table_id']
        actual_columns = set(test_table.pert_columns(rxn_id))
        assert not {
            'rxn_no',
            'rxn',
            'k_i',
            'cj_speed',
            'ind_len_west',
            'ind_len_gav',
            'ind_len_ng',
            'cell_size_west',
            'cell_size_gav',
            'cell_size_ng',
            'sens_cj_speed',
            'sens_ind_len_west',
            'sens_ind_len_gav',
            'sens_ind_len_ng',
            'sens_cell_size_west',
            'sens_cell_size_gav',
            'sens_cell_size_ng',
        }.difference(actual_columns)

    def test_store_and_check_existing_test(self):
        kwargs = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
        }
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
        )
        test_table.store_test_row(**{
            **kwargs,
            'cj_speed': 456,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        })
        assert test_table._check_existing_test(**kwargs)

    def test_fetch_rows_blank_table(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        test_rows = test_table.fetch_test_rows()
        assert all([
            not set(db.Table._test_table_args)
                .difference(set(test_rows.keys())),
            *[item == [] for item in test_rows.values()]
        ])

    def test_fetch_rows_single(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }
        test_table.store_test_row(**kwargs)
        test_rows = test_table.fetch_test_rows()
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key and 'rxn_table_id' not in key:
                if isinstance(kwargs[key], str) or isinstance(kwargs[key], int):
                    checks.append(kwargs[key] == value[0])
                else:
                    checks.append(isclose(kwargs[key], value[0]))
        assert all(checks)

    def test_fetch_rows_multiple(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs = [{
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }, {
            'mechanism': 'Mevel2017.cti',
            'initial_temp': 550,
            'initial_press': 2*101325,
            'fuel': 'H2',
            'oxidizer': 'O2',
            'equivalence': 1.125,
            'diluent': 'None',
            'diluent_mol_frac': 0,
            'inert': 'N2',
            'cj_speed': 2112,
            'ind_len_west': 2.1,
            'ind_len_gav': 2.2,
            'ind_len_ng': 2.3,
            'cell_size_west': 2.4,
            'cell_size_gav': 2.5,
            'cell_size_ng': 2.6
        }]
        for kw in kwargs:
            test_table.store_test_row(**kw)
        test_rows = test_table.fetch_test_rows()
        checks = []
        good_answer = {
            key: [value, kwargs[1][key]] for key, value in kwargs[0].items()
        }
        for key, value in test_rows.items():
            if 'date' not in key and 'rxn_table_id' not in key:
                if (isinstance(kwargs[0][key], str)
                        or isinstance(kwargs[0][key], int)):
                    for good, test in zip(good_answer[key], test_rows[key]):
                        checks.append(test == good)
                else:
                    checks.append(allclose(good_answer[key], value))
        assert all(checks)

    def test__update_test_row(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }
        kwargs_repl = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 2112,
            'ind_len_west': 2.1,
            'ind_len_gav': 2.2,
            'ind_len_ng': 2.3,
            'cell_size_west': 2.4,
            'cell_size_gav': 2.5,
            'cell_size_ng': 2.6
        }
        test_table.store_test_row(**kwargs_init)
        test_table._update_test_row(**kwargs_repl)
        test_rows = test_table.fetch_test_rows()
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key and 'rxn_table_id' not in key:
                if (isinstance(kwargs_repl[key], str)
                        or isinstance(kwargs_repl[key], int)):
                    checks.append(kwargs_repl[key] == value[0])
                else:
                    checks.append(isclose(kwargs_repl[key], value[0]))
        assert all(checks)

    @pytest.mark.filterwarnings('ignore')
    def test_store_test_row_update(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }
        kwargs_repl = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 2112,
            'ind_len_west': 2.1,
            'ind_len_gav': 2.2,
            'ind_len_ng': 2.3,
            'cell_size_west': 2.4,
            'cell_size_gav': 2.5,
            'cell_size_ng': 2.6
        }
        test_table.store_test_row(**kwargs_init)
        test_table.store_test_row(**kwargs_repl, overwrite_existing=True)
        test_table.store_test_row(**kwargs_init, overwrite_existing=False)
        test_rows = test_table.fetch_test_rows()
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key and 'rxn_table_id' not in key:
                if (isinstance(kwargs_repl[key], str)
                        or isinstance(kwargs_repl[key], int)):
                    checks.append(kwargs_repl[key] == value[0])
                else:
                    checks.append(isclose(kwargs_repl[key], value[0]))
        assert all(checks)

    @pytest.mark.filterwarnings('ignore')
    def test_store_pert_row_update(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        test_kwargs = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }
        pert_kwargs_init = {
            'rxn_no': 100,
            'rxn': '3 food + 1 stomach => 1 poop',
            'k_i': 1.2e12,
            'cj_speed': 1234.56789,
            'ind_len_west': 1.1,
            'ind_len_gav': 2.2,
            'ind_len_ng': 3.3,
            'cell_size_west': 4.4,
            'cell_size_gav': 5.5,
            'cell_size_ng': 6.6,
            'sens_cj_speed': 7.7,
            'sens_ind_len_west': 8.8,
            'sens_ind_len_gav': 9.9,
            'sens_ind_len_ng': 0.0,
            'sens_cell_size_west': 1.1,
            'sens_cell_size_gav': 2.2,
            'sens_cell_size_ng': 3.3,
        }
        pert_kwargs_repl = {
            'rxn_no': 100,
            'rxn': '3 food + 1 stomach => 1 poop',
            'k_i': 1.2e12,
            'cj_speed': 1234.56789,
            'ind_len_west': 11.1,
            'ind_len_gav': 12.2,
            'ind_len_ng': 13.3,
            'cell_size_west': 14.4,
            'cell_size_gav': 15.5,
            'cell_size_ng': 16.6,
            'sens_cj_speed': 17.7,
            'sens_ind_len_west': 18.8,
            'sens_ind_len_gav': 19.9,
            'sens_ind_len_ng': 10.0,
            'sens_cell_size_west': 11.1,
            'sens_cell_size_gav': 12.2,
            'sens_cell_size_ng': 13.3,
        }
        table_id = test_table.store_test_row(**test_kwargs)
        test_table.store_perturbed_row(
            rxn_table_id=table_id,
            **pert_kwargs_init
        )
        test_table.store_perturbed_row(
            rxn_table_id=table_id,
            overwrite_existing=True,
            **pert_kwargs_repl,
        )
        test_table.store_perturbed_row(
            rxn_table_id=table_id,
            overwrite_existing=False,
            **pert_kwargs_init,
        )
        test_rows = test_table.fetch_pert_table(
            rxn_table_id=table_id,
            **pert_kwargs_repl
        )
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key:
                if (isinstance(pert_kwargs_repl[key], str)
                        or isinstance(pert_kwargs_repl[key], int)):
                    checks.append(pert_kwargs_repl[key] == value[0])
                else:
                    checks.append(isclose(pert_kwargs_repl[key], value[0]))
        assert all(checks)

    def test_store_base_table(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }
        rxn_table_id = test_table.store_test_row(**kwargs_init)

        gas = ct.Solution(kwargs_init['mechanism'])
        gas.TP = kwargs_init['initial_temp'], kwargs_init['initial_press']
        gas.set_equivalence_ratio(
            kwargs_init['equivalence'],
            kwargs_init['fuel'],
            kwargs_init['oxidizer']
        )
        test_table.store_base_rxn_table(
            rxn_table_id,
            gas
        )

        test_rows = test_table.fetch_base_rxn_table(rxn_table_id)
        checks = [gas.n_reactions == len(test_rows)]
        for row in test_rows:
            current_tests = [
                gas.reaction_equation(row[0]) == row[1],
                isclose(gas.forward_rate_constants[row[0]], row[2])
            ]
            checks.append(current_tests)

        assert all(checks)

    def test_fetch_single_base_rxn(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'equivalence': 1,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'inert': 'None',
            'cj_speed': 1986.12354679687543,
            'ind_len_west': 1,
            'ind_len_gav': 2,
            'ind_len_ng': 3,
            'cell_size_west': 4,
            'cell_size_gav': 5,
            'cell_size_ng': 6
        }
        rxn_table_id = test_table.store_test_row(**kwargs_init)

        gas = ct.Solution(kwargs_init['mechanism'])
        gas.TP = kwargs_init['initial_temp'], kwargs_init['initial_press']
        gas.set_equivalence_ratio(
            kwargs_init['equivalence'],
            kwargs_init['fuel'],
            kwargs_init['oxidizer']
        )
        test_table.store_base_rxn_table(
            rxn_table_id,
            gas
        )

        checks = []
        rates = gas.forward_rate_constants
        for idx in range(gas.n_reactions):
            test_rxn = test_table.fetch_single_base_rxn(
                rxn_table_id,
                idx
            )
            checks.append([
                test_rxn[0] == idx,
                test_rxn[1] == gas.reaction_equation(idx),
                test_rxn[2] == rates[idx]
            ])


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    try:
        subprocess.check_call(
            'pytest test_database.py -vv --noconftest --cov --cov-report html'
        )
    except subprocess.CalledProcessError as e:
        # clean up in case of an unexpected error cropping up
        remove_stragglers()
        raise e
